"""M2: incremental KV-cache, and the generation loop that knows when to stop using it.

The cache only speeds things up while the window still fits in block_size. Past
that point, a KV-cache over LEARNED ABSOLUTE positions (no RoPE) has no state
worth reusing between consecutive steps: the window slides one token per step and
no position ever repeats. That is not a bug, it is a property of the model.

Concretely, at block_size=4 with sequence t0..t4, the step that crops turns the
window [t0,t1,t2,t3] into [t1,t2,t3,t4]. Every surviving token shifts down one
slot -- t1 goes from wpe[1] to wpe[0], t2 from wpe[2] to wpe[1], and so on -- so
every cached K,V is attached to the wrong positional embedding. On top of that,
t1 used to attend to {t0,t1} and now attends to {t1} alone, so its deeper-layer
K,V would be wrong even if the position were patched. Rebuilding K,V for the
whole window IS a full forward, so "invalidate and repopulate every step" pays
for a full forward plus the bookkeeping of a cache nobody reads.

So: cache while the window fits, then hand the rest of the generation to the
no-cache M1 path and never come back.

Cache format: a list with one ``(K, V)`` pair per layer, each shaped
``(n_head, t, d_head)`` where ``t`` is the number of tokens cached so far.
``None`` means empty. ``generate_with_cache`` treats it as opaque -- it only ever
passes it back to ``forward_step_fn`` -- so a stub can use any object it likes.
"""

import numpy as np

from .forward import layer_norm, mlp, softmax
from .generate import generate as _generate_without_cache


def cache_length(cache):
    """Number of token positions held in the cache. ``None`` is an empty cache."""
    if cache is None:
        return 0
    k_first_layer, _v = cache[0]
    return k_first_layer.shape[1]


def forward_step(token_id, position, cache, params, n_head):
    """One-token incremental forward. Returns ``(logits (vocab,), new_cache)``.

    ``position`` indexes ``wpe`` directly and must be the cache length BEFORE this
    token is inserted -- the caller owns that invariant.

    No causal mask appears anywhere below, and that is correct rather than an
    omission: every entry already in the cache is strictly in this token's past,
    and the token attends to itself. There is no future to hide.
    """
    x = params["wte.weight"][token_id] + params["wpe.weight"][position]  # (C,)
    C = x.shape[0]
    d_head = C // n_head
    n_layer = sum(1 for k in params if k.endswith("ln_1.weight"))

    new_cache = []
    for i in range(n_layer):
        pre = f"blocks.{i}."
        attn = pre + "attn."

        h = layer_norm(x, params[pre + "ln_1.weight"], params[pre + "ln_1.bias"])

        def project(name, h=h, attn=attn):
            return (h @ params[attn + name + ".weight"] + params[attn + name + ".bias"]).reshape(
                n_head, 1, d_head
            )

        q = project("q_proj")
        k = project("k_proj")
        v = project("v_proj")

        if cache is None:
            K, V = k, v
        else:
            k_prev, v_prev = cache[i]
            K = np.concatenate([k_prev, k], axis=1)  # old first, new last
            V = np.concatenate([v_prev, v], axis=1)
        new_cache.append((K, V))

        att = (q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)  # (n_head, 1, t+1)
        att = softmax(att, axis=-1)
        y = att @ V  # (n_head, 1, d_head)
        y = y.transpose(1, 0, 2).reshape(C)  # head0[0:64], head1[64:128], ...

        x = x + (y @ params[attn + "c_proj.weight"] + params[attn + "c_proj.bias"])
        x = x + mlp(
            layer_norm(x, params[pre + "ln_2.weight"], params[pre + "ln_2.bias"]),
            params,
            pre + "mlp.",
        )

    x = layer_norm(x, params["ln_f.weight"], params["ln_f.bias"])
    return x @ params["wte.weight"].T, new_cache  # tied, no bias


def generate_with_cache(
    forward_step_fn, forward_full_fn, idx, max_new_tokens, eos_id, block_size, greedy=True
):
    """Greedy generation: incremental while the window fits, full recompute after.

    ``forward_step_fn(token, position, cache) -> (logits, new_cache)`` drives the
    cached regime. On the first step where the sequence would exceed
    ``block_size``, the cache is dropped and the remaining budget is handed to the
    no-cache path from :mod:`engine.generate`, which owns the crop. The cache is
    never rebuilt -- see the module docstring for why there would be nothing to
    rebuild it from.

    Returns the list of newly generated ids, prompt excluded. EOS stops the loop
    without being appended or emitted, exactly as in M1.

    ASYMMETRY, deliberate and scoped: M4 gave the no-cache `engine.generate.generate`
    real sampling (temperature / top-k / top-p / injected rng), and this path did NOT
    get it. `greedy=False` still raises here. Anyone reaching for sampled generation
    must use the M1 path today — including through this function's own post-crop
    fallback, which would otherwise silently sample in the tail and take the argmax in
    the head. (Same posture as the `cross_entropy` rank asymmetry recorded in M3.)
    """
    if not greedy:
        raise NotImplementedError(
            "generate_with_cache is greedy-only; sampling lives in engine.generate.generate"
        )

    idx = np.asarray(idx, dtype=np.int64)
    emitted = []
    cache = None
    cached_len = 0

    for step in range(max_new_tokens):
        if idx.shape[0] > block_size:
            # The window has to slide from here on, so nothing in the cache is
            # reusable. Drop it and let the M1 path finish the generation.
            cache = None
            emitted.extend(
                _generate_without_cache(
                    forward_full_fn,
                    idx,
                    max_new_tokens=max_new_tokens - step,
                    eos_id=eos_id,
                    block_size=block_size,
                )
            )
            return emitted

        for position in range(cached_len, idx.shape[0]):
            logits, cache = forward_step_fn(int(idx[position]), position, cache)
        cached_len = idx.shape[0]

        next_id = np.argmax(np.asarray(logits), axis=-1).item()

        if next_id == eos_id:
            return emitted  # stop WITHOUT appending and WITHOUT emitting.

        idx = np.append(idx, next_id)
        emitted.append(next_id)

    return emitted
