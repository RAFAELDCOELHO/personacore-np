"""M1: autoregressive generation, one full forward per step. No KV-cache, no batch.

M4 added sampling here: `greedy=False` now draws through
`engine.sampling.sample_next_token` instead of raising. The greedy branch is
untouched and still takes a plain argmax without consulting temperature, top-k,
top-p or the rng.

Every step recomputes the whole window from scratch — deliberately slow, and
exactly what PersonaCore does today (`generation/core.py:52-71`; the cache was
measured there and deferred).

`forward_fn` is INJECTED rather than hardcoded to `gpt_forward`. That splits the
loop's CONTROL (crop, last-position selection, argmax, EOS-stop) from the
CALCULATION, so the control is testable with synthetic stubs — without the
checkpoint, and without depending on the model happening to emit an EOS (per the
frozen fixture, it does not, in 300 tokens).

    forward_fn(window) -> logits, shaped (T, vocab) or (1, T, vocab)

Only the last row matters. On the real path:

    forward_fn = lambda window: gpt_forward(window, params, n_head=6)

`idx` is a 1-D array of ids, matching `engine.forward.gpt_forward`. There is no
batch dimension anywhere in this engine.
"""

import numpy as np

from .sampling import sample_next_token


def generate(
    forward_fn,
    idx,
    max_new_tokens,
    eos_id,
    block_size,
    greedy=True,
    temperature=1.0,
    top_k=None,
    top_p=None,
    rng=None,
):
    """Return the list of newly generated ids. The prompt is not included.

    The context is cropped to the last `block_size` ids BEFORE every forward
    call, including the first — a prompt longer than `block_size` is cropped on
    step one, never passed through whole.

    On EOS the loop returns immediately: the EOS id is neither appended to the
    running context nor added to the returned list, so it can never surface in
    the output.

    `greedy=True` takes a plain argmax and NEVER touches temperature, top-k,
    top-p or the rng — the short-circuit sits before the pipeline, exactly as in
    PersonaCore's `next_token`. `greedy=False` runs the locked chain
    temperature -> top-k -> top-p -> softmax -> draw.

    The rng is created ONCE here rather than per step: a fresh unseeded Generator
    inside the loop would reseed from the OS every token, which is both slower and
    impossible to reproduce even when the caller wanted the default.
    """
    idx = np.asarray(idx, dtype=np.int64)
    emitted = []

    if not greedy and rng is None:
        rng = np.random.default_rng()

    for _ in range(max_new_tokens):
        idx_cond = idx[-block_size:] if idx.shape[0] > block_size else idx

        logits = np.asarray(forward_fn(idx_cond))
        last_logits = logits[..., -1, :]
        if greedy:
            next_id = np.argmax(last_logits, axis=-1).item()
        else:
            next_id = sample_next_token(
                last_logits, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng
            )

        if next_id == eos_id:
            return emitted  # stop WITHOUT appending and WITHOUT emitting.

        idx = np.append(idx, next_id)
        emitted.append(next_id)

    return emitted
