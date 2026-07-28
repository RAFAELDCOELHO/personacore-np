"""Pure-NumPy forward pass of the PersonaCore GPT. No autograd, no cache, no batch.

M0: one sequence at a time, `(T,)` of ids in, `(T, vocab)` of logits out. There
is no computation graph, no state between calls, no KV-cache — every call
recomputes the whole window, which is exactly what PersonaCore does today (the
cache was measured and deferred there as well).

Convention: Linear weights in (in, out), the forward computes `x @ W + b`. The
transpose lives in the loader (`engine.weights`), never here.

Internal arrays are (T, C); attention works in (n_head, T, d_head). No batch
dimension: the checkpoint runs one sequence at a time in on-device inference,
and one fewer dimension is one fewer dimension to get wrong.
"""

import numpy as np

LN_EPS = 1e-5
GELU_COEFF = 0.044715


def layer_norm(x, weight, bias, eps=LN_EPS):
    """POPULATION variance (ddof=0), eps INSIDE the sqrt — same as nn.LayerNorm."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def gelu(x):
    """GELU tanh-approx (PyTorch ``approximate="tanh"``), NOT the erf version."""
    inner = np.sqrt(2.0 / np.pi) * (x + GELU_COEFF * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def softmax(x, axis=-1):
    """Stable softmax: subtract the max before the exp (the same thing PyTorch does).

    Masked positions arrive as -inf; `-inf - max` stays -inf and `exp` gives
    exactly 0, so the mass vanishes without producing NaN. The diagonal is never
    masked, so no row is entirely -inf.
    """
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(x, p, prefix, n_head):
    """Multi-head causal self-attention. x: (T, C) -> (T, C)."""
    T, C = x.shape
    d_head = C // n_head

    def project(name):
        return x @ p[prefix + name + ".weight"] + p[prefix + name + ".bias"]

    def split_heads(a):
        return a.reshape(T, n_head, d_head).transpose(1, 0, 2)  # (n_head, T, d_head)

    q = split_heads(project("q_proj"))
    k = split_heads(project("k_proj"))
    v = split_heads(project("v_proj"))

    # Scale 1/sqrt(d_head) AFTER the matmul, and d_head (64), not n_embd (384).
    att = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_head)

    # Mask BEFORE the softmax: the future becomes -inf and comes out zeroed by exp.
    causal = np.tril(np.ones((T, T), dtype=bool))
    att = np.where(causal, att, -np.inf)
    att = softmax(att, axis=-1)

    y = att @ v  # (n_head, T, d_head)
    y = y.transpose(1, 0, 2).reshape(T, C)  # head0[0:64], head1[64:128], ...
    return y @ p[prefix + "c_proj.weight"] + p[prefix + "c_proj.bias"]


def mlp(x, p, prefix):
    """4x feed-forward with tanh-approx GELU. x: (T, C) -> (T, C)."""
    h = x @ p[prefix + "fc_in.weight"] + p[prefix + "fc_in.bias"]
    h = gelu(h)
    return h @ p[prefix + "fc_out.weight"] + p[prefix + "fc_out.bias"]


def block(x, p, i, n_head):
    """Pre-norm block: the norm sits INSIDE the residual branch, the skip carries raw x."""
    pre = f"blocks.{i}."
    x = x + attention(
        layer_norm(x, p[pre + "ln_1.weight"], p[pre + "ln_1.bias"]), p, pre + "attn.", n_head
    )
    x = x + mlp(layer_norm(x, p[pre + "ln_2.weight"], p[pre + "ln_2.bias"]), p, pre + "mlp.")
    return x


def gpt_forward(idx, params, n_head=6):
    """(T,) of ids -> (T, vocab) of logits.

    The position is the index WITHIN the window (0..T-1), not the absolute
    position in a longer sequence — that is what `wpe[:T]` encodes, and it is
    what PersonaCore does after cropping the context to block_size.
    """
    idx = np.asarray(idx)
    T = idx.shape[0]

    x = params["wte.weight"][idx] + params["wpe.weight"][:T]

    n_layer = sum(1 for k in params if k.endswith("ln_1.weight"))
    for i in range(n_layer):
        x = block(x, params, i, n_head)

    x = layer_norm(x, params["ln_f.weight"], params["ln_f.bias"])
    return x @ params["wte.weight"].T  # tied, no bias


def cross_entropy(logits, targets):
    """Mean CE over the tokens — the same as F.cross_entropy(logits, targets)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(-log_probs[np.arange(len(targets)), targets].mean())
