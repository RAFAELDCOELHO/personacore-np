"""M6: the forward pass on MLX (Metal GPU) — a 1:1 port of engine/forward.py.

SAME structure, SAME operation order, SAME (in, out) weight convention as the
NumPy engine; the only intended differences are the array library and the
dtype. Anything else that differs is a porting bug, and the parity tests in
tests/test_forward_mlx.py exist to catch exactly that category.

dtype: FLOAT32 everywhere. Metal has no float64 (measured: any f64 GPU op
raises `ValueError: float64 is not supported on the GPU`), and forcing the CPU
stream would defeat the milestone. The cast is lossless for the weights — the
checkpoint is fp32 — and the f32 tolerance is derived in the test module.

MLX is LAZY: these functions build a graph; nothing computes until `mx.eval`
(or a materialization like `np.asarray`) forces it. Callers timing this
forward must force evaluation inside the timed region.

This module imports mlx.core unconditionally and is itself imported ONLY
behind a guard (pytest.importorskip in the test module) — the NumPy engine
never touches it, so the suite still runs/skips cleanly without mlx installed.
"""

import mlx.core as mx
import numpy as np

LN_EPS = 1e-5
GELU_COEFF = 0.044715


def to_mlx_params(np_params):
    """NumPy f64 params dict -> mx.array float32 dict, same keys, same layout.

    The f64 values are exact fp32 casts of the checkpoint, so the f32
    conversion is bit-exact — the GPU sees the very weights PyTorch trained.
    """
    return {k: mx.array(v.astype(np.float32)) for k, v in np_params.items()}


def layer_norm(x, weight, bias, eps=LN_EPS):
    """POPULATION variance (ddof=0), eps INSIDE the sqrt — same as nn.LayerNorm."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps) * weight + bias


def gelu(x):
    """GELU tanh-approx (PyTorch ``approximate="tanh"``), NOT the erf version."""
    inner = mx.sqrt(mx.array(2.0 / np.pi)) * (x + GELU_COEFF * x**3)
    return 0.5 * x * (1.0 + mx.tanh(inner))


def softmax(x, axis=-1):
    """Stable softmax: subtract the max before the exp. -inf elements -> exactly 0."""
    z = x - mx.max(x, axis=axis, keepdims=True)
    e = mx.exp(z)
    return e / mx.sum(e, axis=axis, keepdims=True)


def attention(x, p, prefix, n_head):
    """Multi-head causal self-attention. x: (..., T, C) -> (..., T, C).

    Rank-generic over leading axes exactly like the NumPy version: swapaxes
    with negative indices, batch as a broadcast dimension of ``@``.
    """
    *lead, T, C = x.shape
    d_head = C // n_head

    def project(name):
        return x @ p[prefix + name + ".weight"] + p[prefix + name + ".bias"]

    def split_heads(a):
        # (..., T, C) -> (..., T, n_head, d_head) -> (..., n_head, T, d_head)
        return mx.swapaxes(a.reshape(*lead, T, n_head, d_head), -3, -2)

    q = split_heads(project("q_proj"))
    k = split_heads(project("k_proj"))
    v = split_heads(project("v_proj"))

    # Scale 1/sqrt(d_head) AFTER the matmul, and d_head (64), not n_embd (384).
    att = (q @ mx.swapaxes(k, -1, -2)) / mx.sqrt(mx.array(float(d_head)))

    # Mask BEFORE the softmax: the future becomes -inf and comes out zeroed by exp.
    causal = mx.tri(T, dtype=mx.bool_)
    att = mx.where(causal, att, mx.array(-float("inf")))
    att = softmax(att, axis=-1)

    y = att @ v  # (..., n_head, T, d_head)
    y = mx.swapaxes(y, -3, -2).reshape(*lead, T, C)  # head0[0:64], head1[64:128], ...
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


def gpt_forward_mlx(idx, params, n_head=6):
    """(T,) of ids -> (T, vocab) of logits; (B, T) -> (B, T, vocab). Lazy mx.array out.

    Same window-relative position rule as the NumPy engine: `wpe[:T]` over the
    already-cropped window, right-padding only for batches.
    """
    idx = idx if isinstance(idx, mx.array) else mx.array(np.asarray(idx))
    T = idx.shape[-1]  # LAST axis — anything before it is a batch axis.

    x = params["wte.weight"][idx] + params["wpe.weight"][:T]

    n_layer = sum(1 for k in params if k.endswith("ln_1.weight"))
    for i in range(n_layer):
        x = block(x, params, i, n_head)

    x = layer_norm(x, params["ln_f.weight"], params["ln_f.bias"])
    return x @ params["wte.weight"].T  # tied, no bias
