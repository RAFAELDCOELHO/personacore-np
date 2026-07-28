"""M5: int8 fake quantization of the frozen weights.

===========================================================================
PASSO 2 — scheme decisions, each justified by the Passo 1 MEASUREMENTS
(numbers in tests/test_quantize.py's module docstring), not by literature
===========================================================================

* SYMMETRIC, no zero-point: scale = |w|max / 127, zero maps to code 0. Justified
  because every quantized category measured symmetric on THIS checkpoint —
  |skew| <= 0.043 and |mean| << std for all Linear weights and both embeddings.
  (LayerNorm measured heavily asymmetric — skew to -8.0, all-positive — and is
  excluded below, so no asymmetric path is needed anywhere.)

* PER-CHANNEL: one scale per OUTPUT channel of each Linear. In this engine's
  (in, out) storage the output channel is the LAST axis, so the reduction runs
  over every leading axis (`axis=tuple(range(ndim - 1))`, keepdims for
  broadcast). Justified: measured channel |w|max spreads of 1.98x-6.17x inside
  single tensors — a per-tensor scale would waste up to ~2.6 of the 8 bits on
  the quietest channel. The same last-axis rule covers wte/wpe (one scale per
  embedding dimension; measured spread 3.64x, essentially identical to the
  per-row 3.65x, so the uniform rule costs nothing).

* RANGE [-127, 127], not [-128, 127]: symmetric on purpose — using -128 would
  give the negative side one extra code and make the scheme asymmetric by one
  step. QMAX = 127.

* ELIGIBLE: the 36 Linear weights of attention and MLP (76.4% of params) PLUS
  wte and wpe (23.4%). wte is 22.6% of the model on its own and is tied to the
  lm_head, so its quantization error reaches every vocab logit — which is why
  the perplexity test measures it separately (with and without). EXCLUDED:
  LayerNorm weight/bias (9,984 params = 0.072% — no memory gain, real
  stability risk, and the one asymmetric category) and Linear biases (0.149%).

* FAKE QUANTIZATION: weights are STORED as int8 + float32 scale and DEQUANTIZED
  to float64 before every matmul — no integer matmul. The win this milestone
  claims is scheme correctness plus memory footprint (~8x vs the engine's
  float64, measured in the storage test), NOT execution speed; speed is
  explicitly M6's (MLX) objective.

The scale is stored (and returned) as float32 — the storage dtype — and the
quantization divides by that SAME f32 value, so the round-trip bound in
test_quantize_dequantize_round_trip_error_bounded is stated against the scale
actually shipped, not an internal f64 one.

There is no guard for an all-zero channel (scale 0 -> 0/0 = nan): the real
checkpoint has none (measured channel |w|max minimum is 0.0395), and a nan
storm on a hypothetical future checkpoint is a louder failure than a silent
zero-fill.
"""

import numpy as np

QMAX = 127  # symmetric [-127, 127]; code -128 deliberately unused

# Same Linear-weight suffixes as engine.weights transposes — the two lists must
# describe the same tensors, since eligibility is defined on the engine's
# (in, out) orientation.
_LINEAR_WEIGHTS = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.c_proj.weight",
    "mlp.fc_in.weight",
    "mlp.fc_out.weight",
)
_EMBEDDINGS = ("wte.weight", "wpe.weight")


def is_quantizable(name, include_embeddings=True):
    """True for the tensors the Passo 2 decision quantizes — never LN, never bias."""
    if name.endswith(_LINEAR_WEIGHTS):
        return True
    return include_embeddings and name in _EMBEDDINGS


def quantize_int8(weight, per_channel=True):
    """weight (f64) -> (q int8, scale float32). Symmetric, round-to-nearest.

    per_channel=True: one scale per last-axis channel (keepdims, so `q * scale`
    broadcasts). per_channel=False: a single scalar scale for the whole tensor.
    """
    if per_channel:
        amax = np.abs(weight).max(axis=tuple(range(weight.ndim - 1)), keepdims=True)
    else:
        amax = np.abs(weight).max()
    scale = (amax / QMAX).astype(np.float32)
    q = np.clip(np.rint(weight / scale.astype(np.float64)), -QMAX, QMAX).astype(np.int8)
    return q, scale


def dequantize_int8(q_weight, scale):
    """(int8, float32) -> float64 weight, the fake-quantization read path."""
    return q_weight.astype(np.float64) * scale.astype(np.float64)


def quantize_params(params, include_embeddings=True):
    """Replace every eligible weight with its (q_weight, scale) pair.

    LayerNorm parameters and biases pass through untouched (same objects)."""
    return {
        name: quantize_int8(w) if is_quantizable(name, include_embeddings) else w
        for name, w in params.items()
    }


def dequantize_params(qparams):
    """Materialize a params dict `gpt_forward` can run: pairs -> float64 arrays."""
    return {
        name: dequantize_int8(*v) if isinstance(v, tuple) else v
        for name, v in qparams.items()
    }
