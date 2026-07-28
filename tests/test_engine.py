"""Engine units, fixture-free: LayerNorm, GELU, softmax and the causal mask.

These tests never touch the checkpoint. The oracle for each one is a formula
transcribed by hand right here (or a structural property), never the
implementation itself — a test that calls `layer_norm` to produce the expected
value of `layer_norm` proves nothing.

Engine convention, documented once and valid across the repo: Linear weights are
stored as **(in, out)** and the forward computes `x @ W + b`. The npz arrives as
(out, in) (PyTorch convention), so the transpose happens ONCE, at load time
(`engine.weights.load_weights`), and never in the forward. See
`test_parity.py::test_attention_weight_transpose_by_value_not_shape`.
"""

import math

import numpy as np
import pytest

from engine.forward import attention, cross_entropy, gelu, layer_norm, softmax

# ---------------------------------------------------------------- (1) LayerNorm


def test_layernorm_matches_formula():
    """POPULATION variance (ddof=0) and eps INSIDE the sqrt.

    The vector [1,2,3,4] was chosen because it separates the two variances
    coarsely: the sum of squared deviations is 5.0, so var_pop = 5/4 = 1.25 and
    var_sample = 5/3 = 1.6667 — a ratio of 4/3, or ~15% in the standard
    deviation. An implementation using ddof=1 gets the first decimal place
    wrong, far above any numerical tolerance.

    The values 2.5 and 1.25 below are computed by hand (mean = 10/4 = 2.5;
    deviations -1.5,-0.5,0.5,1.5; squares 2.25,0.25,0.25,2.25; sum 5.0; /4 =
    1.25), NOT derived from the implementation under test.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    gamma = np.ones(4, dtype=np.float64)
    beta = np.zeros(4, dtype=np.float64)

    expected = (x - 2.5) / np.sqrt(1.25 + 1e-5)

    got = layer_norm(x, gamma, beta)
    assert np.abs(got - expected).max() < 1e-15


def test_layernorm_eps_is_inside_the_sqrt_not_outside():
    """`sqrt(var + eps)` != `sqrt(var) + eps` — separated with a tiny variance.

    With var ~1e-10 the eps of 1e-5 dominates: sqrt(1e-10 + 1e-5) = 3.162e-3,
    while sqrt(1e-10) + 1e-5 = 1.1e-5. Almost three orders of magnitude apart.
    With a vector of normal variance (~1) the two forms nearly coincide and the
    bug would go unnoticed — hence the deliberately flattened input.
    """
    x = np.array([1.0, 1.0 + 2e-5, 1.0 - 2e-5, 1.0], dtype=np.float64)
    gamma = np.ones(4, dtype=np.float64)
    beta = np.zeros(4, dtype=np.float64)

    mean = x.mean()
    var = np.mean((x - mean) ** 2)
    inside = (x - mean) / np.sqrt(var + 1e-5)
    outside = (x - mean) / (np.sqrt(var) + 1e-5)
    assert np.abs(inside - outside).max() > 1e-3, "input does not discriminate the two forms"

    got = layer_norm(x, gamma, beta)
    assert np.abs(got - inside).max() < 1e-14


def test_layernorm_applies_gamma_and_beta_not_just_normalizes():
    """Asymmetric, non-trivial gamma/beta: catches an ignored or swapped affine."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    gamma = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)
    beta = np.array([-1.0, 0.5, 11.0, -0.25], dtype=np.float64)

    expected = (x - 2.5) / np.sqrt(1.25 + 1e-5) * gamma + beta

    got = layer_norm(x, gamma, beta)
    assert np.abs(got - expected).max() < 1e-15


# --------------------------------------------------------------------- (2) GELU


def test_gelu_tanh_approx_not_erf():
    """x = +/-1.5, chosen because that is where tanh-approx and erf DIVERGE.

    The two formulas coincide at x=0 (both give 0) and converge in the tails
    (both -> 0 as x -> -inf, both -> x as x -> +inf). Maximum separation sits in
    the region |x| ~ 1-2. At x=1.5 the difference is ~1.4e-3, eleven orders of
    magnitude above the 1e-15 tolerance used here — so an engine using `erf`
    fails unambiguously. At x=0.01 the difference is ~1e-9 and the test would
    pass with the wrong formula; that is why we do NOT test near zero.

    The tolerance is 1e-15 (and not the one derived for the forward) because
    this is a closed-form expression of half a dozen operations — there is no
    matmul accumulation here.
    """
    for x_scalar in (1.5, -1.5):
        x = np.array([x_scalar], dtype=np.float64)

        # tanh-approx formula transcribed from the definition, independent of the
        # implementation.
        inner = math.sqrt(2.0 / math.pi) * (x_scalar + 0.044715 * x_scalar**3)
        expected_tanh = 0.5 * x_scalar * (1.0 + math.tanh(inner))

        # The erf version (the WRONG formula for this checkpoint), to prove that
        # the chosen point really does discriminate between the two.
        expected_erf = 0.5 * x_scalar * (1.0 + math.erf(x_scalar / math.sqrt(2.0)))
        assert abs(expected_tanh - expected_erf) > 1e-4, "x does not discriminate tanh from erf"

        got = gelu(x)
        assert abs(float(got[0]) - expected_tanh) < 1e-15


def test_gelu_is_not_relu_and_is_negative_for_small_negative_x():
    """GELU lets a little negative signal through — ReLU zeroes it. Catches a disguised relu."""
    got = gelu(np.array([-0.5], dtype=np.float64))
    assert float(got[0]) < 0.0
    assert float(got[0]) > -0.2


# ------------------------------------------------------------------- softmax


def test_softmax_does_not_overflow_on_large_logits():
    """ADDED after the mutation check: mutant SM-1 survived the original suite.

    SM-1 removes the max subtraction in `softmax`. On this checkpoint's real
    data the attention scores sit around 10, far from the fp64 exp overflow
    limit (~709), so the mutant is numerically identical across the ENTIRE
    parity battery — it would survive as "equivalent".

    It is not equivalent: it is defensive code whose effect only shows up on
    extreme input. This test forces the extreme input. Without the subtraction,
    `exp(1000) = inf` and the result becomes `inf/inf = nan` — silent data loss,
    the kind of thing you never simplify away.

    The -inf in the vector is deliberate: it is the value the causal mask
    injects, and it has to come out as exactly 0 even in the overflow regime.
    """
    x = np.array([[1000.0, 999.0, -np.inf]], dtype=np.float64)
    p = softmax(x, axis=-1)

    assert np.isfinite(p).all(), "softmax overflowed to inf/nan"
    assert abs(float(p.sum()) - 1.0) < 1e-15
    assert float(p[0, 2]) == 0.0, "a masked position must zero out exactly"


def test_cross_entropy_does_not_overflow_on_large_logits():
    """ADDED after the mutation check: mutant CE-2 survived the suite.

    CE-2 removes the max subtraction in `cross_entropy`. It is the SAME class as
    SM-1 above, in the same repo, and it stayed open because I had only closed
    `softmax`: the largest logit in the checkpoint is 19.03 and
    `exp(19.03) = 1.84e8`, against an fp64 overflow ceiling of ~1.8e308. That
    leaves 300 orders of magnitude of headroom, and the stabilization is
    unreachable on any real data from the fixture — the mutant is bitwise
    identical there.

    1e4 was chosen because `exp(1e4)` overflows to `inf` in fp64 (the exponent
    ceiling is ~709.78), while `1e4 - max = 0` does not. A value like 100 would
    NOT discriminate: `exp(100) = 2.7e43` is perfectly finite and both forms
    would give the same result.

    The failure mode is `inf`, NOT `nan`: exp(1e4)=inf -> sum=inf -> log(inf)=inf
    -> log_probs = z - inf = -inf -> loss = +inf. It would only come out `nan`
    (from `inf-inf`) if the target logit itself were already infinite, which
    finite input does not produce.

    The target is position 1 (the SMALL logit), not 0: with the target on the
    large logit the stabilized loss is `-0.0`, and "-0.0 is finite" would be a
    weak assertion. With the target at 1 it is exactly 9999.0 — `exp(-9999)` and
    `exp(-9998)` underflow to zero, the sum is exactly 1.0, `log(1.0)` is 0, so
    log_probs == z and the loss is the negation of -9999 with no rounding along
    the way.
    """
    logits = np.array([[1e4, 1.0, 2.0]], dtype=np.float64)
    targets = np.array([1])

    # The UNSTABILIZED form, written out here, to prove that 1e4 discriminates.
    with np.errstate(over="ignore"):
        unstable = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        unstable_loss = float(-unstable[np.arange(1), targets].mean())
    assert not np.isfinite(unstable).any(), "1e4 does not overflow here — pick a larger value"
    assert np.isinf(unstable_loss) and unstable_loss > 0

    got = cross_entropy(logits, targets)
    assert np.isfinite(got), "cross_entropy overflowed — the stabilization is gone"
    assert got == 9999.0


# ------------------------------------------------------------- (5) causal mask


def _synthetic_attn_params(n_embd, seed):
    """Small random weights to isolate the mask — without touching the checkpoint.

    (in, out) convention, the same as the loader's. Random (not identity) on
    purpose: with Wv = I attention would collapse into an average of the inputs
    themselves, and a future leak could cancel out by symmetry.
    """
    rng = np.random.default_rng(seed)
    p = {}
    for name in ("q_proj", "k_proj", "v_proj", "c_proj"):
        p[f"{name}.weight"] = rng.normal(0, 0.3, size=(n_embd, n_embd))
        p[f"{name}.bias"] = rng.normal(0, 0.1, size=(n_embd,))
    return p


def test_causal_mask_blocks_future_positions():
    """Perturb the future position t=5 and require outputs at t<=4 NOT to move.

    This is not "the output looks reasonable": it is the literal difference
    between two forwards that differ only in one future row. Without a mask, the
    softmax at position 2 receives mass from position 5 and the output at 2
    changes in the first decimal place. With a mask the difference has to be
    EXACTLY zero — not "small", zero: future positions never enter the sum.
    """
    n_embd, n_head, T, t_future = 12, 3, 8, 5
    p = _synthetic_attn_params(n_embd, seed=7)

    rng = np.random.default_rng(99)
    x = rng.normal(0, 1.0, size=(T, n_embd))
    x_perturbed = x.copy()
    x_perturbed[t_future] += 100.0  # huge perturbation: nothing subtle can hide it

    y = attention(x, p, "", n_head)
    y_perturbed = attention(x_perturbed, p, "", n_head)

    past = slice(0, t_future)
    assert np.array_equal(y[past], y_perturbed[past]), "the future leaked into the past"

    # Counter-proof: the perturbation IS visible where it should be (at the
    # future position itself). Without this, an `attention` returning zeros
    # would pass the assertion above.
    assert not np.allclose(y[t_future], y_perturbed[t_future])


def test_causal_mask_position_zero_attends_only_to_itself():
    """t=0 has a single visible key, so the softmax is 1.0 on it.

    A consequence testable without a reference mask: the output at t=0 is a
    function of x[0] alone. Replacing the ENTIRE rest of the sequence cannot
    move it.
    """
    n_embd, n_head, T = 12, 3, 8
    p = _synthetic_attn_params(n_embd, seed=13)

    rng = np.random.default_rng(101)
    x = rng.normal(0, 1.0, size=(T, n_embd))
    x_other = x.copy()
    x_other[1:] = rng.normal(0, 5.0, size=(T - 1, n_embd))

    assert np.array_equal(attention(x, p, "", n_head)[0], attention(x_other, p, "", n_head)[0])


@pytest.mark.parametrize("n_head", [1, 2, 3, 6])
def test_attention_output_shape_is_preserved_across_head_counts(n_head):
    """(T, C) in, (T, C) out — the head reshape has to be reversible."""
    n_embd, T = 12, 5
    p = _synthetic_attn_params(n_embd, seed=n_head)
    x = np.random.default_rng(3).normal(0, 1.0, size=(T, n_embd))
    assert attention(x, p, "", n_head).shape == (T, n_embd)
