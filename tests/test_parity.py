"""Parity of the NumPy engine against the real PersonaCore, via a frozen oracle.

The fixture `fixtures/personacore_parity.npz` (read-only symlink to
~/tensorforge/fixtures) carries the 13.9M weights of the `best.pt` checkpoint
(49k steps, val_loss 0.7378), a real window from `data/val.bin` (T=256), and the
`ref_logits`/`ref_loss` produced by PyTorch with the model promoted to float64.
It is not an oracle written here: it is another framework, another kernel
author, the same numbers.

===========================================================================
TOLERANCE DERIVATION — written BEFORE any measurement
===========================================================================

Written before the implementation existed and before running any forward. No
number below was adjusted after seeing the result.

Premise: float64, eps = 2.220446049250313e-16. Both sides compute the SAME math
on the SAME input bits (the fp32 -> fp64 cast is lossless), so there is no input
error — what remains is exclusively floating-point accumulation order, which
differs because PyTorch's BLAS and NumPy's BLAS block and sum the inner products
in different ways.

Error model: an inner product of length K summed pairwise (what any modern BLAS
does, via blocking) has relative error ~ sqrt(K)*eps. (Strictly sequential
summation would give K*eps; using sqrt(K) is the realistic estimate and the
tighter of the two, therefore the harder one to justify.)

Critical path of ONE logit, contraction by contraction:

  Per block (every contraction a logit passes through):
    1. q = ln_1(x) @ Wq        K = 384    sqrt(384) = 19.6
    2. k = ln_1(x) @ Wk        K = 384    sqrt(384) = 19.6
    3. scores = q @ k^T        K = 64     sqrt(64)  =  8.0   <- d_head, NOT n_embd
    4. softmax (sum on axis)   K = 256    sqrt(256) = 16.0
    5. v = ln_1(x) @ Wv        K = 384    sqrt(384) = 19.6
    6. y = att @ v             K = 256    sqrt(256) = 16.0
    7. c_proj                  K = 384    sqrt(384) = 19.6
    8. fc_in                   K = 384    sqrt(384) = 19.6
    9. fc_out                  K = 1536   sqrt(1536)= 39.2
                                          --------------
                              per-block sum = 177.2 eps

  6 blocks ....................... 6 x 177.2 = 1063.2 eps
  ln_f (normalization, no matmul) ...........    2.0 eps
  lm_head (x @ wte^T, K = 384) ..............   19.6 eps
                                             -----------
                                    TOTAL  ~ 1084.8 eps

  1084.8 x 2.220446e-16 = 2.4e-13   <-- DERIVED ESTIMATE

Two forces push the real value BELOW the estimate:
  - Every LayerNorm renormalizes the residual stream, which prevents the
    relative error from composing multiplicatively across blocks. The additive
    model above is conservative by construction.
  - The metric normalizes by the GLOBAL max|ref_logits|, which is >= the typical
    |logit|, so the reported relative error is <= the elementwise one.
No force pushes it up: this engine is forward-only, the gradient does not
traverse the depth again (that is what would double the bound in an engine with
a backward pass).

Honest note: 2.4e-13 is the same order the tensorforge derived. That is not
copying — it follows from the architecture, the eps and the Ks being the same.
The derivation above was redone from scratch from this engine's forward; it is
the MEASURED value that can diverge, and that is what the canary assertion pins.

Main assertion: rtol = 1e-11, ~40x of slack over the estimate (more than one
order of magnitude). If the measured error gets anywhere near it, the problem is
architectural (transpose, operation order, wrong formula), not numerical — and
the instruction is to stop and investigate, never to loosen the number.

Canary assertion: pinned at the value ACTUALLY measured, one order of magnitude
above it. Filled in during the refactor step, after GREEN — never before.

Metric (identical to the tensorforge's, replicated on purpose so the numbers are
comparable between the two projects):

    error = max(|got - ref|) / max(|ref|)

GLOBAL relative error normalized by the maximum. It is NOT `np.allclose`, which
is elementwise and uses atol+rtol — the two give different numbers and mixing
them would make the reports incomparable.
"""

import os

import numpy as np
import pytest

from engine.forward import cross_entropy, gpt_forward
from engine.weights import load_weights

FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures", "personacore_parity.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE),
    reason="fixtures/personacore_parity.npz not found — generate it with "
    "tensorforge/scripts/gen_parity_fixture.py against a trained PersonaCore "
    "checkpoint (see README)",
)

VOCAB, N_EMBD, N_HEAD, N_LAYER, BLOCK = 8192, 384, 6, 6, 256

LOGITS_RTOL = 1e-11  # derived above: 2.4e-13 with ~40x of slack

# CANARY — pinned AFTER the first GREEN, at the value actually measured.
# Measurement: logits relative error = 2.5205311896719e-15 (max|diff| 4.796e-14
# over max|ref| 19.028). The canary sits one order of magnitude above that.
# If the error rises from 2.5e-15 to, say, 3e-14, the main rtol would still pass
# (it is 400x larger) and the regression would slip by unnoticed — the canary
# fires first and reports it. It is NOT a derived tolerance; it is a drift
# detector. If it fires, the question is "what changed", not "which number to
# loosen".
LOGITS_CANARY = 1e-14

# The loss came out BITWISE identical to PyTorch's (relative error exactly 0.0).
# One does not pin a canary at 0.0 — a sum of 256 terms could reassociate under
# a different BLAS version with nothing being wrong. 1e-14 is the same drift
# detector, and the fact that the measurement was exact is recorded here.
LOSS_CANARY = 1e-14


def relative_error(got, expected):
    """Global relative error normalized by the maximum — the tensorforge's metric."""
    return np.abs(got - expected).max() / np.abs(expected).max()


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def params():
    return load_weights(FIXTURE)


@pytest.fixture(scope="module")
def logits(params, ref):
    """One forward, reused by the tests (T=256 x 6 layers: expensive)."""
    return gpt_forward(ref["input_x"], params, n_head=N_HEAD)


# ------------------------------------------- (3) transpose, by VALUE not shape


def test_attention_weight_transpose_by_value_not_shape(ref, params):
    """The 4 attention projections are (384,384) — shape does NOT discriminate a transpose.

    This is the test shape cannot do. `q_proj.weight` is square, so forgetting
    the `.T` in the loader changes no shape, raises no exception, and turns into
    numerical divergence with no apparent cause down in the forward. The only
    defense is comparing a VALUE at an asymmetric position.

    We first prove the chosen point discriminates (raw[0,1] != raw[1,0], i.e.
    the matrix is not symmetric there), and only then require the crossed
    equality. Without the first assertion, an accidentally symmetric matrix
    would make the test pass with or without the transpose.
    """
    raw = ref["w::blocks.0.attn.q_proj.weight"]  # (out, in), PyTorch convention
    got = params["blocks.0.attn.q_proj.weight"]  # (in, out), this engine's convention

    assert raw[0, 1] != raw[1, 0], "chosen point is symmetric — it does not discriminate"
    assert raw[0, 5] != raw[5, 0], "chosen point is symmetric — it does not discriminate"

    assert got[0, 1] == np.float64(raw[1, 0])
    assert got[1, 0] == np.float64(raw[0, 1])
    assert got[0, 5] == np.float64(raw[5, 0])
    assert got[5, 0] == np.float64(raw[0, 5])


def test_non_square_linear_weights_are_transposed_to_in_out(params):
    """fc_in/fc_out: here the shape ALREADY tells. Belt and braces over the previous test.

    npz: fc_in (1536,384) and fc_out (384,1536) in (out,in). In this engine's
    (in,out) convention they become (384,1536) and (1536,384) — swapped with
    each other.
    """
    assert params["blocks.0.mlp.fc_in.weight"].shape == (N_EMBD, 4 * N_EMBD)
    assert params["blocks.0.mlp.fc_out.weight"].shape == (4 * N_EMBD, N_EMBD)


def test_embeddings_and_biases_are_not_transposed(ref, params):
    """wte/wpe and every bias load straight through — transposing one would be a bug."""
    assert params["wte.weight"].shape == (VOCAB, N_EMBD)
    assert params["wpe.weight"].shape == (BLOCK, N_EMBD)
    assert params["blocks.0.attn.q_proj.bias"].shape == (N_EMBD,)
    assert np.array_equal(params["wte.weight"], ref["w::wte.weight"].astype(np.float64))


def test_all_loaded_weights_are_float64(params):
    """fp32 on disk, fp64 in memory: a single forgotten fp32 destroys the parity.

    The error of a whole forward in fp32 is ~1e-7, many orders of magnitude
    above the tolerance. This fails fast and points at the culprit.
    """
    wrong = [k for k, v in params.items() if v.dtype != np.float64]
    assert wrong == [], f"non-float64 weights: {wrong}"


# --------------------------------------------------------- (4) weight tying


def test_lm_head_is_tied_not_loaded_separately(ref, params):
    """`lm_head.weight` exists in the npz but must NOT exist in the loaded params.

    PyTorch serializes the tied parameter under BOTH names (101 keys, 100
    storages). Loading `lm_head` as a separate weight would be harmless today
    (the values are identical) but would mask a future break of the tying: if
    the checkpoint changed and the two tables diverged, an engine that loads
    `lm_head` would keep "passing" while producing wrong logits.

    The chosen invariant is the stronger of the two on offer: the key does not
    exist. The logits come from `wte.weight` and from nothing else.
    """
    assert "w::lm_head.weight" in ref.files, "premise changed: the key is gone from the npz"
    assert "lm_head.weight" not in params
    assert not any("lm_head" in k for k in params)


def test_the_npz_lm_head_really_is_wte(ref):
    """The tying premise, verified and not assumed — exact equality."""
    assert np.array_equal(ref["w::lm_head.weight"], ref["w::wte.weight"])


# ------------------------------------------------------- (6) the main test


def test_forward_matches_personacore_logits(logits, ref):
    """Logits against PyTorch, at the tolerance derived in the module docstring."""
    expected = ref["ref_logits"]
    assert logits.shape == expected.shape == (BLOCK, VOCAB)

    err = relative_error(logits, expected)
    print(f"\n[parity] logits relative error: {err:.3e}  (rtol {LOGITS_RTOL:.0e})")
    assert err < LOGITS_RTOL, f"error {err:.3e} — investigate the architecture, not the tolerance"
    assert err < LOGITS_CANARY, f"error {err:.3e} rose above the measured 2.5e-15 — what changed?"


def test_argmax_prediction_is_identical_token_for_token(logits, ref):
    """Discrete, no tolerance at all: the SAME token predicted at all 256 positions.

    The most important gate for M1. Greedy generation depends only on the
    argmax, not on the absolute value of the logit — an error concentrated at
    the top of the distribution, too small to break the continuous rtol, would
    change the generated text and slip past the whole numerical battery above.
    """
    assert np.array_equal(logits.argmax(-1), ref["ref_logits"].argmax(-1))


# ------------------------------------------------------------------ (7) loss


def test_forward_loss_matches(logits, ref):
    """Mean cross-entropy over the 256 tokens, same derived tolerance.

    The CE traverses exactly the same chain as the logits and adds a log-sum-exp
    (sum of K=8192, sqrt ~ 90, or +2e-14) and a mean over 256 terms (sqrt ~ 16).
    The addition is a fraction of the 2.4e-13 estimate, so the same rtol holds
    without a new derivation.
    """
    expected = float(ref["ref_loss"])
    got = cross_entropy(logits, ref["input_y"])

    err = abs(got - expected) / abs(expected)
    print(f"[parity] loss relative error: {err:.3e}  (loss = {got:.12f})")
    assert err < LOGITS_RTOL
    assert err < LOSS_CANARY, f"error {err:.3e} rose above the measured 0.0 exact — what changed?"


def test_prefix_logits_are_stable_because_the_model_is_causal(params, ref, logits):
    """ADDED after the mutation check: mutant GP-2 survived the original suite.

    GP-2 swaps `wpe[:T]` for `wpe[-T:]` in `gpt_forward`. With T=256 and the
    `wpe` table being exactly (256, 384), the two slices are THE SAME array —
    the mutant is indistinguishable across the whole parity battery, because the
    fixture only exercises the T == block_size case.

    For T < block_size they diverge: `wpe[:37]` is positions 0..36 and
    `wpe[-37:]` is positions 219..255. That is exactly the failure mode that
    would break M1 generation, where T grows from 1 up to 256 — every token
    decoded with the wrong positional embedding, and no test would see it.

    The oracle here is not PyTorch (there is no reference for T<256): it is
    CAUSALITY. Position i only sees 0..i, and the position is the index within
    the window, so running the prefix of length n has to produce exactly the
    same logits as the first n rows of the full forward. A property derived from
    the architecture, with no external oracle.
    """
    n = 37  # neither 1 (degenerate) nor 256 (the case the fixture already covers)
    short = gpt_forward(ref["input_x"][:n], params, n_head=N_HEAD)

    assert short.shape == (n, VOCAB)
    err = relative_error(short, logits[:n])
    print(f"[prefix n={n}] relative error vs full forward: {err:.3e}")
    assert err < LOGITS_CANARY
    assert np.array_equal(short.argmax(-1), logits[:n].argmax(-1))


def test_the_fixture_carries_a_real_corpus_window(ref):
    """Real input: ids within the vocab, y is x shifted by 1, non-degenerate text."""
    x, y = ref["input_x"], ref["input_y"]
    assert x.shape == (BLOCK,) and y.shape == (BLOCK,)
    assert x.max() < VOCAB and x.min() >= 0
    assert np.array_equal(x[1:], y[:-1]), "y is x shifted by one position"
    assert len(np.unique(x)) > 50, "real text, not one repeated token"
