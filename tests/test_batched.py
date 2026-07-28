"""M3: batched forward over right-padded sequences.

===========================================================================
WHY THERE IS NO PADDING MASK  (derivation, done BEFORE any code was written)
===========================================================================

The trap this milestone was meant to guard against is real. Zeroing a padding
token's V without masking its SCORE leaves `exp(score_pad)` in the shared softmax
denominator, so the weights on the REAL tokens get diluted even though V_pad
contributes nothing. One query, four keys, the last two padding:

    scores = [0.5, 1.2, 3.0, 2.5]        V = [1.0, 2.0, 9.9, -7.7]

    zeroing only V_pad:   weights = [0.044277, 0.089162, 0.539399, 0.327162]
                          output  = 0.2226007535
    masking the score:    weights = [0.331812, 0.668188, 0.0,      0.0     ]
                          output  = 1.6681877722

    denominator with pad = 1.8539145466   without pad = 1.4965853038

The real tokens' weights collapse from [0.3318, 0.6682] to [0.0443, 0.0892].

BUT with RIGHT-padding the trap cannot be reached, because every padding position
is in the FUTURE of every real query, and the causal mask already blocks it:

    query real i=0: causal allows keys [0]     -- pad already blocked
    query real i=1: causal allows keys [0, 1]  -- pad already blocked

In general: a real query sits at i < T_real, every pad key sits at j >= T_real,
so j > i and `tril` masks it. Not "usually" -- always, by construction.

Measured on the real checkpoint, causal mask ONLY, no padding mask anywhere,
seq_a(100) and seq_b(180) batched together:

    perturbing seq_a's pad region with token 8191 -> real logits change 0.000e+00
    perturbing it with token 4242                 -> real logits change 0.000e+00

Exactly zero, not "small". Contrast with LEFT-padding, same perturbation:

    real logits change 7.399e-01

So right-padding is load-bearing for two independent reasons: it keeps `wpe`
aligned with the unbatched forward (positions stay 0..T_real-1), AND it makes a
padding mask redundant. An explicit padding mask here would be provably dead
code at every real position. It is therefore not implemented, and this file
proves the invariant instead of assuming it.

`real_lengths` never enters the forward. It belongs to the caller, who knows
which slice of the output is meaningful. Logits at padding positions are
undefined and no test reads them.

===========================================================================
TOLERANCE -- written BEFORE measuring (only tests 1 and 2 need one)
===========================================================================

Batching changes no arithmetic. Every reduction keeps the same length: the
projections still contract over C=384, the scores over d_head=64, the softmax
and `att @ V` over the row's unmasked keys. The batch axis is a leading
broadcast dimension, not a term in any sum.

What CAN differ is BLAS blocking. A (180,64)x(64,180) matmul may accumulate a
length-64 dot product in a different order than a (100,64)x(64,100) one, and
`att @ V` over a padded row sums the same real terms interleaved with exact
zeros -- exact zeros do not change a value, but they do change how pairwise
summation groups the partials.

So the estimate is the M0 estimate unchanged: the critical path sums to
1084.8 eps, and 1084.8 x 2.220446e-16 = 2.4e-13. Do NOT assume batching is
worse; it performs the same reductions. The measurement decides.

Main assertion: rtol = 1e-11, ~40x of slack. Canary pinned after GREEN.
"""

import os

import numpy as np
import pytest

from engine.forward import gpt_forward
from engine.weights import load_weights

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures")
PARITY_FIXTURE = os.path.join(_FIXTURE_DIR, "personacore_parity.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(PARITY_FIXTURE),
    reason="fixtures/personacore_parity.npz not found — generate it with "
    "tensorforge/scripts/gen_parity_fixture.py against a trained PersonaCore "
    "checkpoint (see README)",
)

N_HEAD = 6
VOCAB = 8192

LOGITS_RTOL = 1e-11  # derived above: 2.4e-13 with ~40x of slack

# CANARY — pinned AFTER the first GREEN, one order of magnitude above the worst
# measured value. A drift detector, not a derived bound: the derived rtol is
# thousands of times looser and would not notice a regression from 1e-15 to
# 1e-13. If this fires, ask what changed in the batching, not which number to
# raise.
LOGITS_CANARY = 1e-14

# The leak a LEFT-padded batch produces, measured at 7.399e-01. The threshold
# sits at 1e-3 — twelve orders of magnitude above float64 noise, so rounding
# cannot trip it, and far enough below the measured value to survive a different
# corpus window.
LEAK_THRESHOLD = 1e-3

LEN_A, LEN_B = 100, 180


@pytest.fixture(scope="module")
def params():
    return load_weights(PARITY_FIXTURE)


@pytest.fixture(scope="module")
def sequences():
    """Two REAL corpus windows of different lengths, from the parity fixture.

    `input_x` is the frozen val.bin window the M0 fixture already carries, so
    these are real token ids without adding a dependency on a val.bin that a
    fresh clone does not have. Two different slices, not nested prefixes, so the
    two rows of the batch carry genuinely different content.

    Both are under block_size=256, so no crop interferes and every position maps
    straight onto `wpe`.
    """
    x = np.load(PARITY_FIXTURE)["input_x"].astype(np.int64)
    seq_a = x[:LEN_A]
    seq_b = x[len(x) - LEN_B :]
    assert seq_a.shape == (LEN_A,) and seq_b.shape == (LEN_B,)
    assert seq_a.max() < VOCAB and seq_b.max() < VOCAB
    return seq_a, seq_b


def right_pad(rows, fill=0):
    """Right-pad to the longest length. Real tokens keep positions 0..len-1."""
    t_max = max(len(s) for s in rows)
    out = np.full((len(rows), t_max), fill, dtype=np.int64)
    for i, seq in enumerate(rows):
        out[i, : len(seq)] = seq
    return out


def padded_batch_and_regions(rows, side, fill=0):
    """Pad on `side`; return `(batch, real_slice, pad_slice)` for row 0.

    The two slices are DERIVED from `side`, never hardcoded, and that coupling is
    the whole point. An earlier version of the contrast test below hardcoded them
    against right-padding, so flipping the padder silently repointed "the pad
    region" at real tokens — the assertions still passed, for reasons that had
    nothing to do with left versus right. Mutants PD-7/PD-8 found exactly that.

    Left-padding is produced here for the contrast test only; production never
    emits it.
    """
    t_max = max(len(s) for s in rows)
    out = np.full((len(rows), t_max), fill, dtype=np.int64)
    for i, seq in enumerate(rows):
        if side == "right":
            out[i, : len(seq)] = seq
        else:
            out[i, t_max - len(seq) :] = seq

    n0 = len(rows[0])
    if side == "right":
        return out, slice(0, n0), slice(n0, t_max)
    return out, slice(t_max - n0, t_max), slice(0, t_max - n0)


# ------------------------------------------- (1) batch axis alone, no padding


def test_batched_forward_without_padding_matches_per_sequence(params, sequences):
    """Two REAL sequences of the SAME length — isolates the batch axis by itself.

    No padding anywhere, so anything this catches is a batching bug and nothing
    else: a wrong axis in the head reshape, a transpose that mixes B with T, a
    `wpe` slice that reads the batch dimension as the sequence length. Test 2
    cannot separate those from padding effects; this one can.
    """
    seq_a, _ = sequences
    other = np.load(PARITY_FIXTURE)["input_x"].astype(np.int64)[LEN_A : 2 * LEN_A]
    assert other.shape == (LEN_A,)

    batched = gpt_forward(np.stack([seq_a, other]), params, n_head=N_HEAD)
    solo = np.stack(
        [gpt_forward(seq_a, params, n_head=N_HEAD), gpt_forward(other, params, n_head=N_HEAD)]
    )

    assert batched.shape == solo.shape == (2, LEN_A, VOCAB)
    err = np.abs(batched - solo).max() / np.abs(solo).max()
    print(f"\n[batch, no padding] relative error: {err:.3e}")
    assert err < LOGITS_RTOL
    assert err < LOGITS_CANARY


# ----------------------------------- (2) the main gate: padded batch vs solo


def test_padded_batch_matches_unpadded_per_sequence_at_real_positions(params, sequences):
    """seq_a(100) right-padded to 180 alongside seq_b(180). Real slices only.

    This is the claim the milestone rests on: packing a short sequence next to a
    long one, with padding to make the rectangle, leaves the short sequence's
    real logits identical to running it alone. Only `[0, :100]` and `[1, :180]`
    are read; the padding rows are undefined by contract.
    """
    seq_a, seq_b = sequences
    batch = right_pad([seq_a, seq_b])
    assert batch.shape == (2, LEN_B)

    logits = gpt_forward(batch, params, n_head=N_HEAD)
    solo_a = gpt_forward(seq_a, params, n_head=N_HEAD)
    solo_b = gpt_forward(seq_b, params, n_head=N_HEAD)

    err_a = np.abs(logits[0, :LEN_A] - solo_a).max() / np.abs(solo_a).max()
    err_b = np.abs(logits[1, :LEN_B] - solo_b).max() / np.abs(solo_b).max()
    print(f"\n[padded batch] seq_a {err_a:.3e}   seq_b {err_b:.3e}")

    for name, err in (("seq_a", err_a), ("seq_b", err_b)):
        assert err < LOGITS_RTOL, f"{name}: {err:.3e} — investigate batching, not the tolerance"
        assert err < LOGITS_CANARY, f"{name}: {err:.3e} rose above the measured floor"


# --------------------------- (3) the structural invariant: padding cannot leak


def test_right_padding_perturbation_does_not_change_real_output(params, sequences):
    """Rewrite seq_a's padding with extreme ids — real logits must not move at all.

    ZERO IS THE EXPECTED RESULT HERE, not a suspiciously good one. With
    right-padding the causal mask gives every pad key a weight of exactly
    `exp(-inf) == 0.0` for every real query, so the pad's content never enters
    any sum. This asserts an exact 0.0 rather than a tolerance precisely because
    a float-epsilon difference would mean the mask is leaking somewhere, and a
    tolerance would hide it.

    Two different fill values, because a single one could coincidentally produce
    an embedding that happens not to matter.
    """
    seq_a, seq_b = sequences
    batch = right_pad([seq_a, seq_b])
    baseline = gpt_forward(batch, params, n_head=N_HEAD)[0, :LEN_A]

    for fill in (VOCAB - 1, 4242):
        perturbed = batch.copy()
        perturbed[0, LEN_A:] = fill
        got = gpt_forward(perturbed, params, n_head=N_HEAD)[0, :LEN_A]
        delta = np.abs(got - baseline).max()
        print(f"\n[pad perturbation fill={fill}] max change at real positions: {delta:.3e}")
        assert delta == 0.0, f"padding leaked into real positions: {delta:.3e}"


# ------------------------- (4) why right-padding, proven against the alternative


def test_left_padding_would_leak_but_right_padding_does_not(params, sequences):
    """The contrast that justifies the architecture: left-padding DOES leak.

    Without this, test 3 proves only that the current code does not leak — it
    cannot show that the choice mattered, and would pass just as happily against
    an implementation where leaking was impossible for uninteresting reasons.

    Left-padding is built here in the test only; production never emits it. Under
    left-padding the pad sits in the PAST of every real query, so the causal mask
    lets it through, and on top of that every real token's `wpe` row shifts.

    Both directions are asserted in one test on purpose: `left > threshold` alone
    would pass if every perturbation moved the output, and `right == 0` alone is
    test 3 again. The pair is the claim.

    Real and pad regions come from `padded_batch_and_regions`, which derives them
    from the side. Hardcoding them is what let PD-7/PD-8 survive the first time:
    with the slices frozen against right-padding, flipping the padder made the
    test perturb REAL tokens and report the resulting change as a leak.
    """
    seq_a, seq_b = sequences

    def leak(side):
        batch, real, pad = padded_batch_and_regions([seq_a, seq_b], side)
        assert batch[0, pad].size == LEN_B - LEN_A
        assert (batch[0, pad] == 0).all(), f"{side}: the 'pad' region holds real tokens"
        base = gpt_forward(batch, params, n_head=N_HEAD)[0, real]
        hit = batch.copy()
        hit[0, pad] = VOCAB - 1
        return np.abs(gpt_forward(hit, params, n_head=N_HEAD)[0, real] - base).max()

    left_delta = leak("left")
    right_delta = leak("right")

    print(f"\n[left vs right] left leak {left_delta:.3e}   right leak {right_delta:.3e}")

    assert left_delta > LEAK_THRESHOLD, (
        f"left-padding did not leak ({left_delta:.3e}) — this test no longer "
        "discriminates, so test 3 proves nothing about the choice"
    )
    assert right_delta == 0.0, f"right-padding leaked: {right_delta:.3e}"


# ------------------------------------------------- (5) isolation across batch rows


def test_batch_items_do_not_attend_to_each_other(params, sequences):
    """Perturb a REAL token of seq_b; seq_a's logits must not move at all.

    A different failure class from padding leakage: this one is about the batch
    axis being treated as a contraction axis somewhere — an einsum subscript
    reused, a reshape that folds B into T. A padding mask would not help against
    it, so it needs its own test even though both assert zero.

    Position 5 of seq_b is early enough that every later real position attends to
    it, which makes the perturbation loud on seq_b's own row — the counter-
    assertion below confirms the perturbation was not silently a no-op.
    """
    seq_a, seq_b = sequences
    batch = right_pad([seq_a, seq_b])
    baseline = gpt_forward(batch, params, n_head=N_HEAD)

    perturbed = batch.copy()
    perturbed[1, 5] = (int(perturbed[1, 5]) + 1234) % VOCAB
    got = gpt_forward(perturbed, params, n_head=N_HEAD)

    row_b_delta = np.abs(got[1, :LEN_B] - baseline[1, :LEN_B]).max()
    row_a_delta = np.abs(got[0, :LEN_A] - baseline[0, :LEN_A]).max()
    print(f"\n[cross-batch] seq_b moved {row_b_delta:.3e}   seq_a moved {row_a_delta:.3e}")

    assert row_b_delta > LEAK_THRESHOLD, "the perturbation did nothing — the test is vacuous"
    assert row_a_delta == 0.0, f"seq_a saw seq_b's tokens: {row_a_delta:.3e}"


# ----------------------------------------------------------- (6) shapes


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("t_max", [7, 64])
def test_gpt_forward_batched_shapes(params, batch_size, t_max):
    """(B, T) in -> (B, T, V) out, for B beyond the 2 the parity tests use.

    B=1 is the interesting one: it must return (1, T, V), NOT the (T, V) of the
    unbatched call. An implementation that squeezed leading size-1 axes would
    pass every other test in this file and still break the shape contract.
    """
    idx = np.arange(batch_size * t_max, dtype=np.int64).reshape(batch_size, t_max) % VOCAB
    out = gpt_forward(idx, params, n_head=N_HEAD)
    assert out.shape == (batch_size, t_max, VOCAB)


def test_unbatched_call_still_returns_two_dimensions(params, sequences):
    """The M0 contract is untouched: (T,) in still gives (T, V) out, not (1, T, V).

    Generalising `gpt_forward` over a leading batch axis must not quietly promote
    the single-sequence call, which every M0/M1/M2 test and both generation loops
    depend on.
    """
    seq_a, _ = sequences
    assert gpt_forward(seq_a, params, n_head=N_HEAD).shape == (LEN_A, VOCAB)
