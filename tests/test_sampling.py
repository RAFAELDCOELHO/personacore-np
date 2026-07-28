"""M4: temperature / top-k / top-p sampling.

===========================================================================
NO BIT-PARITY IN THIS FILE, AND THAT IS NOT A WEAKENING
===========================================================================

Every earlier milestone compared against a frozen PyTorch tensor. This one
cannot: numpy's Generator and torch's Generator are different algorithms, so the
same seed produces different draws by construction. Chasing an identical sample
would be chasing a coincidence.

So every test here proves a STRUCTURAL property — the right formula, the right
support, the right order — verified against arithmetic written out by hand in
the test, never against this engine's own implementation.

===========================================================================
CONVENTIONS, READ OFF THE PERSONACORE SOURCE (not off the literature)
===========================================================================

Confirmed by reading src/personacore/generation/sampling.py and by measuring the
real torch functions. Where PersonaCore diverges from the common convention,
PersonaCore wins — this engine replicates it, rough edges included.

1. TEMPERATURE — `logits / max(temperature, 1e-8)`.
   Division, with a FLOOR rather than a guard. Consequences, both measured:
     * temperature=0 does NOT raise and does NOT produce inf/nan; it divides by
       1e-8. On [2.0, 1.0, 0.5] that gives [2e8, 1e8, 5e7].
     * temperature NEGATIVE is silently clamped to the same floor. -2.0 gives
       exactly the same result as 0.0. It does not invert the distribution; it
       sharpens it maximally. There is no validation anywhere.
   The upstream docstring claims "temperature == 0 is handled as the greedy
   branch upstream in next_token". That branch does not exist — `next_token`
   only branches on `greedy`, and `core.generate` passes temperature straight
   through. The docstring is wrong about its own code; the code is what is
   replicated here.

2. TOP-K — mask to -inf BEFORE the softmax, with a STRICT `<` cutoff:
       out[out < kth_largest] = -inf
   So logits TIED with the k-th largest survive, and more than k tokens can
   remain. Measured on [5.0, 3.0, 3.0, 1.0] with k=2: three survivors. `k` is
   not a hard cap under ties.
   There is NO internal guard for k <= 0 (torch raises IndexError for 0 and
   RuntimeError for negatives). The guard lives in the caller, `next_token`:
       if top_k is not None and top_k > 0:
   That split is replicated exactly — filter unguarded, caller guarded.

3. TOP-P — nucleus, convention (a): the crossing token is KEPT.
       sorted_mask = cum_probs >= p
       sorted_mask[..., 1:] = sorted_mask[..., :-1]
       sorted_mask[..., 0] = False
   A token is masked iff the mass through its PREDECESSOR already reached p, so
   the token that crosses the line stays and the top-1 token is never masked.
   `>=` (not `>`) means an exact landing closes the nucleus. Measured on
   probabilities [0.5, 0.25, 0.125, 0.125], cumulative [0.5, 0.75, 0.875, 1.0]:
       p=0.50 -> 1 kept     p=0.75 -> 2 kept (exact landing, `>` would keep 3)
       p=0.80 -> 3 kept     p=0.90 -> 4 kept
   At least one token always survives, structurally.

   Binary fractions on purpose. With [0.5, 0.3, 0.15, 0.05] the "exact" landing
   at p=0.80 is not exact: 0.5+0.3 is 0.7999999999999999 in float64, so numpy AND
   torch-float64 both keep three tokens, while torch-float32 keeps two. The
   `>=`/`>` distinction is only observable where the sum is representable.

4. ORDER — temperature, then top-k, then top-p, then softmax, then the draw.
   top-p therefore operates on the ALREADY top-k-filtered set. Test 8 proves the
   order is observable rather than assumed.

Scope: sampling is wired into `engine.generate.generate` (the no-cache M1 path)
only. `generate_with_cache` stays greedy-only this milestone.
"""

import numpy as np
import pytest

from engine.forward import softmax
from engine.generate import generate
from engine.sampling import apply_temperature, sample_next_token, top_k_filter, top_p_filter


def entropy(p):
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def support(logits):
    """Indices that survived a filter — finite, i.e. not masked to -inf."""
    return set(np.flatnonzero(np.isfinite(np.asarray(logits))).tolist())


def filter_chain(logits, *, temperature, top_k, top_p):
    """The three filters in the locked order, no draw.

    Written out here rather than imported so the order-of-operations test states
    its own expectation instead of borrowing the implementation's.
    """
    x = apply_temperature(logits, temperature)
    if top_k is not None and top_k > 0:
        x = top_k_filter(x, top_k)
    if top_p is not None:
        x = top_p_filter(x, top_p)
    return x


# ----------------------------------------------------------- temperature


def test_temperature_matches_formula():
    """`logits / max(T, 1e-8)`, checked against the division written out by hand.

    The logits are deliberately uneven and unsorted so scaling changes the RATIOS
    of the probabilities observably — all-equal logits are invariant under
    temperature and would make this vacuous.
    """
    logits = np.array([2.0, -1.0, 0.5, 3.25, -0.75])

    for temperature in (1.0, 0.5, 2.0, 10.0):
        expected = logits / temperature
        got = apply_temperature(logits, temperature)
        assert np.array_equal(got, expected), f"T={temperature}: {got} != {expected}"

    # The floor, replicated from PersonaCore: zero and negatives both land on
    # 1e-8 rather than raising or inverting the distribution.
    floored = logits / 1e-8
    assert np.array_equal(apply_temperature(logits, 0.0), floored)
    assert np.array_equal(apply_temperature(logits, -2.0), floored)


def test_temperature_changes_entropy_monotonically():
    """Higher temperature -> strictly higher entropy. Structural, no RNG involved.

    This is the property the parameter exists for, and it needs no reference
    implementation: dividing by a larger number flattens the softmax, and a
    flatter distribution has more entropy. A mutation that multiplied instead of
    divided reverses the ordering.
    """
    logits = np.array([3.0, 1.5, 0.5, -1.0, -2.5])
    temps = [0.1, 0.5, 1.0, 2.0, 5.0]

    entropies = [entropy(softmax(apply_temperature(logits, t), axis=-1)) for t in temps]
    print(f"\n[entropy by temperature] {[(t, round(e, 4)) for t, e in zip(temps, entropies)]}")

    assert all(a < b for a, b in zip(entropies, entropies[1:])), entropies


def test_temperature_near_zero_approaches_greedy_argmax():
    """T=0.01 must put essentially all mass on the argmax — the VALUE, not the rank.

    "The argmax is still the argmax" would pass at T=1000 too. The probability
    itself is what shows the distribution actually collapsed.

    T=0.0 is included because the Passo-1 measurement showed it does NOT divide
    by zero: PersonaCore floors it at 1e-8, so it is a legitimate input rather
    than undefined behaviour.
    """
    logits = np.array([2.0, 1.0, 0.5, -3.0])
    peak = int(np.argmax(logits))

    for temperature in (0.01, 0.0):
        p = softmax(apply_temperature(logits, temperature), axis=-1)
        print(f"\n[T={temperature}] p[argmax] = {p[peak]:.12f}")
        assert p[peak] > 0.99
        assert int(np.argmax(p)) == peak
        assert abs(p.sum() - 1.0) < 1e-12


# ---------------------------------------------------------------- top-k


def test_top_k_restricts_support_to_exactly_k_highest_by_value():
    """The k survivors are the k largest BY VALUE, checked against value not position.

    The logits are shuffled so rank-by-value and rank-by-index disagree; a filter
    that kept the first k slots, or sorted ascending, would otherwise agree with
    the expected set by coincidence.
    """
    rng = np.random.default_rng(7)
    logits = rng.permutation(np.array([9.0, 7.5, 6.0, 4.5, 3.0, 1.5, 0.0, -1.5]))

    for k in (1, 3, 5):
        out = top_k_filter(logits, k)
        kept = support(out)
        expected = set(np.argsort(logits)[-k:].tolist())

        assert len(kept) == k, f"k={k}: kept {len(kept)}"
        assert kept == expected, f"k={k}: {kept} != {expected}"
        dropped = [logits[i] for i in range(len(logits)) if i not in kept]
        assert min(logits[i] for i in kept) > max(dropped, default=-np.inf)
        # Kept logits must be untouched, not merely finite.
        assert all(out[i] == logits[i] for i in kept)


def test_top_k_keeps_more_than_k_when_the_cutoff_is_tied():
    """PersonaCore's cutoff is a STRICT `<`, so ties at the boundary all survive.

    Replicated deliberately, not fixed. On [5, 3, 3, 1] with k=2 the real torch
    implementation keeps THREE tokens; a "correct" hard cap of k would be a
    divergence from the model being mirrored.
    """
    logits = np.array([5.0, 3.0, 3.0, 1.0])
    out = top_k_filter(logits, 2)

    assert support(out) == {0, 1, 2}, "the tie at the k-th value was not kept"
    assert out[3] == -np.inf


def test_top_k_zero_or_negative_is_a_no_op():
    """The guard lives in the CALLER, exactly as in PersonaCore's `next_token`.

    `top_k_filter` itself is unguarded upstream (torch raises IndexError at k=0
    and RuntimeError at k<0); `next_token` wraps it in `if top_k is not None and
    top_k > 0`. So the no-op is asserted through `sample_next_token`, the layer
    that owns the guard.
    """
    logits = np.array([1.0, 5.0, 2.0, 4.0])
    rng = np.random.default_rng(0)

    for k in (0, -1, None):
        drawn = {
            sample_next_token(logits, temperature=1.0, top_k=k, top_p=None, rng=rng)
            for _ in range(400)
        }
        assert drawn == {0, 1, 2, 3}, f"top_k={k} filtered something: {sorted(drawn)}"


# ---------------------------------------------------------------- top-p


def test_top_p_cuts_by_the_confirmed_convention():
    """The crossing token stays, and an EXACT landing closes the nucleus.

    Probabilities are binary fractions — [0.5, 0.25, 0.125, 0.125], cumulative
    [0.5, 0.75, 0.875, 1.0] — so 0.75 is representable with no rounding and
    `cum[1] == 0.75` is exactly True. That is what makes p=0.75 able to tell `>=`
    from `>`: under `>=` the nucleus closes at two tokens, under `>` the third
    would be pulled in.

    The obvious choice, [0.5, 0.3, 0.15, 0.05] with p=0.80, CANNOT do this job.
    0.5 + 0.3 is 0.7999999999999999 in float64, so `>= 0.8` is False and three
    tokens survive. Measured: numpy-float64 and torch-float64 both keep {0,1,2};
    only torch in float32 keeps {0,1}, because there the same sum rounds up to
    0.800000011920929. An assertion built on that is asserting the precision, not
    the convention.
    """
    logits = np.log(np.array([0.5, 0.25, 0.125, 0.125]))
    cum = np.cumsum(softmax(logits, axis=-1))
    assert cum[1] == 0.75, "the exact-landing case is not exact — the test cannot discriminate"

    assert support(top_p_filter(logits, 0.75)) == {0, 1}  # `>=` closes here; `>` would not
    assert support(top_p_filter(logits, 0.80)) == {0, 1, 2}  # crossing token kept
    assert support(top_p_filter(logits, 0.50)) == {0}
    assert support(top_p_filter(logits, 0.90)) == {0, 1, 2, 3}


def test_top_p_keeps_at_least_one_token_when_p_is_very_small():
    """p=0.01 with a dominant token still leaves exactly the top-1 alive.

    Structural, not defensive: `sorted_mask[..., 0] = False` makes it impossible
    to mask the highest-probability token, whatever p is. Convention (b) — mask
    anything whose PRECEDING mass already reached p — would empty the support
    here and make the following softmax all-nan.
    """
    logits = np.log(np.array([0.7, 0.2, 0.07, 0.03]))

    for p in (0.01, 0.0, 0.5):
        kept = support(top_p_filter(logits, p))
        assert kept == {0}, f"p={p}: {kept}"


def test_top_p_selects_by_probability_rank_not_by_position():
    """Shuffled logits: the nucleus follows value order, not index order.

    Every other top-p test here uses descending logits, where the sort is a no-op
    and an implementation that skipped it entirely would still pass.
    """
    logits = np.log(np.array([0.125, 0.5, 0.125, 0.25]))
    assert support(top_p_filter(logits, 0.75)) == {1, 3}


# ------------------------------------------- the order of the three filters


def test_temperature_then_top_k_then_top_p_order_matters():
    """A token inside the top-p nucleus but outside the top-k must not survive.

    The milestone claims the pipeline is temperature -> top-k -> top-p, with
    top-p narrowing a set top-k already narrowed. That is only a claim worth
    making if the reversed order gives a different answer, so both orders are
    computed and their DISAGREEMENT is asserted before asserting which one the
    implementation follows. Same discipline as M0/M3: prove the case
    discriminates before comparing against it.
    """
    logits = np.log(np.array([0.4, 0.3, 0.2, 0.07, 0.03]))
    k, p = 3, 0.75

    forward = top_p_filter(top_k_filter(logits, k), p)
    backward = top_k_filter(top_p_filter(logits, p), k)

    print(f"\n[order] k-then-p: {sorted(support(forward))}   p-then-k: {sorted(support(backward))}")

    assert support(forward) != support(backward), (
        "the two orders agree on this input, so it cannot discriminate — pick "
        "logits where they diverge before asserting"
    )
    assert support(filter_chain(logits, temperature=1.0, top_k=k, top_p=p)) == support(forward)

    # And through the ENGINE, not just the chain written above: `filter_chain` is
    # this file's own copy of the order, so asserting against it alone would let a
    # reversed pipeline inside `sample_next_token` pass untouched. Drawing reveals
    # the real support — {0,1} under the correct order, {0,1,2} under the reversed.
    rng = np.random.default_rng(11)
    drawn = {
        sample_next_token(logits, temperature=1.0, top_k=k, top_p=p, rng=rng)
        for _ in range(2000)
    }
    assert drawn == support(forward), (
        f"sample_next_token drew from {sorted(drawn)}, expected {sorted(support(forward))} "
        "— the filters are applied in the wrong order"
    )


# -------------------------------------------------------------- the draw


def test_sampling_produces_a_valid_probability_distribution():
    """After the full chain the softmax sums to 1, has no negatives, right support."""
    logits = np.log(np.array([0.4, 0.3, 0.2, 0.07, 0.03]))
    filtered = filter_chain(logits, temperature=0.8, top_k=3, top_p=0.9)
    p = softmax(filtered, axis=-1)
    kept = support(filtered)

    assert abs(p.sum() - 1.0) < 1e-12
    assert (p >= 0.0).all()
    assert set(np.flatnonzero(p > 0).tolist()) == kept
    assert (p[sorted(set(range(5)) - kept)] == 0.0).all()


def test_sampling_draws_only_from_the_surviving_support():
    """Thousands of draws never land outside the filtered set, and cover all of it.

    Both halves matter: "never outside" catches a filter that leaks, "covers all"
    catches one that collapsed to a single token and would make every other
    sampling test pass trivially.
    """
    logits = np.log(np.array([0.4, 0.3, 0.2, 0.07, 0.03]))
    expected = support(filter_chain(logits, temperature=1.0, top_k=3, top_p=0.99))
    rng = np.random.default_rng(1234)

    drawn = {
        sample_next_token(logits, temperature=1.0, top_k=3, top_p=0.99, rng=rng)
        for _ in range(3000)
    }
    assert drawn == expected, f"drawn {sorted(drawn)} != support {sorted(expected)}"


# --------------------------------------------------------------- the loop

STUB_VOCAB = 16


def stub_forward(window):
    """A forward whose rows are a fixed, uneven distribution over 16 ids.

    Deterministic and checkpoint-free: the loop tests are about the sampling
    branch, not the model. The peak sits off-centre at id 11 so an argmax bug and
    an off-by-one both surface as a wrong id rather than a plausible one.
    """
    logits = np.tile(np.linspace(-2.0, 2.0, STUB_VOCAB), (len(window), 1))
    logits[:, 11] = 3.0
    return logits


def _run(seed, **kw):
    return generate(
        stub_forward,
        np.array([1, 2, 3], dtype=np.int64),
        max_new_tokens=25,
        eos_id=-1,
        block_size=256,
        greedy=False,
        temperature=1.0,
        rng=np.random.default_rng(seed),
        **kw,
    )


def test_sampling_is_reproducible_with_same_seed():
    """Same seed, same sequence — determinism, NOT parity with torch.

    The second assertion is the load-bearing one: without it, a `generate` that
    ignored the rng entirely and always took the argmax would pass.
    """
    assert _run(42) == _run(42)
    assert _run(42) != _run(99), "the rng is not actually driving the draw"


def test_greedy_bypasses_sampling_entirely():
    """greedy=True with absurd knobs still returns the pure argmax, byte for byte.

    temperature=1000 flattens the distribution to near-uniform and top_k=1 /
    top_p=0.001 clamp it to one token; a random draw under any of those is
    overwhelmingly unlikely to reproduce a 30-token greedy run exactly. Equality
    with the M1 result is therefore proof the greedy branch never touched the
    pipeline.
    """
    prompt = np.array([1, 2, 3], dtype=np.int64)
    common = dict(max_new_tokens=30, eos_id=-1, block_size=256)

    m1 = generate(stub_forward, prompt, greedy=True, **common)
    m4 = generate(
        stub_forward,
        prompt,
        greedy=True,
        temperature=1000.0,
        top_k=1,
        top_p=0.001,
        rng=np.random.default_rng(0),
        **common,
    )
    assert m4 == m1
    assert set(m1) == {11}, "the stub's argmax is id 11 — the greedy run drifted"


def test_generate_with_sampling_actually_samples():
    """greedy=False no longer raises — it runs and yields in-vocabulary ids.

    Replaces `test_generate_rejects_sampling_because_m1_is_greedy_only`, which
    asserted the NotImplementedError M1 raised on purpose while sampling did not
    exist. That fence is now wrong, so it was deleted rather than left to
    contradict this file.
    """
    got = _run(2024, top_k=5, top_p=0.95)

    assert len(got) == 20 or len(got) == 25
    assert all(isinstance(t, int) for t in got)
    assert all(0 <= t < STUB_VOCAB for t in got)
    assert len(set(got)) > 1, "every draw was the same token — this is greedy in disguise"


@pytest.mark.parametrize("greedy", [True, False])
def test_generate_signature_accepts_the_sampling_knobs(greedy):
    """Both branches accept the same call shape — no keyword is greedy-only."""
    got = generate(
        stub_forward,
        np.array([1, 2], dtype=np.int64),
        max_new_tokens=3,
        eos_id=-1,
        block_size=256,
        greedy=greedy,
        temperature=0.9,
        top_k=4,
        top_p=0.9,
        rng=np.random.default_rng(5),
    )
    assert len(got) == 3
