"""M2: incremental KV-cache, with full-recompute fallback once the window slides.

===========================================================================
WHY THE CACHE STOPS BEING USABLE AT THE CROP  (derivation, kept from Passo 1)
===========================================================================

`gpt_forward` uses `wpe[:T]` over the ALREADY-CROPPED window, so a token's
positional embedding is its index INSIDE the window, not its absolute index in
the sequence. Worked example at block_size=4, sequence t0..t4:

    token | index in [t0..t3] | index in [t1..t4] | wpe before | wpe after
    ------|-------------------|-------------------|------------|----------
    t0    | 0                 | evicted           | wpe[0]     | --
    t1    | 1                 | 0                 | wpe[1]     | wpe[0]
    t2    | 2                 | 1                 | wpe[2]     | wpe[1]
    t3    | 3                 | 2                 | wpe[3]     | wpe[2]
    t4    | --                | 3                 | --         | wpe[3]

Every surviving token shifts by exactly -1. Measured on the real checkpoint
with real corpus ids:

    t1: |x_windowA - x_windowB| max = 0.2838   (0 would mean the cache is valid)
    t1: |logits_A - logits_B|   max = 3.735
    t1: argmax_A=111  argmax_B=105  equal=False

So cached K,V are wrong after the first crop, for two INDEPENDENT reasons:
  1. positional shift (the table above), and
  2. context truncation -- t1 used to attend to {t0,t1}; after the crop it
     attends to {t1} only, so its layer>=1 hidden state changes even if the
     position were fixed.

And it never recovers: the window slides ONE token per step, so no position is
ever reused.

    L=5 -> window=[t1..t4]
    L=6 -> window=[t2..t5]
    L=7 -> window=[t3..t6]

Recomputing K,V for the whole window IS a full forward, so "invalidate and
repopulate every step" costs a full forward plus cache bookkeeping -- strictly
worse than having no cache. Hence the architecture under test: cache while the
window fits, then hand the rest of the generation to the M1 path and never come
back.

Measured ceiling for the regime where the cache does help (prompt 10, 246 new
tokens, all within block_size=256): 7999 ms without cache vs 403 ms with, about
20x. At the boundary a single step is 63.11 ms full vs 1.62 ms incremental, 39x.

===========================================================================
TOLERANCE -- written BEFORE measuring (only test 4 needs one)
===========================================================================

Five of the six tests compare TOKEN IDS and use no tolerance at all: argmax
either lands on the same integer or it does not. Only
`test_cached_logits_match_full_recompute_per_step` compares floats.

The incremental path and the batched path contract over the SAME lengths:

    q/k/v projections   K = 384    sqrt(384) = 19.6  (x3)
    scores q.k          K = 64     sqrt(64)  =  8.0
    softmax             K = t+1    sqrt(256) = 16.0  (upper bound)
    att @ V             K = t+1    sqrt(256) = 16.0  (upper bound)
    c_proj              K = 384    sqrt(384) = 19.6
    fc_in               K = 384    sqrt(384) = 19.6
    fc_out              K = 1536   sqrt(1536)= 39.2
                                   ----------------
                       per block  = 177.2 eps

    6 blocks + ln_f (2) + head (19.6)  ~ 1084.8 eps
    1084.8 x 2.220446e-16              = 2.4e-13   <-- DERIVED ESTIMATE

Identical to the M0 estimate, and that is the point: the two paths differ in
BLAS SHAPE (matrix-vector vs matrix-matrix, different blocking and accumulation
order), not in how many terms are summed. Do NOT assume the incremental path is
worse -- it performs fewer, smaller reductions, so it could land either side.
The measurement decides; the canary pins whatever it turns out to be.

Main assertion: rtol = 1e-11, ~40x of slack over the estimate. Canary pinned
after GREEN, one order of magnitude above the measured value.
"""

import os

import numpy as np
import pytest

from engine.cache import cache_length, forward_step, generate_with_cache
from engine.forward import gpt_forward
from engine.generate import generate
from engine.weights import load_weights

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures")
GEN_FIXTURE = os.path.join(_FIXTURE_DIR, "personacore_generate_fixture.npz")
PARITY_FIXTURE = os.path.join(_FIXTURE_DIR, "personacore_parity.npz")
EOS_FIXTURE = os.path.join(_FIXTURE_DIR, "personacore_eos_fixture.npz")

pytestmark = pytest.mark.skipif(
    not (
        os.path.exists(GEN_FIXTURE)
        and os.path.exists(PARITY_FIXTURE)
        and os.path.exists(EOS_FIXTURE)
    ),
    reason="fixtures/*.npz not found — generate them with "
    "tensorforge/scripts/gen_parity_fixture.py, gen_generate_fixture.py and "
    "gen_eos_fixture.py against a trained PersonaCore checkpoint (see README)",
)

N_HEAD = 6
VOCAB = 8192

LOGITS_RTOL = 1e-11  # derived above: 2.4e-13 with ~40x of slack

# CANARY -- pinned AFTER the first GREEN, at the values actually measured:
#   step   5 -> 2.360e-15
#   step  50 -> 1.713e-15
#   step 200 -> 1.934e-15
# One order of magnitude above the worst of them. The derived rtol is 4000x
# looser, so a drift from 2.4e-15 to 3e-14 would sail past it unnoticed; this
# fires first. Not a derived tolerance -- a drift detector. If it fires, the
# question is what changed in the cache, not which number to loosen.
#
# Worth noting: the incremental path lands in the SAME range as M0's batched
# parity error (2.52e-15). Fewer and smaller reductions did not buy accuracy,
# and did not cost any either.
LOGITS_CANARY = 1e-14


@pytest.fixture(scope="module")
def gen_ref():
    return np.load(GEN_FIXTURE)


@pytest.fixture(scope="module")
def eos_ref():
    return np.load(EOS_FIXTURE)


@pytest.fixture(scope="module")
def params():
    return load_weights(PARITY_FIXTURE)


@pytest.fixture(scope="module")
def step_fn(params):
    return lambda token, position, cache: forward_step(token, position, cache, params, N_HEAD)


@pytest.fixture(scope="module")
def full_fn(params):
    return lambda window: gpt_forward(window, params, n_head=N_HEAD)


# --------------------------------------------------------------- stubs


class ListCacheStub:
    """Cache state is just the list of ids seen so far -- opaque to the loop.

    `generate_with_cache` must never inspect the cache, so a plain list is a
    legitimate cache state. Recording `(position, len(cache_before))` is what
    makes the off-by-one testable with no forward at all.
    """

    def __init__(self, peak_id, vocab=VOCAB):
        self.peak_id = peak_id
        self.vocab = vocab
        self.positions = []
        self.lengths_before_insert = []
        self.cache_sizes_after_insert = []

    def __call__(self, token, position, cache):
        before = 0 if cache is None else len(cache)
        self.positions.append(position)
        self.lengths_before_insert.append(before)
        new_cache = ([] if cache is None else list(cache)) + [token]
        self.cache_sizes_after_insert.append(len(new_cache))
        logits = np.zeros(self.vocab, dtype=np.float64)
        logits[self.peak_id] = 1.0
        return logits, new_cache


class CallOrderStubs:
    """A step-fn and a full-fn that append to ONE shared ordered log.

    A shared log is what makes "no incremental call ever happens after the first
    full call" assertable; two separate counters could not express ordering.
    """

    def __init__(self, peak_id, vocab=VOCAB):
        self.peak_id = peak_id
        self.vocab = vocab
        self.calls = []

    def step(self, token, position, cache):
        self.calls.append("step")
        new_cache = ([] if cache is None else list(cache)) + [token]
        logits = np.zeros(self.vocab, dtype=np.float64)
        logits[self.peak_id] = 1.0
        return logits, new_cache

    def full(self, window):
        self.calls.append("full")
        logits = np.zeros((len(window), self.vocab), dtype=np.float64)
        logits[:, self.peak_id] = 1.0
        return logits


# ------------------------------------------------- (1) parity within block_size


def test_cache_matches_no_cache_within_block_size(gen_ref, step_fn, full_fn):
    """20 greedy tokens, total 30 -- pure incremental cache, the crop never fires.

    Compared against BOTH the M1 implementation and the frozen PyTorch fixture,
    because they are different claims: matching M1 proves the cache path agrees
    with our own no-cache path, matching the fixture proves both still agree with
    PersonaCore. A shared bug in M0's forward would satisfy the first and fail
    the second.
    """
    prompt = gen_ref["prompt_ids"]
    max_new = int(gen_ref["short_max_new_tokens"])
    block_size = int(gen_ref["block_size"])
    eos_id = int(gen_ref["eos_id"])

    cached = generate_with_cache(
        step_fn, full_fn, prompt, max_new_tokens=max_new, eos_id=eos_id, block_size=block_size
    )
    no_cache = generate(
        full_fn, prompt, max_new_tokens=max_new, eos_id=eos_id, block_size=block_size
    )

    assert len(prompt) + max_new <= block_size, "this run crosses block_size -- wrong fixture"
    assert cached == no_cache
    assert cached == gen_ref["short_generated_ids"].tolist()


# ------------------------------------------- (2) parity crossing block_size


@pytest.mark.slow
def test_cache_matches_no_cache_crossing_block_size(gen_ref, step_fn, full_fn):
    """280 tokens, total 290 -- incremental until 256, then delegated to M1.

    Marked `slow`: stays in the normal suite, excluded from the mutation loop.
    Measured BEFORE excluding it: it participates in 35 of the 51 kills and is the
    SOLE killer of none. With both slow tests gone, the mutants it covered most
    thinly are GE-1, GE-2 and KV-2, which keep 2 killers each.

    Two regimes in one run, so this is the test that proves the HANDOVER is
    exact: the fallback has to pick up the sequence in the state the cache left
    it, with the crop applied from that point on. An off-by-one in where the
    handover happens shows up as a divergent token here and nowhere else.
    """
    prompt = gen_ref["prompt_ids"]
    max_new = int(gen_ref["long_max_new_tokens"])
    block_size = int(gen_ref["block_size"])
    eos_id = int(gen_ref["eos_id"])

    cached = generate_with_cache(
        step_fn, full_fn, prompt, max_new_tokens=max_new, eos_id=eos_id, block_size=block_size
    )

    assert len(prompt) + max_new > block_size, "this run does not cross block_size -- bad fixture"
    assert cached == gen_ref["long_generated_ids"].tolist()


# ------------------------------------------ (2b) EOS-stop parity, cached path


def test_generate_with_cache_stops_on_real_eos_matches_personacore(eos_ref, step_fn, full_fn):
    """The same real-EOS fixture, through the cached path. Exact integer equality.

    Not redundant with the M1 twin: the EOS check lives in a DIFFERENT loop here.
    `generate_with_cache` owns its own copy of "argmax, compare to eos_id, return
    before appending", and a bug in that copy -- returning after the append,
    comparing against the wrong variable, checking before the prefill catches up
    -- would be invisible to every M1 test and to every other test in this file,
    since none of them ever reach an EOS.

    Stop at step 16 with block_size=256 means the whole run stays in the cached
    regime, so this is specifically the incremental loop's EOS handling under
    test, not the fallback's (the fallback's is M1's, already covered by the twin).
    """
    prompt = eos_ref["prompt_ids"]
    expected = eos_ref["generated_ids"].tolist()
    eos_id = int(eos_ref["artificial_eos_id"])
    max_new = int(eos_ref["max_new_tokens"])
    block_size = int(eos_ref["block_size"])

    got = generate_with_cache(
        step_fn, full_fn, prompt, max_new_tokens=max_new, eos_id=eos_id, block_size=block_size
    )

    assert got == expected
    assert len(got) == int(eos_ref["eos_step"]) < max_new, "this run did not stop early"
    assert eos_id not in got, "EOS must not be emitted"
    assert len(prompt) + max_new <= block_size, "this run left the cached regime -- wrong fixture"


# ---------------------------------------------- (3) the off-by-one, isolated


def test_cache_position_index_is_cache_length_before_insert(gen_ref):
    """The position handed to the step fn is the cache size BEFORE the insert.

    No forward runs here at all. If the cache already holds 10 entries at
    positions 0..9, the next token is position 10 -- the length before inserting,
    not after. Off by one in the "after" direction skips wpe[0] entirely and
    shifts every token's positional embedding by one slot; off by one the other
    way makes two tokens share a position.

    Asserted three ways because they fail differently: the recorded positions
    equal the recorded pre-insert lengths (the invariant itself), the positions
    are exactly 0..N-1 (no gaps, no repeats), and the first position is 0 (a
    1-based implementation would start at 1 and satisfy neither of the others).
    """
    block_size = int(gen_ref["block_size"])
    stub = ListCacheStub(peak_id=77)

    generate_with_cache(
        stub,
        lambda window: np.zeros((len(window), VOCAB)),
        np.array([11, 22, 33, 44, 55], dtype=np.int64),
        max_new_tokens=4,
        eos_id=int(gen_ref["eos_id"]),
        block_size=block_size,
    )

    assert stub.positions == stub.lengths_before_insert
    assert stub.positions == list(range(len(stub.positions)))
    assert stub.positions[0] == 0
    # And the complement: after inserting, the size is length-before + 1.
    assert stub.cache_sizes_after_insert == [n + 1 for n in stub.lengths_before_insert]


# --------------------------------- (4) cached logits vs full recompute, floats


@pytest.mark.parametrize("step", [5, 50, 200])
def test_cached_logits_match_full_recompute_per_step(gen_ref, params, step_fn, full_fn, step):
    """Incremental logits vs a full forward of the same window, at three depths.

    Steps 5, 50 and 200 are all in the PRE-CROP regime on purpose. Past 246 the
    implementation IS the full recompute, so comparing there would be comparing a
    function against itself and would pass no matter what the cache does.

    The two paths do the same arithmetic in a different order (matrix-vector vs
    matrix-matrix), so bit equality is not expected. Tolerance derived in the
    module docstring.
    """
    prompt = gen_ref["prompt_ids"]
    idx = np.asarray(prompt, dtype=np.int64)

    cache = None
    for pos in range(len(idx)):
        cached_logits, cache = step_fn(int(idx[pos]), pos, cache)

    for _ in range(step):
        next_id = int(np.argmax(cached_logits))
        idx = np.append(idx, next_id)
        cached_logits, cache = step_fn(next_id, len(idx) - 1, cache)

    assert len(idx) <= int(gen_ref["block_size"]), "sample step left the pre-crop regime"

    full_logits = full_fn(idx)[-1]
    err = np.abs(cached_logits - full_logits).max() / np.abs(full_logits).max()
    print(f"\n[cache step {step}] logits relative error: {err:.3e}")

    assert err < LOGITS_RTOL, f"error {err:.3e} -- investigate the cache, not the tolerance"
    assert err < LOGITS_CANARY, f"error {err:.3e} rose above the measured ~2.4e-15 -- what changed?"
    assert int(np.argmax(cached_logits)) == int(np.argmax(full_logits))


# ------------------------------------------------- (5) fallback at the boundary


def test_cache_falls_back_to_full_recompute_after_crop_boundary(gen_ref):
    """Once the window would exceed block_size, every remaining step is a full one.

    Not just "a full call happens" -- the stronger claim is that ZERO incremental
    calls happen after the first full one. An implementation that tried to rebuild
    the cache after the crop would interleave step calls into the tail, and the
    ordering assertion below is the only thing in this file that would notice.
    """
    block_size, prompt_len, max_new = 8, 5, 10
    stubs = CallOrderStubs(peak_id=3)

    generate_with_cache(
        stubs.step,
        stubs.full,
        np.arange(prompt_len, dtype=np.int64),
        max_new_tokens=max_new,
        eos_id=int(gen_ref["eos_id"]),
        block_size=block_size,
    )

    assert "full" in stubs.calls, "the fallback never fired"
    assert "step" in stubs.calls, "the incremental path never ran"

    first_full = stubs.calls.index("full")
    assert "step" not in stubs.calls[first_full:], f"incremental call after fallback: {stubs.calls}"

    # The boundary is exact: steps run while len(idx) <= block_size, so with a
    # 5-token prompt the incremental regime covers lengths 5,6,7,8 -- four steps,
    # the first of which prefills all five prompt tokens.
    assert stubs.calls[:first_full] == ["step"] * (prompt_len + 3)
    assert stubs.calls[first_full:] == ["full"] * (max_new - 4)


# --------------------------------------------- (6) the cache never overgrows


def test_cache_never_grows_past_block_size(gen_ref):
    """The cache is never allowed past block_size entries, not even for one step.

    The failure this guards against is subtle: an implementation could produce
    correct tokens (because it falls back on the NEXT step) while briefly holding
    a block_size+1 cache. That state is meaningless -- the model has no wpe row
    for it -- and it is the shape a "grow first, check later" loop takes.
    """
    block_size, prompt_len, max_new = 8, 5, 10
    stub = ListCacheStub(peak_id=3)

    generate_with_cache(
        stub,
        lambda window: np.zeros((len(window), VOCAB)),
        np.arange(prompt_len, dtype=np.int64),
        max_new_tokens=max_new,
        eos_id=int(gen_ref["eos_id"]),
        block_size=block_size,
    )

    assert stub.cache_sizes_after_insert, "the incremental path never ran"
    assert max(stub.cache_sizes_after_insert) <= block_size, stub.cache_sizes_after_insert
    # And it really did reach the limit -- otherwise a loop that bails out early
    # would pass the assertion above without ever exercising the boundary.
    assert max(stub.cache_sizes_after_insert) == block_size


# ------------------------------------------------------------- housekeeping


def test_cache_length_reports_zero_for_an_empty_cache(gen_ref, step_fn):
    """`cache_length(None)` is 0, and it grows by one per step. Used by the tests above."""
    assert cache_length(None) == 0

    _, cache = step_fn(11, 0, None)
    assert cache_length(cache) == 1
    _, cache = step_fn(22, 1, cache)
    assert cache_length(cache) == 2


def test_generate_with_cache_rejects_sampling_because_m2_is_greedy_only(gen_ref):
    """`greedy=False` still raises here -- the cached path is greedy-only by design.

    No longer "same as M1": M4 gave `engine.generate.generate` real sampling and
    deliberately did not extend it to this path. So this is not a not-yet-built
    fence, it is a live asymmetry, and the raise is what keeps a caller from
    believing the cached loop sampled when it did not. Silent fallback to greedy
    would be the dangerous alternative.
    """
    with pytest.raises(NotImplementedError):
        generate_with_cache(
            ListCacheStub(peak_id=1),
            lambda window: np.zeros((len(window), VOCAB)),
            np.array([1, 2], dtype=np.int64),
            max_new_tokens=1,
            eos_id=int(gen_ref["eos_id"]),
            block_size=int(gen_ref["block_size"]),
            greedy=False,
        )
