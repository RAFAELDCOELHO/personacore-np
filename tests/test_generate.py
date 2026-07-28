"""M1: greedy generation, no KV-cache, full forward recomputed every step.

Parity oracle: `fixtures/personacore_generate_fixture.npz`, frozen by
~/tensorforge/scripts/gen_generate_fixture.py from the real PersonaCore
`generate()` (checkpoint best.pt, float64, attn_impl="manual", greedy=True,
called directly on `personacore.generation.core.generate` — no `generate_text`
wrapper, no forbid_ids, no eos_id prefix).

NO TOLERANCE ANYWHERE IN THIS FILE. Everything generation produces is an integer
token id; argmax either lands on the same id or it does not. `list == list` is
the whole assertion. Float tolerance belongs in test_parity.py, where logits
live.

Loop-control tests use INJECTED STUBS, not the real forward. That is the entire
point of `generate` taking `forward_fn` as a parameter: the control (crop,
EOS-stop, argmax, position selection) is testable without the checkpoint and
without depending on the model producing an EOS by luck — which, per the fixture,
it never does in 300 tokens.

Measured coverage note (recorded because it decides which test catches what):
the crop only fires once `len(idx) > block_size`, i.e. from step 247 of the long
run onward, and from that point the window is ALWAYS exactly 256 — where
`wpe[:256] == wpe[-256:]`. So the GP-2-class bug (absolute vs window-relative
position) is invisible in the cropped region and is actually caught by the SHORT
test, where T runs 10..30. The long test earns its keep on the crop itself, not
on positions.
"""

import os

import numpy as np
import pytest

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
def real_forward(params):
    """The real forward, injected. Loop-control tests deliberately do NOT use this."""
    return lambda window: gpt_forward(window, params, n_head=N_HEAD)


# --------------------------------------------------------------- stubs


class RecordingStub:
    """Records the shape of every window it receives; returns a fixed-argmax logit block.

    The peak sits at a constant id, so the generated token is predictable without
    the checkpoint. Every row of the returned (T, VOCAB) block is identical —
    which is exactly why this stub CANNOT catch a wrong-position bug, and why the
    parity tests carry that job instead.
    """

    def __init__(self, peak_id, vocab=VOCAB):
        self.peak_id = peak_id
        self.vocab = vocab
        self.window_shapes = []

    def __call__(self, window):
        self.window_shapes.append(np.asarray(window).shape)
        logits = np.zeros((len(window), self.vocab), dtype=np.float64)
        logits[:, self.peak_id] = 1.0
        return logits


class EosOnStepStub:
    """Returns an ordinary peak until `eos_on_call`, then peaks on eos_id.

    Call-counter driven so the EOS moment is exact and does not depend on the
    checkpoint ever emitting an EOS (per the fixture, it does not).
    """

    def __init__(self, eos_id, eos_on_call, ordinary_id, vocab=VOCAB):
        self.eos_id = eos_id
        self.eos_on_call = eos_on_call
        self.ordinary_id = ordinary_id
        self.vocab = vocab
        self.calls = 0

    def __call__(self, window):
        self.calls += 1
        logits = np.zeros((len(window), self.vocab), dtype=np.float64)
        peak = self.eos_id if self.calls == self.eos_on_call else self.ordinary_id
        logits[:, peak] = 1.0
        return logits


# ------------------------------------------------------------ (1) short parity


def test_generate_matches_personacore_short(gen_ref, real_forward):
    """20 greedy tokens against PyTorch, exact integer equality.

    Total length stays at 30, well under block_size=256, so the crop never fires
    once. What this test does cover, and the long one cannot, is the
    window-relative positional embedding: T runs 10..30 here, and in that range
    `wpe[:T]` and `wpe[-T:]` are different slices.
    """
    prompt = gen_ref["prompt_ids"]
    expected = gen_ref["short_generated_ids"].tolist()
    max_new = int(gen_ref["short_max_new_tokens"])

    got = generate(
        real_forward,
        prompt,
        max_new_tokens=max_new,
        eos_id=int(gen_ref["eos_id"]),
        block_size=int(gen_ref["block_size"]),
    )

    assert got == expected
    assert len(got) == max_new, "the fixture recorded no EOS in this run"


# ------------------------------------------------- (2) long parity, crosses block_size


@pytest.mark.slow
def test_generate_matches_personacore_crosses_block_size(gen_ref, real_forward):
    """280 greedy tokens, total 290 — the crop fires on every step from 247 on.

    Marked `slow`: it stays in the normal suite but is excluded from the mutation
    loop, which reruns the whole suite 54 times. Measured BEFORE excluding it: it
    participates in 32 of the 51 kills and is the SOLE killer of none, so the
    loop's kill count is unchanged. With both slow tests gone the thinnest
    dependant is GE-7 (crop from the wrong end), which drops from 3 killers to 1
    and stays covered by
    `test_generate_position_embedding_uses_window_index_after_crop`.

    Deliberately NOT skipped as redundant with the short test: different length,
    different window regime. Removing the crop makes the window grow past 256 and
    `wpe[:T]` stops broadcasting; cropping from the wrong end changes the tail.
    Neither failure is reachable from a 30-token run.
    """
    prompt = gen_ref["prompt_ids"]
    expected = gen_ref["long_generated_ids"].tolist()
    max_new = int(gen_ref["long_max_new_tokens"])
    block_size = int(gen_ref["block_size"])

    got = generate(
        real_forward,
        prompt,
        max_new_tokens=max_new,
        eos_id=int(gen_ref["eos_id"]),
        block_size=block_size,
    )

    assert len(prompt) + max_new > block_size, "this run does not cross block_size — bad fixture"
    assert got == expected


# ------------------------------------------------------------------ (3) crop


def test_generate_crops_window_before_forward_call(gen_ref):
    """No forward call may ever see a window longer than block_size — including the first.

    The prompt is 300 tokens, already over block_size, so the very FIRST call has
    to be cropped. A crop written as "only after we start appending" would let
    the first call through at 300 and this catches it.
    """
    block_size = int(gen_ref["block_size"])
    prompt = np.arange(300, dtype=np.int64) % VOCAB
    stub = RecordingStub(peak_id=42)

    got = generate(
        stub, prompt, max_new_tokens=3, eos_id=int(gen_ref["eos_id"]), block_size=block_size
    )

    assert len(stub.window_shapes) == 3, "one forward call per generated token"
    assert all(shape[-1] <= block_size for shape in stub.window_shapes), stub.window_shapes
    assert stub.window_shapes[0][-1] == block_size, "the first call was not cropped"
    assert got == [42, 42, 42]


def test_generate_does_not_crop_when_the_window_still_fits(gen_ref):
    """Below block_size the window must grow untouched — catches an always-on crop.

    The counterpart to the test above: a crop applied unconditionally would slice
    a 5-token prompt down and this is the only test that would notice.
    """
    block_size = int(gen_ref["block_size"])
    stub = RecordingStub(peak_id=7)

    generate(
        stub,
        np.array([1, 2, 3, 4, 5], dtype=np.int64),
        max_new_tokens=3,
        eos_id=int(gen_ref["eos_id"]),
        block_size=block_size,
    )

    assert [shape[-1] for shape in stub.window_shapes] == [5, 6, 7]


# ------------------------------------------------------------------- (4) EOS


def test_generate_stops_on_eos_without_appending_or_yielding(gen_ref):
    """EOS terminates BEFORE the append and BEFORE the emit — it is never in the output.

    The stub peaks on an ordinary id on call 1 and on eos_id on call 2, so the
    EOS moment is exact and independent of the checkpoint (which, per the
    fixture, produced no EOS in 300 tokens).

    Three distinct assertions, because three distinct bugs live here: emitting
    the EOS (list of 2 ending in eos_id), appending it and continuing (list of 5),
    and stopping one token too early (empty list).
    """
    eos_id = int(gen_ref["eos_id"])
    stub = EosOnStepStub(eos_id=eos_id, eos_on_call=2, ordinary_id=123)

    got = generate(
        stub,
        np.array([1, 2, 3], dtype=np.int64),
        max_new_tokens=5,
        eos_id=eos_id,
        block_size=int(gen_ref["block_size"]),
    )

    assert got == [123], "EOS must stop after exactly one emitted token"
    assert eos_id not in got
    assert stub.calls == 2, "the loop must stop on the EOS call, not keep going"


def test_generate_stops_on_real_eos_matches_personacore(eos_ref, real_forward):
    """EOS-stop against the real checkpoint, not a stub. Exact integer equality.

    Every other EOS test in this file drives a synthetic stub, so until this
    fixture existed the EOS path had never been compared to PyTorch in ANY
    milestone -- both generation fixtures ran out of budget instead of stopping.

    The fixture uses an ARTIFICIAL eos_id (261, a token the model actually emits
    at step 16) because the real one (8184) is never the argmax in a short window.
    `generate` treats eos_id as a pure stop comparison on both sides, so swapping
    it changes no logit -- it only decides where the loop ends.

    The `len(got) < max_new` assertion is what makes this a real EOS test rather
    than a second copy of the short-parity test: without it, an implementation
    that ignored eos_id entirely would still match on the first 16 tokens and
    only differ in the 14 it kept generating.
    """
    prompt = eos_ref["prompt_ids"]
    expected = eos_ref["generated_ids"].tolist()
    eos_id = int(eos_ref["artificial_eos_id"])
    max_new = int(eos_ref["max_new_tokens"])

    got = generate(
        real_forward,
        prompt,
        max_new_tokens=max_new,
        eos_id=eos_id,
        block_size=int(eos_ref["block_size"]),
    )

    assert got == expected
    assert len(got) == int(eos_ref["eos_step"]) < max_new, "this run did not stop early"
    assert eos_id not in got, "EOS must not be emitted"


def test_generate_runs_to_max_new_tokens_when_eos_never_comes(gen_ref):
    """Counter-proof to the test above: without EOS the loop uses its full budget.

    Without this, a `generate` that stopped after one token unconditionally would
    pass the EOS test.
    """
    stub = RecordingStub(peak_id=99)
    got = generate(
        stub,
        np.array([1, 2, 3], dtype=np.int64),
        max_new_tokens=5,
        eos_id=int(gen_ref["eos_id"]),
        block_size=int(gen_ref["block_size"]),
    )
    assert got == [99] * 5


# --------------------------------------------- (5) window-relative position, isolated


def test_generate_position_embedding_uses_window_index_after_crop(gen_ref, params, real_forward):
    """A 260-token prompt: generate(1) must equal a direct forward on the [-256:] window.

    The GP-2 property from M0, lifted to the loop level and proven WITHOUT the
    PyTorch fixture. If `generate` leaked an absolute position into the forward,
    or cropped from the wrong end, the two argmaxes would part ways.

    The prompt reuses real corpus ids (the parity fixture's `input_x`, extended by
    its own head) so every id is in-vocab and the logits are not degenerate.
    """
    block_size = int(gen_ref["block_size"])
    parity = np.load(PARITY_FIXTURE)
    prompt = np.concatenate([parity["input_x"], parity["input_x"][:4]])
    assert prompt.shape == (260,) and prompt.max() < VOCAB

    got = generate(
        real_forward,
        prompt,
        max_new_tokens=1,
        eos_id=int(gen_ref["eos_id"]),
        block_size=block_size,
    )

    window = prompt[-block_size:]
    expected = int(gpt_forward(window, params, n_head=N_HEAD)[-1].argmax())

    assert got == [expected]


# ------------------------------------------------------- forward_fn output shapes


def test_generate_accepts_batched_logits_shape(gen_ref):
    """forward_fn may return (T, vocab) or (1, T, vocab) — only the last row matters.

    The spec allows both shapes, so both are exercised. This is also the only
    test where an argmax taken on the wrong axis produces a wrong SHAPE rather
    than the same scalar: on (1, vocab), `argmax(axis=0)` returns a vocab-long
    array instead of one id.
    """
    vocab = 16

    def batched_forward(window):
        logits = np.zeros((1, len(window), vocab), dtype=np.float64)
        logits[0, :, 5] = 1.0
        return logits

    got = generate(
        batched_forward,
        np.array([1, 2], dtype=np.int64),
        max_new_tokens=2,
        eos_id=int(gen_ref["eos_id"]),
        block_size=int(gen_ref["block_size"]),
    )
    assert got == [5, 5]


def test_generate_reads_the_last_position_not_the_first(gen_ref):
    """A forward whose rows DISAGREE: row 0 peaks elsewhere, the last row decides.

    Every other stub in this file returns identical rows, so none of them can
    tell `logits[-1]` from `logits[0]`. This one can, and it is the only
    fixture-free test that does.
    """
    vocab = 16

    def disagreeing_forward(window):
        logits = np.zeros((len(window), vocab), dtype=np.float64)
        logits[:, 3] = 1.0  # every row peaks at 3 ...
        logits[-1, 3] = 0.0
        logits[-1, 11] = 2.0  # ... except the last, which peaks at 11
        return logits

    got = generate(
        disagreeing_forward,
        np.array([1, 2, 3], dtype=np.int64),
        max_new_tokens=1,
        eos_id=int(gen_ref["eos_id"]),
        block_size=int(gen_ref["block_size"]),
    )
    assert got == [11]
