"""M6: the MLX (Metal GPU) port of the forward, proven against the NumPy engine.

There is NO new external oracle here. The reference is engine/forward.py itself,
already proven against PyTorch in M0/M3 — so "MLX matches NumPy" is
transitively "MLX matches PyTorch", without re-testing against PyTorch (that
would be redundant and slower).

Scope (decided in Passo 0, before any code): ONLY the pure forward (M0 + the
M3 batch support). No KV-cache, no sampling, no quantization in this pass —
the forward is where a GPU can actually win over NumPy CPU; cache and sampling
are light bookkeeping that does not justify a port until the speed gain is
confirmed by test 5.

dtype decision: FLOAT32. Measured in Passo 0.3: `mx.float64` constructs, but
any GPU op raises `ValueError: float64 is not supported on the GPU` — running
f64 would force the CPU stream and defeat the milestone. The weights lose
nothing: the checkpoint is fp32, so f64 -> f32 on load is bit-exact.

===========================================================================
TOLERANCE DERIVATION for f32 — written BEFORE any measurement (M0 discipline)
===========================================================================

TWO error sources now, where M0 had one:

1. ACCUMULATION (same term-count model as M0, new eps). The critical path of
   one logit sums to 1084.8 units of eps (derivation in test_parity.py). In
   f32, eps = 2^-23 = 1.19e-7:

       1084.8 * 1.19e-7 = 1.29e-4   <-- dominant term

2. INTERMEDIATE ROUNDING. The weights are bit-exact (fp32 checkpoint), but
   every intermediate is rounded to f32 at each of the ~57 op-stages on the
   path (9 contractions + norms/gelu/softmax stages x 6 blocks + head), each
   contributing <= eps/2 relative:

       ~57 * 6.0e-8 = 3.4e-6        <-- two orders below source 1

   DERIVED ESTIMATE ~ 1.3e-4 relative (max|diff| / max|ref|).

The same two forces as M0 push the real value below the estimate (LayerNorm
renormalizes the residual stream every block, and the metric normalizes by the
global max). In M0 that damping made the measured value ~200x smaller than the
estimate. The assertion sits at 1e-3 (estimate with ~8x slack); a tighter
canary is pinned at the measured value after GREEN, to catch drift.
"""

import os
import time

import numpy as np
import pytest

mx = pytest.importorskip(
    "mlx.core",
    reason="mlx not installed — Apple-Silicon-only optional extra: pip install -e '.[mlx]'",
    exc_type=ImportError,
)

from engine.forward import gpt_forward  # noqa: E402  (import must follow the skip)
from engine.forward_mlx import gpt_forward_mlx, to_mlx_params  # noqa: E402
from engine.weights import load_weights  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures", "personacore_parity.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE),
    reason="fixtures/personacore_parity.npz absent — run the tensorforge generator "
    "gen_parity_fixture.py against a trained checkpoint to produce it",
)

N_HEAD = 6
DERIVED_TOL = 1e-3  # 1.3e-4 estimate, ~8x slack — see module docstring
# Pinned AFTER the first GREEN at the measured value (1.205e-6, ~100x below the
# estimate — the same LayerNorm damping M0 measured) with ~4x headroom for
# run-to-run GPU scheduling noise. Drift past this means the port changed.
CANARY = 5e-6


@pytest.fixture(scope="module")
def np_params():
    return load_weights(FIXTURE)


@pytest.fixture(scope="module")
def mlx_params(np_params):
    return to_mlx_params(np_params)


@pytest.fixture(scope="module")
def input_x():
    return np.load(FIXTURE)["input_x"]


def relative_error(got, expected):
    """Same metric as M0: global max abs diff over global max abs reference."""
    return np.abs(got - expected).max() / np.abs(expected).max()


# ---------------------------------------------------------------------------
# 1. the device really is the GPU, not a silent CPU fallback
# ---------------------------------------------------------------------------


def test_mlx_available_and_uses_gpu_device():
    assert mx.metal.is_available(), "Metal not available — MLX would silently run on CPU"
    dev = mx.default_device()
    assert dev == mx.Device(mx.DeviceType.gpu), f"default device is {dev}, not the GPU"


# ---------------------------------------------------------------------------
# 2. single-sequence parity against the NumPy engine (the transitive oracle)
# ---------------------------------------------------------------------------


def test_mlx_forward_matches_numpy_forward_single_sequence(np_params, mlx_params, input_x):
    expected = gpt_forward(input_x, np_params, n_head=N_HEAD)
    got = np.asarray(gpt_forward_mlx(input_x, mlx_params, n_head=N_HEAD), dtype=np.float64)
    assert got.shape == expected.shape
    rel = relative_error(got, expected)
    assert rel < DERIVED_TOL, f"measured {rel:.3e} vs derived bound {DERIVED_TOL:.0e}"
    if CANARY is not None:
        assert rel < CANARY, f"drift: measured {rel:.3e} vs canary {CANARY:.3e}"


# ---------------------------------------------------------------------------
# 3. batched parity — the M3 right-padded shape survives the port
# ---------------------------------------------------------------------------


def test_mlx_forward_matches_numpy_forward_batched(np_params, mlx_params, input_x):
    """Same right-padding convention as M3: row 0 full, row 1 shorter and padded
    with zeros to the rectangle. Both engines compute the pad rows too, and the
    comparison covers the FULL logits — pad positions included — because the two
    engines must agree everywhere, not just where the tokens are real."""
    t_short = 100
    batch = np.zeros((2, input_x.shape[0]), dtype=np.int64)
    batch[0] = input_x
    batch[1, :t_short] = input_x[:t_short]

    expected = gpt_forward(batch, np_params, n_head=N_HEAD)
    got = np.asarray(gpt_forward_mlx(batch, mlx_params, n_head=N_HEAD), dtype=np.float64)
    assert got.shape == expected.shape
    rel = relative_error(got, expected)
    assert rel < DERIVED_TOL, f"measured {rel:.3e} vs derived bound {DERIVED_TOL:.0e}"


# ---------------------------------------------------------------------------
# 4. argmax parity — exact, position by position
# ---------------------------------------------------------------------------


def test_mlx_argmax_matches_numpy_argmax(np_params, mlx_params, input_x):
    """Independent of the continuous tolerance: the greedy decision must agree
    exactly. If f32 noise flips the argmax anywhere, the assertion message
    reports HOW MANY of the 256 positions diverge — that count is information,
    not just a boolean."""
    expected = gpt_forward(input_x, np_params, n_head=N_HEAD).argmax(axis=-1)
    got = np.asarray(gpt_forward_mlx(input_x, mlx_params, n_head=N_HEAD)).argmax(axis=-1)
    mism = int((got != expected).sum())
    assert mism == 0, f"{mism}/{len(expected)} positions pick a different argmax under f32"


# ---------------------------------------------------------------------------
# 5. the reason this milestone exists: wall-clock speed
# ---------------------------------------------------------------------------


@pytest.mark.slow  # ~5s of timing loops; excluded from the mutation loop, where
# speed discriminates nothing — every MX mutant is killed by the parity tests.
def test_mlx_forward_is_faster_than_numpy_forward(np_params, mlx_params, input_x):
    """Wall clock, 10 timed reps after 3 warmups (MLX compiles kernels lazily on
    first use; timing the warmup would charge compilation to every call).
    `mx.eval` inside the timed region — MLX is lazy, and without forcing
    evaluation the "forward" would time the graph construction only.

    If MLX is NOT faster (13.9M params may be too small to amortize GPU
    dispatch), this failure IS the milestone's central result — the assertion
    message carries both timings either way."""

    def time_fn(fn, reps=10, warmup=3):
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        return float(np.mean(samples)), float(np.std(samples))

    def run_np():
        gpt_forward(input_x, np_params, n_head=N_HEAD)

    idx_mx = mx.array(input_x)

    def run_mlx():
        mx.eval(gpt_forward_mlx(idx_mx, mlx_params, n_head=N_HEAD))

    np_mean, np_std = time_fn(run_np)
    mlx_mean, mlx_std = time_fn(run_mlx)
    report = (
        f"numpy f64 CPU: {np_mean * 1e3:.2f} ± {np_std * 1e3:.2f} ms | "
        f"mlx f32 GPU: {mlx_mean * 1e3:.2f} ± {mlx_std * 1e3:.2f} ms | "
        f"speedup {np_mean / mlx_mean:.2f}x"
    )
    assert mlx_mean < np_mean, report
