# personacore-np

A pure-NumPy inference engine for the PersonaCore GPT — a 13.9M-parameter
decoder-only transformer. No PyTorch, no autograd, no computation graph: just
arrays and the arithmetic the model actually needs at inference time.

The point is not speed. The point is that every operation is written out by hand
and proven to agree with the PyTorch implementation it mirrors.

## Status

| Milestone | Scope | Proof |
|---|---|---|
| **M0** | Forward pass — LayerNorm, GELU, causal attention, MLP, tied `lm_head`, cross-entropy | Logits and loss vs. a frozen PyTorch reference, float64 |
| **M1** | Greedy generation — sliding context window, EOS stop | Generated token ids, exact integer equality |
| **M2** | Incremental KV-cache, falling back to full recompute once the window slides | Token ids vs. both the no-cache path and PyTorch; cached logits vs. full recompute |
| **M3** | Batched forward over right-padded sequences | Batched logits vs. the per-sequence forward; padding and cross-row isolation asserted at exactly zero |
| **M4** | Temperature / top-k / top-p sampling | Structural: formulas by hand, support sets, filter order, entropy monotonicity |
| **M5** | Int8 weight quantization — symmetric, per-channel, fake-quant | Measurement with control: perplexity vs. baseline vs. a deliberately coarse int4 scheme |
| **M6** | MLX (Metal GPU) port of the forward, f32 | Transitive parity vs. the NumPy engine (1.2e-6, argmax-exact) + measured 15x wall-clock speedup |

All seven complete. **86 tests**, all passing.

Parity is argmax-exact on token ids and ~1e-15 relative on logits — the residual
is float64 operation-order noise, not approximation.

M4 is the exception, deliberately. numpy's RNG and torch's RNG are different
algorithms, so no seed makes them draw the same token and chasing an identical
sample would be chasing a coincidence. Its tests prove structure instead: the
formula written out by hand, the surviving support after each filter, the order
the filters compose in, and entropy rising monotonically with temperature.

### The one thing M2 does not do

A KV-cache over *learned absolute* positional embeddings (no RoPE) stops being
reusable the moment the context window slides: every surviving token shifts one
slot down in `wpe`, and its attention context gets truncated. Rebuilding the
cache costs a full forward anyway. So the cache runs while the window fits in
`block_size` and hands the rest of the generation to the M1 path.

That is a property of the model, not a defect in the cache. The derivation, with
the measured evidence, is in the module docstring of `tests/test_cache.py`.

### And the one thing M3 does not need

Batching pads sequences to a rectangle, which normally calls for a padding mask:
zeroing a pad token's value is not enough, because `exp(score_pad)` stays in the
shared softmax denominator and dilutes the weights on the real tokens.

With **right**-padding that cannot happen. Every pad position sits in the future
of every real query, so the causal mask already blocks it — measured on the real
checkpoint, rewriting the padding with arbitrary tokens moves the real logits by
exactly `0.000e+00`. Under left-padding the same perturbation moves them by
`7.399e-01`. So there is no padding mask here, and `tests/test_batched.py` proves
the invariant rather than assuming it.

### What M6 adds — and deliberately does not

`engine/forward_mlx.py` is a 1:1 port of the forward to MLX (Apple Metal GPU),
float32 because Metal has no float64. Its oracle is the NumPy engine itself —
already proven against PyTorch — so parity is transitive and PyTorch is never
re-tested. Measured on an M3 Pro: relative error 1.2e-6 against NumPy f64
(the derived f32 estimate was 1.3e-4; the same LayerNorm damping M0 measured
pushed it ~100x lower), argmax identical at all 256 positions, and the
256-token forward drops from 63.1 ms (NumPy f64 CPU) to 4.2 ms — **15x**.
Only the pure forward is ported: cache and sampling are light bookkeeping
that the GPU would not accelerate, and they stay in NumPy. `mlx` is an
optional extra (`pip install -e ".[mlx]"`); without it the MLX tests skip
cleanly and nothing else changes.

### How M5 proves quality without a formula

Quantization error has no analytic tolerance the way float accumulation does,
so M5's claim is relative, measured against a control: on a real 256-token
window of val.bin, int8 per-channel moves perplexity by −0.0009 against the
float64 baseline (2.4280 → 2.4270 — noise), while a deliberately coarse
simulated-int4 control moves it by +0.165 — a 175x wider gap. Every scheme
decision (symmetric, per-channel, LayerNorm excluded, tied wte included) was
made from measured statistics of this checkpoint, written down in
`engine/quantize.py` before the tests ran. Measured compression: 7.95x on the
eligible tensors, 7.83x on the whole model.

## Discipline

- **TDD, red-first.** Every test was watched failing before the code existed.
  Where a genuine red was structurally unavailable (the oracle already existed),
  it was replaced by a revert cycle: break the production code, confirm the test
  dies, restore.
- **Numerical tolerance derived in writing before measuring.** The estimate comes
  from a term-count model of the critical path (`sqrt(K)·eps` per reduction,
  summed); the assertion sits above it with slack; a second, tighter canary is
  pinned at the value actually measured, to catch drift the loose bound would
  miss. Tolerances are never loosened to make a test pass.
- **Replicate the source, not the textbook.** Where PersonaCore's sampling
  diverges from the common convention, this engine follows PersonaCore. Its
  temperature is floored rather than validated, so `0` and negatives both sharpen
  instead of raising; its top-k cutoff is a strict `<`, so ties at the boundary
  push the surviving set past `k`; its nucleus keeps the token that crosses `p`.
  Each was read off the source and measured against the real torch functions
  before being written here.
- **Manual, targeted mutation testing.** `scripts/mutation_check.py` holds **76**
  hand-picked mutations, each attacking a specific architectural decision — ddof,
  eps inside vs. outside the sqrt, attention scale, mask ordering, head reshape,
  weight tying, crop direction, cache position index, batch-axis bookkeeping,
  filter order, quantization scale and rounding, MLX porting errors. **70
  killed, 6 alive**, and every survivor is accounted for:

  | Survivor | Why it lives |
  |---|---|
  | `GE-3` | Dead store — appends to a local immediately before a `return` that never reads it |
  | `KV-5` | No target. There is no causal mask in the cache path, because everything cached is strictly past. The recurring SKIP is the assertion |
  | `KV-7` | Provably equivalent: attention is a permutation-invariant weighted sum, and position is baked into K,V at creation. Measured diff 4.4e-15, identical argmax |
  | `PD-6` | No target. Cross-row isolation is not enforced by a line of code — it is what `@` means on arrays with leading axes |
  | `QT-2` | Provably equivalent: with the max-based scale `s = f32(amax/127)`, `\|w/s\| ≤ 127/(1−2⁻²⁴) ≈ 127.0000076`, which rounds to 127 — the clip never fires. Measured: worst `\|rint(w/s)\|` across all 38 eligible tensors is exactly 127. The clip guards scale bugs (compose it with `QT-1` and it matters), not correct code |
  | `MX-3` | No target. The Passo 0.3 probe found no operation where MLX broadcasting diverges from NumPy (add, where, keepdims-reductions all NumPy-identical), so there is no line to mutate. The recurring SKIP is the assertion |

  Two of the mutations target a *test* rather than the engine, because the one
  test whose job is to detect a difference between two configurations is the one
  that could silently compare something to itself. Both survived on the first
  run and exposed a real gap, which is what they were for.

  Run it with `python scripts/mutation_check.py`. It skips two long parity tests,
  measured beforehand to be the sole killer of nothing.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

Without fixtures (see below) the pure-NumPy unit tests still run and the parity
tests skip with a message naming the generator to run. Nothing fails, nothing
errors.

## Reproducing the parity suite

**`fixtures/` is not included in this repository.** The parity oracles are ~157 MB
of `.npz` derived from a trained PersonaCore checkpoint, and the directory was a
symlink to a location on one machine.

Regenerating them needs two things this repository does not ship:

1. A **trained PersonaCore checkpoint**.
2. The fixture generators, which live in a **companion repository (`tensorforge`)
   that is private** — access to it is required, and cloning this repository does
   not grant it.

With both, the generators write `personacore_parity.npz`,
`personacore_generate_fixture.npz`, `personacore_eos_fixture.npz` and
`val_windows.npz` into a `fixtures/` directory here, and the 51 fixture-bound
tests run alongside the rest.

Without them, the engine and its unit tests still stand on their own.
