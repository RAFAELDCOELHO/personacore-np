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

All three complete. **46 tests**, all passing.

Parity is argmax-exact on token ids and ~1e-15 relative on logits — the residual
is float64 operation-order noise, not approximation.

### The one thing M2 does not do

A KV-cache over *learned absolute* positional embeddings (no RoPE) stops being
reusable the moment the context window slides: every surviving token shifts one
slot down in `wpe`, and its attention context gets truncated. Rebuilding the
cache costs a full forward anyway. So the cache runs while the window fits in
`block_size` and hands the rest of the generation to the M1 path.

That is a property of the model, not a defect in the cache. The derivation, with
the measured evidence, is in the module docstring of `tests/test_cache.py`.

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
- **Manual, targeted mutation testing.** `scripts/mutation_check.py` holds **54**
  hand-picked mutations, each attacking a specific architectural decision — ddof,
  eps inside vs. outside the sqrt, attention scale, mask ordering, head reshape,
  weight tying, crop direction, cache position index. **51 killed, 3 alive**, and
  every survivor is accounted for:

  | Survivor | Why it lives |
  |---|---|
  | `GE-3` | Dead store — appends to a local immediately before a `return` that never reads it |
  | `KV-5` | No target. There is no causal mask in the cache path, because everything cached is strictly past. The recurring SKIP is the assertion |
  | `KV-7` | Provably equivalent: attention is a permutation-invariant weighted sum, and position is baked into K,V at creation. Measured diff 4.4e-15, identical argmax |

  Run it with `python scripts/mutation_check.py`. It skips two long parity tests,
  measured beforehand to be the sole killer of nothing.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

Without fixtures (see below) this gives `13 passed, 33 skipped` — the pure-NumPy
unit tests run, and the parity tests skip with a message naming the generator to
run. Nothing fails, nothing errors.

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
`personacore_generate_fixture.npz` and `personacore_eos_fixture.npz` into a
`fixtures/` directory here, and the full 46-test suite runs.

Without them, the engine and its unit tests still stand on their own.
