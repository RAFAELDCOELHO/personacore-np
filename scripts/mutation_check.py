"""Manual mutation check — the same discipline as the tensorforge, which has no tooling.

The tensorforge applies mutations "line by line" by hand and requires the suite
to catch each one. This automates the APPLICATION (edit, run, restore), not the
judgement: every mutation below was hand-picked to attack a specific
architectural decision — ddof, eps placement, attention scale, mask ordering,
transposition, tying. A blind operator would generate hundreds of trivial
mutants and none of these.

Usage:  python scripts/mutation_check.py
Output: KILLED/SURVIVED per mutant, with the tests that killed each one.

The mutated file is always restored (try/finally).
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FORWARD = ROOT / "engine" / "forward.py"
WEIGHTS = ROOT / "engine" / "weights.py"
GENERATE = ROOT / "engine" / "generate.py"
CACHE = ROOT / "engine" / "cache.py"
TESTS_BATCHED = ROOT / "tests" / "test_batched.py"
SAMPLING = ROOT / "engine" / "sampling.py"
QUANTIZE = ROOT / "engine" / "quantize.py"
FORWARD_MLX = ROOT / "engine" / "forward_mlx.py"

# (id, file, description, original_snippet, mutated_snippet)
MUTATIONS = [
    # ---- LayerNorm ----
    ("LN-1", FORWARD, "population variance -> sample variance (ddof=1)",
     "var = x.var(axis=-1, keepdims=True)",
     "var = x.var(axis=-1, keepdims=True, ddof=1)"),
    ("LN-2", FORWARD, "eps INSIDE the sqrt -> OUTSIDE",
     "return (x - mean) / np.sqrt(var + eps) * weight + bias",
     "return (x - mean) / (np.sqrt(var) + eps) * weight + bias"),
    ("LN-3", FORWARD, "drop beta (the norm's bias)",
     "return (x - mean) / np.sqrt(var + eps) * weight + bias",
     "return (x - mean) / np.sqrt(var + eps) * weight"),
    ("LN-4", FORWARD, "drop gamma (the norm's weight)",
     "return (x - mean) / np.sqrt(var + eps) * weight + bias",
     "return (x - mean) / np.sqrt(var + eps) + bias"),
    ("LN-5", FORWARD, "eps 1e-5 -> 1e-12",
     "LN_EPS = 1e-5",
     "LN_EPS = 1e-12"),
    ("LN-6", FORWARD, "do not center (remove the mean subtraction)",
     "return (x - mean) / np.sqrt(var + eps) * weight + bias",
     "return x / np.sqrt(var + eps) * weight + bias"),

    # ---- GELU ----
    ("GL-1", FORWARD, "tanh-approx -> erf (the wrong formula for this checkpoint)",
     "    inner = np.sqrt(2.0 / np.pi) * (x + GELU_COEFF * x**3)\n"
     "    return 0.5 * x * (1.0 + np.tanh(inner))",
     "    import math\n"
     "    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))"),
    ("GL-2", FORWARD, "cubic coefficient 0.044715 -> 0.0",
     "GELU_COEFF = 0.044715",
     "GELU_COEFF = 0.0"),
    ("GL-3", FORWARD, "factor 0.5 -> 1.0",
     "return 0.5 * x * (1.0 + np.tanh(inner))",
     "return 1.0 * x * (1.0 + np.tanh(inner))"),
    ("GL-4", FORWARD, "sqrt(2/pi) -> 1.0",
     "inner = np.sqrt(2.0 / np.pi) * (x + GELU_COEFF * x**3)",
     "inner = 1.0 * (x + GELU_COEFF * x**3)"),
    ("GL-5", FORWARD, "gelu -> relu",
     "return 0.5 * x * (1.0 + np.tanh(inner))",
     "return np.maximum(x, 0.0)"),

    # ---- Attention: scale ----
    ("AT-1", FORWARD, "scale 1/sqrt(d_head) -> 1/sqrt(n_embd)",
     "att = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_head)",
     "att = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(C)"),
    ("AT-2", FORWARD, "divide -> multiply by the scale",
     "att = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_head)",
     "att = (q @ np.swapaxes(k, -1, -2)) * np.sqrt(d_head)"),
    ("AT-3", FORWARD, "remove the scale entirely",
     "att = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_head)",
     "att = q @ np.swapaxes(k, -1, -2)"),

    # ---- Attention: mask ----
    ("MK-1", FORWARD, "remove the causal mask",
     "    att = np.where(causal, att, -np.inf)\n    att = softmax(att, axis=-1)",
     "    att = softmax(att, axis=-1)"),
    ("MK-2", FORWARD, "mask AFTER the softmax instead of before",
     "    att = np.where(causal, att, -np.inf)\n    att = softmax(att, axis=-1)",
     "    att = softmax(att, axis=-1)\n    att = np.where(causal, att, 0.0)"),
    ("MK-3", FORWARD, "tril -> triu (inverted mask)",
     "causal = np.tril(np.ones((T, T), dtype=bool))",
     "causal = np.triu(np.ones((T, T), dtype=bool))"),
    ("MK-4", FORWARD, "tril -> tril(k=-1) (hides the current position)",
     "causal = np.tril(np.ones((T, T), dtype=bool))",
     "causal = np.tril(np.ones((T, T), dtype=bool), -1)"),

    # ---- Attention: heads / projections ----
    ("HD-1", FORWARD, "split_heads without transpose (wrong interleaving)",
     "return np.swapaxes(a.reshape(*lead, T, n_head, d_head), -3, -2)",
     "return a.reshape(*lead, n_head, T, d_head)"),
    ("HD-2", FORWARD, "head merge without transpose",
     "y = np.swapaxes(y, -3, -2).reshape(*lead, T, C)",
     "y = y.reshape(*lead, T, C)"),
    ("HD-3", FORWARD, "q and k swapped",
     "att = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_head)",
     "att = (k @ np.swapaxes(q, -1, -2)) / np.sqrt(d_head)"),
    ("HD-4", FORWARD, "drop the c_proj bias",
     'return y @ p[prefix + "c_proj.weight"] + p[prefix + "c_proj.bias"]',
     'return y @ p[prefix + "c_proj.weight"]'),

    # ---- Block: pre-norm / residual ----
    ("BK-1", FORWARD, "drop the attention residual",
     "    x = x + attention(\n"
     '        layer_norm(x, p[pre + "ln_1.weight"], p[pre + "ln_1.bias"]), p, pre + "attn.", n_head\n'
     "    )",
     "    x = attention(\n"
     '        layer_norm(x, p[pre + "ln_1.weight"], p[pre + "ln_1.bias"]), p, pre + "attn.", n_head\n'
     "    )"),
    ("BK-2", FORWARD, "drop the mlp residual",
     '    x = x + mlp(layer_norm(x, p[pre + "ln_2.weight"], p[pre + "ln_2.bias"]), p, pre + "mlp.")',
     '    x = mlp(layer_norm(x, p[pre + "ln_2.weight"], p[pre + "ln_2.bias"]), p, pre + "mlp.")'),
    ("BK-3", FORWARD, "pre-norm -> raw x into the attention sublayer (no norm)",
     "    x = x + attention(\n"
     '        layer_norm(x, p[pre + "ln_1.weight"], p[pre + "ln_1.bias"]), p, pre + "attn.", n_head\n'
     "    )",
     '    x = x + attention(x, p, pre + "attn.", n_head)'),
    ("BK-4", FORWARD, "ln_1 and ln_2 swapped",
     '        layer_norm(x, p[pre + "ln_1.weight"], p[pre + "ln_1.bias"]), p, pre + "attn.", n_head',
     '        layer_norm(x, p[pre + "ln_2.weight"], p[pre + "ln_2.bias"]), p, pre + "attn.", n_head'),

    # ---- GPT: embeddings / head ----
    ("GP-1", FORWARD, "drop the positional embedding",
     'x = params["wte.weight"][idx] + params["wpe.weight"][:T]',
     'x = params["wte.weight"][idx]'),
    ("GP-2", FORWARD, "wpe[:T] -> wpe[-T:] (slice from the end of the table)",
     'x = params["wte.weight"][idx] + params["wpe.weight"][:T]',
     'x = params["wte.weight"][idx] + params["wpe.weight"][-T:]'),
    ("GP-3", FORWARD, "drop the final ln_f",
     '    x = layer_norm(x, params["ln_f.weight"], params["ln_f.bias"])',
     "    pass"),
    ("GP-4", FORWARD, "run only the first block",
     "    for i in range(n_layer):",
     "    for i in range(1):"),
    ("GP-5", FORWARD, "blocks in reverse order",
     "    for i in range(n_layer):",
     "    for i in reversed(range(n_layer)):"),

    # ---- softmax / cross entropy ----
    ("SM-1", FORWARD, "softmax without the max subtraction",
     "z = x - x.max(axis=axis, keepdims=True)",
     "z = x - 0.0"),
    ("CE-1", FORWARD, "cross_entropy uses sum instead of mean",
     "return float(-log_probs[np.arange(len(targets)), targets].mean())",
     "return float(-log_probs[np.arange(len(targets)), targets].sum())"),
    ("CE-2", FORWARD, "cross_entropy without the negative sign",
     "return float(-log_probs[np.arange(len(targets)), targets].mean())",
     "return float(log_probs[np.arange(len(targets)), targets].mean())"),
    ("CE-3", FORWARD, "cross_entropy without the log-sum-exp stabilization",
     "z = logits - logits.max(axis=-1, keepdims=True)",
     "z = logits"),
    ("CE-4", FORWARD, "cross_entropy reads the neighbouring target, (target + 1) % vocab",
     "return float(-log_probs[np.arange(len(targets)), targets].mean())",
     "return float(-log_probs[np.arange(len(targets)), (targets + 1) % logits.shape[-1]].mean())"),

    # ---- loader ----
    ("WT-1", WEIGHTS, "do not transpose the Linear weights",
     "            w = w.T",
     "            w = w"),
    ("WT-2", WEIGHTS, "load lm_head as a separate weight (breaks the tying invariant)",
     "        if name == _TIED_TO_WTE:\n            continue",
     "        if False:\n            continue"),
    ("WT-3", WEIGHTS, "load in float32 instead of float64",
     "w = blob[key].astype(np.float64)",
     "w = blob[key].astype(np.float32)"),
    ("WT-4", WEIGHTS, "transpose wte TOO (over-transpose)",
     "        if name.endswith(_LINEAR_WEIGHTS):\n            w = w.T",
     "        if name.endswith(_LINEAR_WEIGHTS) or name == 'wte.weight':\n            w = w.T"),

    # ---- M1: greedy generation loop ----
    ("GE-1", GENERATE, "remove the context crop entirely",
     "        idx_cond = idx[-block_size:] if idx.shape[0] > block_size else idx",
     "        idx_cond = idx"),
    ("GE-2", GENERATE, "invert the crop condition (crops when it should not, and vice versa)",
     "        idx_cond = idx[-block_size:] if idx.shape[0] > block_size else idx",
     "        idx_cond = idx[-block_size:] if idx.shape[0] <= block_size else idx"),
    ("GE-3", GENERATE, "EOS-stop appends to idx BEFORE checking",
     "        if next_id == eos_id:\n"
     "            return emitted  # stop WITHOUT appending and WITHOUT emitting.\n"
     "\n"
     "        idx = np.append(idx, next_id)\n"
     "        emitted.append(next_id)",
     "        idx = np.append(idx, next_id)\n"
     "        if next_id == eos_id:\n"
     "            return emitted\n"
     "\n"
     "        emitted.append(next_id)"),
    ("GE-4", GENERATE, "EOS-stop emits BEFORE checking",
     "        if next_id == eos_id:\n"
     "            return emitted  # stop WITHOUT appending and WITHOUT emitting.\n"
     "\n"
     "        idx = np.append(idx, next_id)\n"
     "        emitted.append(next_id)",
     "        emitted.append(next_id)\n"
     "        if next_id == eos_id:\n"
     "            return emitted\n"
     "\n"
     "        idx = np.append(idx, next_id)"),
    ("GE-5", GENERATE, "read the FIRST logits position instead of the last",
     "        last_logits = logits[..., -1, :]",
     "        last_logits = logits[..., 0, :]"),
    ("GE-6", GENERATE, "argmax on the wrong axis",
     "        next_id = np.argmax(last_logits, axis=-1).item()",
     "        next_id = np.argmax(last_logits, axis=0).item()"),
    ("GE-7", GENERATE, "crop from the WRONG END (oldest tokens kept instead of newest)",
     "        idx_cond = idx[-block_size:] if idx.shape[0] > block_size else idx",
     "        idx_cond = idx[:block_size] if idx.shape[0] > block_size else idx"),

    # ---- M2: incremental KV-cache ----
    ("KV-1", CACHE, "off-by-one: position = cache length AFTER insert, not before",
     "            logits, cache = forward_step_fn(int(idx[position]), position, cache)",
     "            logits, cache = forward_step_fn(int(idx[position]), position + 1, cache)"),
    ("KV-2", CACHE, "remove the full-recompute fallback (cache grows past block_size)",
     "        if idx.shape[0] > block_size:",
     "        if False:"),
    ("KV-3", CACHE, "fall back one step too early (>= instead of >)",
     "        if idx.shape[0] > block_size:",
     "        if idx.shape[0] >= block_size:"),
    ("KV-4", CACHE, "K concatenated in the wrong order (new before old), V left alone",
     "            K = np.concatenate([k_prev, k], axis=1)  # old first, new last",
     "            K = np.concatenate([k, k_prev], axis=1)  # old first, new last"),
    ("KV-5", CACHE, "incremental causal mask leaks the future -- NO TARGET, see report",
     "causal = np.tril(np.ones((T, T), dtype=bool))",
     "causal = np.triu(np.ones((T, T), dtype=bool))"),
    ("KV-6", CACHE, "the new token cannot attend to itself (its own key masked out)",
     "        att = softmax(att, axis=-1)",
     "        att[..., -1] = -np.inf\n        att = softmax(att, axis=-1)"),
    ("KV-7", CACHE, "K AND V both reversed consistently (predicted equivalent)",
     "            K = np.concatenate([k_prev, k], axis=1)  # old first, new last\n"
     "            V = np.concatenate([v_prev, v], axis=1)",
     "            K = np.concatenate([k, k_prev], axis=1)\n"
     "            V = np.concatenate([v, v_prev], axis=1)"),
    # KV-8/KV-9 -- the two EOS mutations proved dead BY HAND at M2 closure but
    # never added to this reproducible loop. Same expected killer:
    # test_generate_with_cache_stops_on_real_eos_matches_personacore.
    ("KV-8", CACHE, "cached path ignores eos_id entirely (hand-checked: +14 extra items)",
     "        if next_id == eos_id:\n"
     "            return emitted  # stop WITHOUT appending and WITHOUT emitting.",
     "        if False:\n"
     "            return emitted"),
    ("KV-9", CACHE, "EOS token emitted before stopping (hand-checked: 261 items, +1)",
     "        if next_id == eos_id:\n"
     "            return emitted  # stop WITHOUT appending and WITHOUT emitting.",
     "        if next_id == eos_id:\n"
     "            emitted.append(next_id)\n"
     "            return emitted"),

    # ---- M3: batching axes ----
    # There is deliberately NO padding-mask mutation family here. With
    # right-padding the causal mask already blocks every pad key from every real
    # query, so a padding mask would be dead code and mutating it would prove
    # nothing. The derivation and the measurement (perturbation = 0.000e+00) are
    # in tests/test_batched.py. What IS mutable is the axis bookkeeping below.
    ("BT-1", FORWARD, "sequence length read from the FIRST axis, not the last",
     "    T = idx.shape[-1]  # LAST axis — anything before it is a batch axis.",
     "    T = idx.shape[0]"),
    ("BT-2", FORWARD, "head split swaps the wrong pair of axes",
     "        return np.swapaxes(a.reshape(*lead, T, n_head, d_head), -3, -2)",
     "        return np.swapaxes(a.reshape(*lead, T, n_head, d_head), -2, -1)"),
    ("BT-3", FORWARD, "K transposed on the head axis instead of the last two",
     "    att = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_head)",
     "    att = (q @ np.swapaxes(k, -3, -2)) / np.sqrt(d_head)"),

    # PD-6 — batch rows attending to each other. NO TARGET, and that absence is
    # the proof: isolation is not enforced by any line, it is what `@` MEANS on
    # arrays with leading axes. numpy batches the matmul over every axis before
    # the last two, so row b's queries can only ever meet row b's keys; there is
    # no subscript, no index and no reshape that mixes them. Same posture as
    # KV-5 in M2 — the recurring SKIP documents the property, and it breaks the
    # day someone flattens B into T.
    ("PD-6", FORWARD, "batch rows attend to each other -- NO TARGET, see report",
     "    att = einsum_over_batch_and_position(q, k)",
     "    att = einsum_mixing_batch_rows(q, k)"),

    # PD-7/PD-8 — mutate the TEST, not the engine. test_left_padding_would_leak
    # is the only test whose job is to detect a DIFFERENCE between two paddings,
    # so it is the only one that could silently pass by comparing something to
    # itself. Flipping each side in turn proves it discriminates both ways.
    ("PD-7", TESTS_BATCHED, "test 4 measures the LEFT leak with right-padding (left side flipped)",
     '    left_delta = leak("left")',
     '    left_delta = leak("right")'),
    ("PD-8", TESTS_BATCHED, "test 4 measures the RIGHT leak with left-padding (right side flipped)",
     '    right_delta = leak("right")',
     '    right_delta = leak("left")'),

    # ---- M4: temperature / top-k / top-p ----
    ("TP-1", SAMPLING, "temperature MULTIPLIES instead of dividing",
     "    return logits / max(temperature, TEMPERATURE_FLOOR)",
     "    return logits * max(temperature, TEMPERATURE_FLOOR)"),
    ("TP-2", GENERATE, "greedy no longer short-circuits -- it falls through to the draw",
     "        if greedy:\n            next_id = np.argmax(last_logits, axis=-1).item()",
     "        if False:\n            next_id = np.argmax(last_logits, axis=-1).item()"),
    ("TK-1", SAMPLING, "top-k off by one (keeps k+1)",
     "    kth = np.take(np.sort(logits, axis=-1), -k, axis=-1)  # the k-th largest value",
     "    kth = np.take(np.sort(logits, axis=-1), -(k + 1), axis=-1)"),
    ("TK-2", SAMPLING, "caller guard for top_k <= 0 removed",
     "    if top_k is not None and top_k > 0:\n        x = top_k_filter(x, top_k)",
     "    if top_k is not None:\n        x = top_k_filter(x, top_k)"),
    ("NP-1", SAMPLING, "nucleus cutoff >= becomes > (exact landing no longer closes it)",
     "    sorted_mask = cum >= p",
     "    sorted_mask = cum > p"),
    ("NP-2", SAMPLING, "nucleus sorts ASCENDING instead of descending",
     '    order = np.argsort(-logits, axis=-1, kind="stable")',
     '    order = np.argsort(logits, axis=-1, kind="stable")'),
    ("NP-3", SAMPLING, "order inverted -- top-p runs BEFORE top-k",
     "    if top_k is not None and top_k > 0:\n        x = top_k_filter(x, top_k)\n"
     "    if top_p is not None:\n        x = top_p_filter(x, top_p)",
     "    if top_p is not None:\n        x = top_p_filter(x, top_p)\n"
     "    if top_k is not None and top_k > 0:\n        x = top_k_filter(x, top_k)"),

    # ---- M5: int8 quantization ----
    ("QT-1", QUANTIZE, "scale from mean(|w|) instead of max(|w|) -- underscales, clips the tails",
     "        amax = np.abs(weight).max(axis=tuple(range(weight.ndim - 1)), keepdims=True)",
     "        amax = np.abs(weight).mean(axis=tuple(range(weight.ndim - 1)), keepdims=True)"),
    ("QT-2", QUANTIZE, "forget the clip to [-127, 127] (silent int8 overflow)",
     "    q = np.clip(np.rint(weight / scale.astype(np.float64)), -QMAX, QMAX).astype(np.int8)",
     "    q = np.rint(weight / scale.astype(np.float64)).astype(np.int8)"),
    ("QT-3", QUANTIZE, "same scale for every channel -- per-channel is per-tensor in disguise",
     "        amax = np.abs(weight).max(axis=tuple(range(weight.ndim - 1)), keepdims=True)",
     "        amax = np.abs(weight).max(keepdims=True)"),
    ("QT-4", QUANTIZE, "eligibility filter ignored -- LayerNorm and biases get quantized too",
     "    if name.endswith(_LINEAR_WEIGHTS):\n        return True",
     "    if name.endswith(_LINEAR_WEIGHTS):\n        return True\n    if True:\n        return True"),
    ("QT-5", QUANTIZE, "floor instead of round-to-nearest -- biases the error downward",
     "    q = np.clip(np.rint(weight / scale.astype(np.float64)), -QMAX, QMAX).astype(np.int8)",
     "    q = np.clip(np.floor(weight / scale.astype(np.float64)), -QMAX, QMAX).astype(np.int8)"),
    ("QT-6", QUANTIZE, "dequantize with the WRONG channel's scale (rolled by one)",
     "    return q_weight.astype(np.float64) * scale.astype(np.float64)",
     "    return q_weight.astype(np.float64) * np.roll(scale, 1, axis=-1).astype(np.float64)"),

    # ---- M6: MLX port -- mutations DIRECTED at porting errors only (the shared
    # logic is already proven in NumPy; re-mutating it here would be duplicate work) ----
    ("MX-1", FORWARD_MLX, "silent dtype error: weights land as float16 instead of float32",
     "    return {k: mx.array(v.astype(np.float32)) for k, v in np_params.items()}",
     "    return {k: mx.array(v.astype(np.float16)) for k, v in np_params.items()}"),
    ("MX-2", FORWARD_MLX, "MLX axis error: head split swaps the wrong pair of axes",
     "        return mx.swapaxes(a.reshape(*lead, T, n_head, d_head), -3, -2)",
     "        return mx.swapaxes(a.reshape(*lead, T, n_head, d_head), -2, -1)"),
    ("MX-3", FORWARD_MLX, "MLX broadcasting divergence from NumPy -- NO TARGET, see report",
     "# no broadcasting divergence found in the Passo 0.3 probe: add/where/keepdims",
     "# all behave NumPy-identically, so there is no line whose MLX broadcasting"),
]


def run_suite():
    """Run the suite MINUS the slow tests. Returns (passed, names of failures).

    `-m "not slow"` drops the two 280-token parity runs, which are ~90% of the
    suite's wall-clock and get paid for 55 times here (baseline + 54 mutants):
    11:26 becomes about a minute and a half.

    This is a measured exclusion, not a guess. The full run WITH them killed 51 of
    54; they participate in 32 and 35 of those kills respectively and are the SOLE
    killer of NONE, so nothing here loses its last line of defence. They still run
    in the normal `pytest` invocation. If a future mutant's only killer turns out
    to be a slow test, this flag has to go -- re-measure before assuming it holds.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rf", "-m", "not slow"],
        cwd=ROOT, capture_output=True, text=True,
    )
    failed = [
        line.split(" ")[1].split("::")[-1].split("[")[0]
        for line in proc.stdout.splitlines()
        if line.startswith(("FAILED", "ERROR")) and len(line.split(" ")) > 1
    ]
    return proc.returncode == 0, sorted(set(failed))


def main():
    try:
        import mlx  # noqa: F401
    except ImportError:
        print(
            "=" * 78 + "\n"
            "WARNING: mlx is not importable with this interpreter.\n"
            "tests/test_forward_mlx.py will SKIP entirely, so the MX-1/MX-2/MX-3\n"
            "results in this run are NOT reliable -- MX-1 and MX-2 will show as\n"
            "SURVIVED without meaning anything. Use an mlx-capable interpreter:\n"
            "  ~/PersonaCore/.venv/bin/python scripts/mutation_check.py\n"
            + "=" * 78 + "\n"
        )

    baseline_ok, _ = run_suite()
    if not baseline_ok:
        print("BASELINE IS RED — fix the suite before mutating.")
        return 1

    survivors = []
    for mid, path, desc, old, new in MUTATIONS:
        original = path.read_text()
        if old not in original:
            print(f"{mid:6} SKIP     snippet not found in {path.name}: {desc}")
            survivors.append((mid, desc, "snippet not found"))
            continue
        try:
            path.write_text(original.replace(old, new, 1))
            passed, failed = run_suite()
        finally:
            path.write_text(original)

        if passed:
            print(f"{mid:6} SURVIVED {path.name}: {desc}")
            survivors.append((mid, desc, ""))
        else:
            # ALL killers, not the first 3: deciding whether a test can be
            # dropped from the loop requires knowing whether it is some mutant's
            # ONLY killer, and a truncated list cannot answer that.
            print(f"{mid:6} killed   {path.name}: {desc}  <- {', '.join(failed)}")

    killed = len(MUTATIONS) - len(survivors)
    print(f"\n{killed}/{len(MUTATIONS)} killed, {len(survivors)} alive")
    for mid, desc, note in survivors:
        print(f"  SURVIVOR {mid}: {desc} {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
