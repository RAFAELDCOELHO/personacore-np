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
     "att = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_head)",
     "att = (q @ k.transpose(0, 2, 1)) / np.sqrt(C)"),
    ("AT-2", FORWARD, "divide -> multiply by the scale",
     "att = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_head)",
     "att = (q @ k.transpose(0, 2, 1)) * np.sqrt(d_head)"),
    ("AT-3", FORWARD, "remove the scale entirely",
     "att = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_head)",
     "att = q @ k.transpose(0, 2, 1)"),

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
     "return a.reshape(T, n_head, d_head).transpose(1, 0, 2)",
     "return a.reshape(n_head, T, d_head)"),
    ("HD-2", FORWARD, "head merge without transpose",
     "y = y.transpose(1, 0, 2).reshape(T, C)",
     "y = y.reshape(T, C)"),
    ("HD-3", FORWARD, "q and k swapped",
     "att = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_head)",
     "att = (k @ q.transpose(0, 2, 1)) / np.sqrt(d_head)"),
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
