"""Build public parity fixtures from PersonaCore release m1-demo-v1.

No private companion repo. The oracle is PersonaCore's own GPT (cloned at the
slim file's ``git_sha``) run in float64, greedy. The input window is a public
deterministic id sequence, not private ``val.bin``.

Writes (gitignored, regenerated):

    fixtures/personacore_parity.npz
    fixtures/personacore_generate_fixture.npz
    fixtures/personacore_eos_fixture.npz
    fixtures/val_windows.npz

Usage:  python scripts/gen_public_fixtures.py
        make infer
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
ORACLE_DIR = ROOT / ".oracle" / "PersonaCore"
ORACLE_URL = "https://github.com/RAFAELDCOELHO/PersonaCore.git"
STAMP = FIXTURES / ".public_source"

PARITY = FIXTURES / "personacore_parity.npz"
GEN = FIXTURES / "personacore_generate_fixture.npz"
EOS = FIXTURES / "personacore_eos_fixture.npz"
VAL = FIXTURES / "val_windows.npz"

SHORT_NEW = 20
LONG_NEW = 280
PROMPT_LEN = 10
EOS_MAX_NEW = 30
EOS_STEP = 15  # 0-based index of the token used as artificial EOS

# Public window: in-vocab, 257 consecutive ids, 256 unique in the LM pair.
# NOT private val.bin. Tests that only need uniqueness / shift structure still hold.
VOCAB = 8192


def public_window(n: int = 257) -> np.ndarray:
    return ((np.arange(n, dtype=np.int64) * 17 + 3) % VOCAB)


def needed_paths() -> tuple[Path, ...]:
    return (PARITY, GEN, EOS, VAL)


def fixtures_ready() -> bool:
    return all(p.is_file() and p.stat().st_size > 0 for p in needed_paths())


def _stamp_text(url: str, git_sha: str, step: int) -> str:
    return f"url={url}\ngit_sha={git_sha}\nstep={step}\nwindow=public_arange_257\n"


def ensure_oracle() -> Path:
    """Clone public PersonaCore (default branch). The slim ``git_sha`` is weight
    provenance from training time and may predate ``load_slim`` / ``generate``.
    """
    src = ORACLE_DIR / "src"
    if (src / "personacore" / "model" / "gpt.py").is_file() and (
        src / "personacore" / "generation" / "core.py"
    ).is_file():
        return src
    ORACLE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if ORACLE_DIR.exists():
        subprocess.check_call(["rm", "-rf", str(ORACLE_DIR)])
    subprocess.check_call(["git", "clone", "--depth", "1", ORACLE_URL, str(ORACLE_DIR)])
    return src


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "gen_public_fixtures.py needs PyTorch to run the public PersonaCore "
            "oracle (weights_only slim load + float64 forward). Install a CPU wheel "
            "and re-run `make infer`:\n"
            "  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc


def generate_fixtures() -> None:
    _require_torch()
    import torch

    sys.path.insert(0, str(ROOT))
    from engine.slim import SLIM_URL, ensure_slim_checkpoint

    slim_path = ensure_slim_checkpoint()
    src = ensure_oracle()
    sys.path.insert(0, str(src))

    from personacore.checkpoint import load_slim
    from personacore.config import ModelConfig
    from personacore.generation.core import generate as pc_generate
    from personacore.model.gpt import GPT

    slim = load_slim(str(slim_path), map_location="cpu")
    git_sha = str(slim["git_sha"])
    allowed = {f.name for f in fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in slim["model_config"].items() if k in allowed})
    model = GPT(cfg, attn_impl="manual")
    model.load_state_dict(slim["model"])
    model.eval()
    model.double()

    window = public_window(257)
    input_x = window[:-1]
    input_y = window[1:]
    idx = torch.from_numpy(np.ascontiguousarray(input_x)).view(1, -1)
    targets = torch.from_numpy(np.ascontiguousarray(input_y)).view(1, -1)
    with torch.no_grad():
        logits, loss = model(idx, targets)
    ref_logits = logits[0].detach().cpu().numpy().astype(np.float64)
    ref_loss = float(loss.detach().cpu())

    payload = {}
    for name, tensor in slim["model"].items():
        payload[f"w::{name}"] = tensor.detach().cpu().contiguous().float().numpy()
    payload["input_x"] = input_x
    payload["input_y"] = input_y
    payload["ref_logits"] = ref_logits
    payload["ref_loss"] = np.float64(ref_loss)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez(PARITY, **payload)

    prompt = torch.from_numpy(input_x[:PROMPT_LEN]).view(1, -1)
    with torch.no_grad():
        short_ids = list(pc_generate(model, prompt, max_new_tokens=SHORT_NEW, greedy=True))
        long_ids = list(pc_generate(model, prompt, max_new_tokens=LONG_NEW, greedy=True))
        eos_run = list(pc_generate(model, prompt, max_new_tokens=EOS_MAX_NEW, greedy=True))

    if len(eos_run) <= EOS_STEP:
        raise RuntimeError(
            f"greedy run produced {len(eos_run)} tokens; need > {EOS_STEP} for the EOS fixture"
        )
    artificial_eos = int(eos_run[EOS_STEP])
    eos_ids = eos_run[:EOS_STEP]

    np.savez(
        GEN,
        prompt_ids=input_x[:PROMPT_LEN],
        short_generated_ids=np.asarray(short_ids, dtype=np.int64),
        short_max_new_tokens=np.int64(SHORT_NEW),
        long_generated_ids=np.asarray(long_ids, dtype=np.int64),
        long_max_new_tokens=np.int64(LONG_NEW),
        block_size=np.int64(cfg.block_size),
        eos_id=np.int64(cfg.eos_id),
    )
    np.savez(
        EOS,
        prompt_ids=input_x[:PROMPT_LEN],
        generated_ids=np.asarray(eos_ids, dtype=np.int64),
        artificial_eos_id=np.int64(artificial_eos),
        max_new_tokens=np.int64(EOS_MAX_NEW),
        block_size=np.int64(cfg.block_size),
        eos_step=np.int64(EOS_STEP),
    )
    np.savez(VAL, windows=window.reshape(1, -1), starts=np.array([0], dtype=np.int64))
    STAMP.write_text(_stamp_text(SLIM_URL, git_sha, int(slim["step"])))


def ensure_public_fixtures() -> None:
    """Generate fixtures if missing. Idempotent."""
    if fixtures_ready():
        return
    generate_fixtures()
    if not fixtures_ready():
        raise RuntimeError("public fixture generation finished but files are still missing")


def main() -> int:
    ensure_public_fixtures()
    print("public fixtures ready:")
    for p in needed_paths():
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size} bytes")
    if STAMP.is_file():
        print(STAMP.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
