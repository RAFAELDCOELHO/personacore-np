"""Test-only writer for a tiny torch-zip slim checkpoint (no torch install).

Produces the same zip layout PersonaCore's ``export_slim`` writes:
``{stem}/data.pkl``, ``byteorder``, ``version``, ``data/{key}`` storages, with
pickle REDUCE to ``torch._utils._rebuild_tensor_v2`` and persistent storage
ids. Production code only *reads* this format; this module is not imported by
the engine.
"""

from __future__ import annotations

import io
import pickle
import sys
import types
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np


def _rebuild_tensor_v2(*_a, **_k):
    raise RuntimeError("dump-only REDUCE target; the loader reconstructs tensors")


_rebuild_tensor_v2.__module__ = "torch._utils"
_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"


class _FloatStorage:
    pass


_FloatStorage.__module__ = "torch"
_FloatStorage.__qualname__ = "FloatStorage"

# Pickle records GLOBAL torch._utils._rebuild_tensor_v2 / torch.FloatStorage.
# Register stub modules so dump works without a torch install; the loader never
# imports these — it maps the names itself.
_torch = types.ModuleType("torch")
_torch_utils = types.ModuleType("torch._utils")
_torch.FloatStorage = _FloatStorage
_torch_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch._utils", _torch_utils)
if not hasattr(sys.modules["torch"], "FloatStorage"):
    sys.modules["torch"].FloatStorage = _FloatStorage
if not hasattr(sys.modules["torch._utils"], "_rebuild_tensor_v2"):
    sys.modules["torch._utils"]._rebuild_tensor_v2 = _rebuild_tensor_v2


class _Storage:
    def __init__(self, key: str, flat: np.ndarray):
        self.key = key
        self.flat = np.ascontiguousarray(flat, dtype=np.float32)


class _Tensor:
    def __init__(self, storage: _Storage, shape: tuple[int, ...]):
        self.storage = storage
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) == 0:
            self.stride = ()
        elif len(self.shape) == 1:
            self.stride = (1,)
        else:
            # row-major contiguous, matching torch's default for these weights
            self.stride = (self.shape[1], 1)

    def __reduce__(self):
        return (
            _rebuild_tensor_v2,
            (self.storage, 0, self.shape, self.stride, False, OrderedDict()),
        )


class _SlimPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _Storage):
            return ("storage", _FloatStorage, obj.key, "cpu", int(obj.flat.size))
        return None


def write_slim_pt(path, slim: dict, *, stem: str | None = None) -> Path:
    """Write ``slim`` (schema dict whose ``model`` values are numpy arrays) as a .pt zip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = stem or path.stem
    model = slim["model"]
    storages: list[_Storage] = []
    pickled_model = OrderedDict()
    for name, array in model.items():
        arr = np.ascontiguousarray(array, dtype=np.float32)
        storage = _Storage(str(len(storages)), arr.reshape(-1))
        storages.append(storage)
        pickled_model[name] = _Tensor(storage, arr.shape)

    payload = {
        "schema_version": slim["schema_version"],
        "model": pickled_model,
        "model_config": dict(slim["model_config"]),
        "git_sha": slim["git_sha"],
        "step": slim["step"],
        "val_loss": slim["val_loss"],
    }
    buf = io.BytesIO()
    _SlimPickler(buf, protocol=2).dump(payload)
    pkl = buf.getvalue()

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        prefix = f"{stem}/"
        z.writestr(prefix + "data.pkl", pkl)
        z.writestr(prefix + "byteorder", b"little")
        z.writestr(prefix + "version", b"3\n")
        z.writestr(prefix + ".format_version", b"1")
        z.writestr(prefix + ".storage_alignment", b"64")
        for s in storages:
            z.writestr(prefix + f"data/{s.key}", s.flat.tobytes())
    return path


def tiny_slim_dict(*, seed: int = 0) -> dict:
    """Minimal slim-shaped GPT: 1 layer / 1 head / 4 / 4, vocab 8. Known q_proj."""
    rng = np.random.default_rng(seed)
    n_embd, n_hidden, vocab, block = 4, 16, 8, 4

    def rand(*shape):
        return rng.standard_normal(shape, dtype=np.float32)

    # Non-symmetric so a missed transpose is visible by value, not only by shape.
    q = np.arange(n_embd * n_embd, dtype=np.float32).reshape(n_embd, n_embd)

    model = OrderedDict()
    model["wte.weight"] = rand(vocab, n_embd)
    model["wpe.weight"] = rand(block, n_embd)
    model["blocks.0.ln_1.weight"] = np.ones(n_embd, dtype=np.float32)
    model["blocks.0.ln_1.bias"] = np.zeros(n_embd, dtype=np.float32)
    model["blocks.0.attn.q_proj.weight"] = q  # (out, in) torch layout
    model["blocks.0.attn.q_proj.bias"] = rand(n_embd)
    model["blocks.0.attn.k_proj.weight"] = rand(n_embd, n_embd)
    model["blocks.0.attn.k_proj.bias"] = rand(n_embd)
    model["blocks.0.attn.v_proj.weight"] = rand(n_embd, n_embd)
    model["blocks.0.attn.v_proj.bias"] = rand(n_embd)
    model["blocks.0.attn.c_proj.weight"] = rand(n_embd, n_embd)
    model["blocks.0.attn.c_proj.bias"] = rand(n_embd)
    model["blocks.0.ln_2.weight"] = np.ones(n_embd, dtype=np.float32)
    model["blocks.0.ln_2.bias"] = np.zeros(n_embd, dtype=np.float32)
    model["blocks.0.mlp.fc_in.weight"] = rand(n_hidden, n_embd)
    model["blocks.0.mlp.fc_in.bias"] = rand(n_hidden)
    model["blocks.0.mlp.fc_out.weight"] = rand(n_embd, n_hidden)
    model["blocks.0.mlp.fc_out.bias"] = rand(n_embd)
    model["ln_f.weight"] = np.ones(n_embd, dtype=np.float32)
    model["ln_f.bias"] = np.zeros(n_embd, dtype=np.float32)
    model["lm_head.weight"] = model["wte.weight"]  # tied: same array object

    cfg = {
        "vocab_size": vocab,
        "eos_id": vocab - 1,
        "block_size": block,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": n_embd,
        "dropout": 0.0,
    }
    return {
        "schema_version": 1,
        "model": model,
        "model_config": cfg,
        "git_sha": "synthetic",
        "step": 0,
        "val_loss": None,
        "q_proj_torch": q,
    }
