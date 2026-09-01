"""Load PersonaCore slim inference checkpoints (``.pt``) into this engine.

The npz oracle path stays in ``engine.weights.load_weights``. This module reads
the public slim schema written by ``personacore.checkpoint.export_slim``:

    {schema_version, model, model_config, git_sha, step, val_loss}

PersonaCore loads that file with ``torch.load(..., weights_only=True)``. This
engine does the same restriction without importing PyTorch: a zip + pickle
unpickler that reconstructs tensors as NumPy arrays and refuses any other
GLOBAL. Linear weights are then transposed into this engine's (in, out)
convention, and the tied ``lm_head.weight`` is dropped, matching ``load_weights``.

The live public artifact is PersonaCore release ``m1-demo-v1``
(``checkpoints/model_slim.pt``, ~55.6 MB). It is downloaded only if missing and
is never committed.
"""

from __future__ import annotations

import io
import os
import pickle
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .weights import _LINEAR_WEIGHTS, _TIED_TO_WTE

SLIM_SCHEMA_VERSION = 1
SLIM_PARAM_COUNT = 13_891_584  # tied embedding counted once
SLIM_URL = (
    "https://github.com/RAFAELDCOELHO/PersonaCore/releases/download/"
    "m1-demo-v1/model_slim.pt"
)
SLIM_REQUIRED_KEYS = frozenset(
    {"schema_version", "model", "model_config", "git_sha", "step", "val_loss"}
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SLIM_PATH = _REPO_ROOT / "checkpoints" / "model_slim.pt"

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


def count_params(params) -> int:
    """Sum of array sizes. Tied ``lm_head`` must already have been dropped."""
    return int(sum(np.asarray(v).size for v in params.values()))


def _want_live_download() -> bool:
    flag = os.environ.get("PERSONACORE_DOWNLOAD_SLIM", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    return True


def download_slim(path=None, *, url: str = SLIM_URL) -> Path:
    """Fetch ``model_slim.pt`` to ``path`` (atomic replace)."""
    path = Path(path) if path is not None else DEFAULT_SLIM_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "personacore-np"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(path)
    return path


def ensure_slim_checkpoint(path=None) -> Path:
    """Return the slim checkpoint path, downloading if missing.

    The public ``m1-demo-v1`` file is ~55.6 MB and is never committed. Set
    ``PERSONACORE_DOWNLOAD_SLIM=0`` only for an air-gapped run that already
    placed the file by hand.
    """
    path = Path(path) if path is not None else DEFAULT_SLIM_PATH
    if path.is_file() and path.stat().st_size > 0:
        return path
    if not _want_live_download():
        raise FileNotFoundError(
            f"{path} is missing and PERSONACORE_DOWNLOAD_SLIM=0. "
            "Place model_slim.pt by hand or unset the flag."
        )
    return download_slim(path)


def load_slim(path) -> tuple[dict, dict]:
    """Load a slim ``.pt`` into engine params + metadata.

    Returns ``(params, meta)`` where ``params`` is the same convention as
    ``load_weights`` (float64, Linear weights transposed, ``lm_head`` dropped)
    and ``meta`` carries ``schema_version``, ``model_config``, ``git_sha``,
    ``step``, ``val_loss``.
    """
    path = Path(path)
    raw = _load_torch_zip(path)
    missing = SLIM_REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(
            f"malformed slim checkpoint {path}: missing keys {sorted(missing)} "
            "(expected schema_version, model, model_config, git_sha, step, val_loss)."
        )
    version = raw.get("schema_version")
    if version != SLIM_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported slim checkpoint schema_version {version!r} in {path} "
            f"(expected {SLIM_SCHEMA_VERSION})."
        )
    params = _params_from_state_dict(raw["model"])
    meta = {
        "schema_version": raw["schema_version"],
        "model_config": dict(raw["model_config"]),
        "git_sha": raw["git_sha"],
        "step": raw["step"],
        "val_loss": raw["val_loss"],
    }
    return params, meta


def _params_from_state_dict(model) -> dict:
    params = {}
    for name, tensor in model.items():
        if name == _TIED_TO_WTE:
            continue
        w = np.asarray(tensor, dtype=np.float64)
        if name.endswith(_LINEAR_WEIGHTS):
            w = np.ascontiguousarray(w.T)
        params[name] = w
    return params


def _rebuild_tensor_v2(
    storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None
):
    """NumPy stand-in for ``torch._utils._rebuild_tensor_v2``."""
    storage = np.ascontiguousarray(storage)
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) for s in stride)
    offset = int(storage_offset)
    if not size:
        return storage[offset].copy()
    itemsize = storage.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(
        storage[offset:],
        shape=size,
        strides=tuple(s * itemsize for s in stride),
        writeable=False,
    )
    return np.ascontiguousarray(view)


def _storage_dtype(storage_type) -> np.dtype:
    name = storage_type if isinstance(storage_type, str) else getattr(
        storage_type, "__name__", type(storage_type).__name__
    )
    try:
        return np.dtype(_STORAGE_DTYPES[name])
    except KeyError as exc:
        raise pickle.UnpicklingError(f"unsupported torch storage type {name!r}") from exc


def _archive_prefix(names) -> str:
    for name in names:
        if name.endswith("data.pkl") and not name.endswith("/"):
            return name[: -len("data.pkl")]
    raise ValueError("not a PyTorch zip archive: no data.pkl")


class _WeightsOnlyUnpickler(pickle.Unpickler):
    """Restricted unpickler: tensors + primitive containers, no code execution."""

    def __init__(self, file, *, storages: dict, zipf: zipfile.ZipFile, prefix: str, byteorder: str):
        super().__init__(file)
        self._storages = storages
        self._zipf = zipf
        self._prefix = prefix
        self._byteorder = byteorder

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {})
        raise pickle.UnpicklingError(
            f"weights_only: refused to import {module}.{name} from slim checkpoint"
        )

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or not pid or pid[0] != "storage":
            raise pickle.UnpicklingError(f"weights_only: bad persistent id {pid!r}")
        # ('storage', storage_type, key, location, numel)
        storage_type, key, _loc, numel = pid[1], pid[2], pid[3], pid[4]
        if key not in self._storages:
            raw = self._zipf.read(f"{self._prefix}data/{key}")
            dtype = _storage_dtype(storage_type)
            arr = np.frombuffer(raw, dtype=dtype, count=int(numel))
            if self._byteorder == "big":
                arr = arr.byteswap()
            self._storages[key] = np.ascontiguousarray(arr)
        return self._storages[key]


def _load_torch_zip(path: Path) -> dict:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a PyTorch zip archive")
    with zipfile.ZipFile(path) as z:
        prefix = _archive_prefix(z.namelist())
        try:
            byteorder = z.read(prefix + "byteorder").decode("ascii").strip()
        except KeyError:
            byteorder = "little"
        unpickler = _WeightsOnlyUnpickler(
            io.BytesIO(z.read(prefix + "data.pkl")),
            storages={},
            zipf=z,
            prefix=prefix,
            byteorder=byteorder,
        )
        loaded = unpickler.load()
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not unpickle to a dict (got {type(loaded)!r})")
    return loaded
