"""Slim-loader tests that always run: a tiny synthetic torch-zip, no 55 MB download.

CI and a laptop clone both execute this file. The live m1-demo-v1 proof lives in
``tests/test_slim_public.py``.
"""

from collections import OrderedDict

import numpy as np
import pytest

from engine.slim import count_params, load_slim
from tests.slim_pt import tiny_slim_dict, write_slim_pt


def _write(tmp_path, slim=None, **kwargs):
    blob = slim if slim is not None else tiny_slim_dict()
    q = blob.pop("q_proj_torch", None)
    path = tmp_path / "synthetic_slim.pt"
    write_slim_pt(path, blob, **kwargs)
    return path, blob, q


def test_synthetic_slim_load_succeeds(tmp_path):
    path, blob, _ = _write(tmp_path)
    params, meta = load_slim(path)
    assert meta["schema_version"] == 1
    assert meta["git_sha"] == "synthetic"
    assert meta["step"] == 0
    assert meta["val_loss"] is None
    assert "wte.weight" in params
    assert "lm_head.weight" not in params


def test_synthetic_slim_config_matches_payload(tmp_path):
    path, blob, _ = _write(tmp_path)
    _params, meta = load_slim(path)
    cfg = meta["model_config"]
    assert cfg["n_layer"] == 1
    assert cfg["n_head"] == 1
    assert cfg["n_embd"] == 4
    assert cfg["block_size"] == 4
    assert cfg["vocab_size"] == 8


def test_synthetic_linear_weight_is_transposed_by_value(tmp_path):
    """q_proj is square: shape cannot catch a missed transpose; values can."""
    path, _blob, q = _write(tmp_path)
    params, _meta = load_slim(path)
    got = params["blocks.0.attn.q_proj.weight"]
    assert got.shape == q.shape
    assert np.array_equal(got, q.T.astype(np.float64))
    assert not np.array_equal(got, q.astype(np.float64))


def test_synthetic_embeddings_are_not_transposed(tmp_path):
    path, blob, _ = _write(tmp_path)
    params, _meta = load_slim(path)
    wte = blob["model"]["wte.weight"]
    assert np.array_equal(params["wte.weight"], wte.astype(np.float64))


def test_synthetic_param_count_drops_tied_lm_head(tmp_path):
    path, blob, _ = _write(tmp_path)
    params, _meta = load_slim(path)
    expected = sum(np.asarray(v).size for k, v in blob["model"].items() if k != "lm_head.weight")
    assert count_params(params) == expected
    assert "lm_head.weight" not in params


def test_synthetic_params_are_float64(tmp_path):
    path, _blob, _ = _write(tmp_path)
    params, _meta = load_slim(path)
    assert all(v.dtype == np.float64 for v in params.values())


def test_unsupported_schema_version_raises(tmp_path):
    blob = tiny_slim_dict()
    blob.pop("q_proj_torch")
    blob["schema_version"] = 99
    path = tmp_path / "bad_schema.pt"
    write_slim_pt(path, blob)
    with pytest.raises(ValueError, match="schema_version"):
        load_slim(path)


def test_missing_slim_keys_raise(tmp_path):
    blob = tiny_slim_dict()
    blob.pop("q_proj_torch")
    blob = {k: v for k, v in blob.items() if k != "git_sha"}
    # Reconstruct a payload missing git_sha via a broken zip by writing then... 
    # write_slim_pt requires the key. Build a dict and patch after dump is harder.
    # Write a complete file then we test the validator with a second helper path:
    # load_slim on a zip whose pickle lacks git_sha.
    from tests.slim_pt import _SlimPickler, _FloatStorage, _Storage, _Tensor, _rebuild_tensor_v2  # noqa: F401
    import io
    import pickle
    import zipfile

    model = OrderedDict()
    wte = np.zeros((2, 2), dtype=np.float32)
    storage = _Storage("0", wte.reshape(-1))
    model["wte.weight"] = _Tensor(storage, wte.shape)
    payload = {
        "schema_version": 1,
        "model": model,
        "model_config": {"n_layer": 1, "n_head": 1, "n_embd": 2, "block_size": 2},
        "step": 0,
        "val_loss": None,
    }
    buf = io.BytesIO()
    _SlimPickler(buf, protocol=2).dump(payload)
    path = tmp_path / "missing_keys.pt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("missing_keys/data.pkl", buf.getvalue())
        z.writestr("missing_keys/byteorder", b"little")
        z.writestr("missing_keys/data/0", storage.flat.tobytes())
    with pytest.raises(ValueError, match="missing"):
        load_slim(path)
