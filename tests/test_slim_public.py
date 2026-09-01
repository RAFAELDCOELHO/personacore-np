"""Load-path checks for the public m1-demo-v1 slim checkpoint.

Parity against PersonaCore's float64 GPT lives in the existing suite once
``make infer`` has built ``fixtures/*.npz`` from this file. This module only
asserts the slim loader itself.
"""

import numpy as np
import pytest

from engine.forward import gpt_forward
from engine.generate import generate
from engine.slim import (
    SLIM_PARAM_COUNT,
    count_params,
    ensure_slim_checkpoint,
    load_slim,
)

pytestmark = pytest.mark.public_slim


@pytest.fixture(scope="module")
def slim_path():
    return ensure_slim_checkpoint()


@pytest.fixture(scope="module")
def loaded(slim_path):
    return load_slim(slim_path)


def test_m1_demo_v1_load_succeeds(loaded):
    params, meta = loaded
    assert meta["schema_version"] == 1
    assert isinstance(meta["git_sha"], str) and meta["git_sha"]
    assert isinstance(meta["step"], int)
    assert "wte.weight" in params
    assert "lm_head.weight" not in params


def test_m1_demo_v1_param_count_tied_embedding_once(loaded):
    params, _meta = loaded
    assert count_params(params) == SLIM_PARAM_COUNT == 13_891_584


def test_m1_demo_v1_config_matches_slim_modelconfig(loaded):
    _params, meta = loaded
    cfg = meta["model_config"]
    assert cfg["n_layer"] == 6
    assert cfg["n_head"] == 6
    assert cfg["n_embd"] == 384
    assert cfg["block_size"] == 256
    assert cfg["vocab_size"] == 8192


def test_m1_demo_v1_greedy_generate_is_deterministic(loaded):
    params, meta = loaded
    cfg = meta["model_config"]
    prompt = np.array([1, 2, 3, 4], dtype=np.int64)

    def fwd(window):
        return gpt_forward(window, params, n_head=cfg["n_head"])

    kwargs = dict(
        forward_fn=fwd,
        idx=prompt,
        max_new_tokens=4,
        eos_id=int(cfg.get("eos_id", 8184)),
        block_size=cfg["block_size"],
        greedy=True,
    )
    a = generate(**kwargs)
    b = generate(**kwargs)
    assert a == b
    assert 1 <= len(a) <= 4
    vocab = int(cfg["vocab_size"])
    assert all(0 <= t < vocab for t in a)
