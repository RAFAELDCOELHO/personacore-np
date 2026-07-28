"""Load the frozen weights from the npz into this engine's convention.

Two transformations, and only two:

1. **fp32 -> fp64.** The npz stores fp32 (the checkpoint is fp32). The forward
   runs in fp64 to match `ref_logits`, which PyTorch produced with the model
   promoted to double. The cast is lossless.

2. **(out, in) -> (in, out)** on Linear weights. The npz comes in the
   `torch.nn.Linear` convention, which stores W as (out, in) and computes
   `x @ W.T`. This engine stores (in, out) and computes `x @ W`, so the
   transpose happens ONCE, here, and the forward stays free of `.T`. Embeddings,
   biases and LayerNorm parameters are NOT transposed.

`lm_head.weight` is deliberately discarded: it is the same tensor as
`wte.weight` (weight tying — PyTorch serializes the tied parameter under both
names). Logits come out of `wte.weight`, and loading the second copy would mask
a future break of the tying.
"""

import numpy as np

# Suffixes of the weights that need transposing: every `.weight` of an
# nn.Linear. Biases, embeddings and LayerNorm are excluded — they are 1-D or
# already in the right orientation.
_LINEAR_WEIGHTS = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.c_proj.weight",
    "mlp.fc_in.weight",
    "mlp.fc_out.weight",
)

_TIED_TO_WTE = "lm_head.weight"


def load_weights(npz_path):
    """npz -> dict[str, np.ndarray] in float64, with the ``w::`` prefix stripped."""
    blob = np.load(npz_path)
    params = {}
    for key in blob.files:
        if not key.startswith("w::"):
            continue
        name = key[3:]
        if name == _TIED_TO_WTE:
            continue
        w = blob[key].astype(np.float64)
        if name.endswith(_LINEAR_WEIGHTS):
            w = w.T
        params[name] = w
    return params
