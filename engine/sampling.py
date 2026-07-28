"""M4: the three logit transforms and the draw, replicating PersonaCore exactly.

Order is locked: temperature -> top-k -> top-p -> softmax -> draw. top-p therefore
operates on the set top-k already narrowed, and on probabilities renormalised over
that set — which is what makes the order observable rather than cosmetic.

There is NO bit-parity with PyTorch here and there cannot be: numpy's Generator and
torch's Generator are different algorithms, so the same seed draws different tokens
by construction. What is replicated is the arithmetic and the edge conventions; what
is proven is structure. See the module docstring of tests/test_sampling.py.

Every function takes last-position logits — 1-D ``(vocab,)`` — and is pure. The rng
is INJECTED, same dependency-injection posture as ``forward_fn`` in M1/M2: tests seed
it, production can leave it out.
"""

import numpy as np

from .forward import softmax

# PersonaCore floors the temperature instead of validating it:
#   return logits / max(temperature, 1e-8)
# so 0.0 does not divide by zero, and a NEGATIVE temperature is silently clamped to
# the same floor rather than inverting the distribution. Replicated, not fixed.
TEMPERATURE_FLOOR = 1e-8


def apply_temperature(logits, temperature):
    """``logits / max(temperature, 1e-8)``.

    A floor, not a guard. temperature=0 sharpens toward the argmax (it does not
    raise, and does not produce inf/nan); temperature<0 lands on the same floor and
    also sharpens — it does NOT flip the ordering. Upstream claims this case is
    routed to the greedy branch by `next_token`; that branch does not exist, so the
    behaviour above is what actually runs.
    """
    return logits / max(temperature, TEMPERATURE_FLOOR)


def top_k_filter(logits, k):
    """Keep the k highest logits, mask the rest to ``-inf``. Masks BEFORE the softmax.

    The cutoff is STRICT (``<``), matching nanoGPT and PersonaCore, so logits TIED
    with the k-th largest all survive and MORE than k tokens can remain. k is not a
    hard cap under ties.

    Raises on ``k <= 0`` rather than silently doing nothing, because that is what the
    original does (torch: IndexError at 0, RuntimeError below it). The "disabled"
    idiom is handled one layer up, in :func:`sample_next_token` — filter unguarded,
    caller guarded. A silent no-op here would make a missing guard undetectable.
    """
    if k <= 0:
        raise ValueError(f"top_k must be positive, got {k} — the caller owns the no-op guard")
    k = min(k, logits.shape[-1])
    kth = np.take(np.sort(logits, axis=-1), -k, axis=-1)  # the k-th largest value
    return np.where(logits < np.expand_dims(kth, -1), -np.inf, logits)


def top_p_filter(logits, p):
    """Nucleus mask keeping the smallest set whose cumulative probability reaches p.

    Convention (a): the token that CROSSES the line is kept, and the top-1 token is
    never masked, so the support is never empty. ``>=`` means a token landing exactly
    on p closes the nucleus instead of pulling in the next one — on
    [0.5, 0.25, 0.125, 0.125] (cumulative [0.5, 0.75, 0.875, 1.0]) that is 2 tokens
    at p=0.75 and 3 at p=0.80. The exactness matters: with non-representable sums the
    landing is never exact and ``>=`` behaves like ``>``.

    Sort descending, cumsum the softmax, mask where the cumulative mass has reached
    p, then shift right by one so the crossing token survives, then scatter the mask
    back to the original token order.
    """
    order = np.argsort(-logits, axis=-1, kind="stable")
    cum = np.cumsum(softmax(np.take_along_axis(logits, order, axis=-1), axis=-1), axis=-1)

    sorted_mask = cum >= p
    # Shift right: a token is masked only once its PREDECESSOR already reached p.
    # `.copy()` because source and destination overlap.
    sorted_mask[..., 1:] = sorted_mask[..., :-1].copy()
    sorted_mask[..., 0] = False  # the most probable token is never masked.

    mask = np.empty_like(sorted_mask)
    np.put_along_axis(mask, order, sorted_mask, axis=-1)
    return np.where(mask, -np.inf, logits)


def sample_next_token(logits, *, temperature=1.0, top_k=None, top_p=None, rng=None):
    """Draw one token id from ``(vocab,)`` last-position logits. Returns an ``int``.

    The locked order runs here, and this is where the ``top_k <= 0`` no-op lives —
    mirroring PersonaCore's `next_token`, which wraps the filter in
    ``if top_k is not None and top_k > 0``. ``top_p`` has no such guard upstream and
    gets none here.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = apply_temperature(np.asarray(logits, dtype=np.float64), temperature)
    if top_k is not None and top_k > 0:
        x = top_k_filter(x, top_k)
    if top_p is not None:
        x = top_p_filter(x, top_p)

    probs = softmax(x, axis=-1)
    return int(rng.choice(probs.shape[-1], p=probs))
