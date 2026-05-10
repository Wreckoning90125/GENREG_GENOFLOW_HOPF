"""Policies for the multi-stage Hopf pre-registered test.

Implements the four contracted architectures:

    HopfPolicy   — K=2 multi-stage Hopf controller per pinned spec
    MLPPolicy    — 6 -> 8 -> 3 tanh net (matched parameter count)
    RandomPolicy — uniform random action (no parameters; floor)

Both Hopf and MLP policies are flat-vector parameterized so the same
ES and SGD/REINFORCE trainers can drive them. The flat parameter
count is exposed via `Policy.n_params`.

Design choices and rationale are pinned in
`experiments/mnist_geometric/PRE_REGISTRATION.md`. This file is the
implementation; if it deviates from the pre-registration, the
pre-registration is the source of truth.
"""

from __future__ import annotations

import math
import numpy as np

from ade_geometry import get_ade
from cell600 import get_geometry
from hopf_controller import (
    qmul,
    qnormalize,
    hopf_project,
    hopf_section,
    poincare_warp,
)


# ---------------------------------------------------------------------------
# Acrobot state -> S^3 quaternion -> 600-cell vertex activations (vMF)
# ---------------------------------------------------------------------------

_ADE_CACHE = None


def _get_ade_cached():
    global _ADE_CACHE
    if _ADE_CACHE is None:
        _ADE_CACHE = get_ade()
    return _ADE_CACHE


def acrobot_obs_to_quat(obs: np.ndarray) -> np.ndarray:
    """Embed (cos t1, sin t1, cos t2, sin t2, dt1, dt2) onto S^3.

    Uses the 2-angle-to-S^3 map specified in the pinning amendment:
    q = (cos(t1/2)*cos(t2/2), sin(t1/2)*cos(t2/2),
         cos(t1/2)*sin(t2/2), sin(t1/2)*sin(t2/2)) and re-normalize.

    Recovers t1/2, t2/2 from the half-angle identity:
      cos(t/2) = sgn * sqrt((1+cos t)/2),  sin(t/2) = sin t / (2 cos(t/2))
    """
    c1, s1, c2, s2 = obs[0], obs[1], obs[2], obs[3]
    # Half-angle from full-angle (sign of cos(t/2) chosen positive)
    ch1 = math.sqrt(max(0.0, (1.0 + c1) / 2.0))
    sh1 = s1 / (2.0 * ch1) if ch1 > 1e-6 else (1.0 if s1 >= 0 else -1.0)
    ch2 = math.sqrt(max(0.0, (1.0 + c2) / 2.0))
    sh2 = s2 / (2.0 * ch2) if ch2 > 1e-6 else (1.0 if s2 >= 0 else -1.0)
    q = np.array([ch1 * ch2, sh1 * ch2, ch1 * sh2, sh1 * sh2], dtype=np.float64)
    return qnormalize(q)


def vmf_activation(q: np.ndarray, kappa: float = 8.0) -> np.ndarray:
    """vMF-soft-assign q onto the 120 vertices of the 600-cell.

    Returns shape (120,), positive, sums to 1. Uses signed dot product
    (NOT abs) to preserve the spinor double cover, consistent with the
    chirality fix.
    """
    geo = get_geometry()
    verts = geo["vertices"]
    raw = verts @ q  # (120,)
    scaled = kappa * raw
    scaled -= scaled.max()
    e = np.exp(scaled)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Multi-stage Hopf policy (K=2, dim-4 eigenspaces, shared rotors per copy)
# ---------------------------------------------------------------------------


class HopfPolicy:
    """K=2 multi-stage Hopf controller for Acrobot.

    Per the pinning amendment:
      - dim-4 eigenspaces only (E3, E7) — each has multiplicity 4
      - one S^3 rotor (3 d.o.f.) per eigenspace per stage, shared
        across all 4 copies of the irrep
      - K=2 stages of (rotor, Hopf-project, section-lift, Poincare warp)
      - readout: linear 24 -> 3 logits, no bias (72 weights)
      - velocity-modulated vMF concentration via the angular-velocity
        components of the obs

    Total learnable d.o.f.: 12 rotor + 72 readout = 84.
    """

    EIG_INDICES = (3, 7)  # E3 and E7, the two dim-4 eigenspaces
    KAPPA_BASE = 8.0
    KAPPA_VEL = 0.5  # added to base kappa per unit angular velocity (capped)
    READOUT_OUT = 3   # number of Acrobot actions

    def __init__(self, K: int = 2) -> None:
        self.K = int(K)
        ade = _get_ade_cached()
        self.eigspaces = [ade["ade_eigenspaces"][i] for i in self.EIG_INDICES]
        # Each V is (120, 16); 16 = 4 copies x 4 irrep coords.
        # We need a per-copy basis: each copy is V @ cb where
        # cb has shape (16, 4). Compose into a single (120, 4, 4) tensor:
        # E_basis[i, copy, irrep_coord] for vertex i.
        self.E_bases = []  # list of (120, copies=4, d=4)
        for es in self.eigspaces:
            V = es["V"]
            copies = es["copies"]  # list of (16, 4) basis matrices
            assert len(copies) == 4 and copies[0].shape[1] == 4
            stack = np.stack([V @ cb for cb in copies], axis=1)  # (120, 4, 4)
            self.E_bases.append(stack)
        # Packed (120, 8, 4) for the unrolled fast forward pass: copies of
        # both eigenspaces concatenated along the copy axis.
        self._E_packed = np.concatenate(self.E_bases, axis=1)

        self.n_eigspaces = len(self.E_bases)  # 2
        self.n_copies = 4
        self.n_irrep = 4

        # Readout input dimension: K-th stage S^2 coords per copy per
        # eigenspace = 3 * 4 * 2 = 24.
        self.readout_in = 3 * self.n_copies * self.n_eigspaces  # 24
        self.readout_out = self.READOUT_OUT

        # Parameter layout (flat vector, mutated by ES / SGD):
        #   rotors:   (K, n_eigspaces, 3) — log-quaternion (axis*angle/2)
        #   readout:  (readout_out, readout_in) — 3 x 24 = 72 weights
        # Total: 2*2*3 + 72 = 12 + 72 = 84
        self.rotor_shape = (self.K, self.n_eigspaces, 3)
        self.readout_shape = (self.readout_out, self.readout_in)

        self.n_rotor = int(np.prod(self.rotor_shape))
        self.n_readout = int(np.prod(self.readout_shape))
        self.n_params = self.n_rotor + self.n_readout

    # ------- parameter (un)flattening ----------------------------------
    def init_params(self, rng: np.random.Generator) -> np.ndarray:
        """Initialize a flat parameter vector. Rotors near identity; readout small Gaussian."""
        rotors = rng.normal(scale=0.05, size=self.rotor_shape)
        readout = rng.normal(scale=0.05, size=self.readout_shape)
        return np.concatenate([rotors.ravel(), readout.ravel()])

    def split(self, theta: np.ndarray):
        rotors = theta[: self.n_rotor].reshape(self.rotor_shape)
        readout = theta[self.n_rotor :].reshape(self.readout_shape)
        return rotors, readout

    @staticmethod
    def _exp_so3_to_quat(v: np.ndarray) -> np.ndarray:
        """Map R^3 -> S^3 via q = (cos|v|, sin|v|/|v| * v). Identity at v=0."""
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        c, s = math.cos(n), math.sin(n) / n
        return np.array([c, s * v[0], s * v[1], s * v[2]])

    # ------- forward pass (vectorized) --------------------------------
    @staticmethod
    def _qmul_batch(rotor: np.ndarray, qs: np.ndarray) -> np.ndarray:
        """Hamilton product rotor (4,) * qs (..., 4)."""
        w1, x1, y1, z1 = rotor
        w2 = qs[..., 0]; x2 = qs[..., 1]; y2 = qs[..., 2]; z2 = qs[..., 3]
        out = np.empty_like(qs)
        out[..., 0] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        out[..., 1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        out[..., 2] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        out[..., 3] = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return out

    @staticmethod
    def _hopf_project_batch(qs: np.ndarray) -> np.ndarray:
        """Hopf S^3 -> S^2 broadcast over last axis. qs: (..., 4) -> (..., 3)."""
        w, a, b, c = qs[..., 0], qs[..., 1], qs[..., 2], qs[..., 3]
        out = np.empty(qs.shape[:-1] + (3,), dtype=qs.dtype)
        out[..., 0] = 2 * (w * b + a * c)
        out[..., 1] = 2 * (w * c - a * b)
        out[..., 2] = w * w + a * a - b * b - c * c
        return out

    @staticmethod
    def _hopf_section_batch(ps: np.ndarray) -> np.ndarray:
        """S^2 -> S^3 canonical-section broadcast. ps: (..., 3) -> (..., 4)."""
        px, py, pz = ps[..., 0], ps[..., 1], ps[..., 2]
        w = 1.0 + pz
        n = np.sqrt(w * w + px * px + py * py)
        safe = n > 1e-10
        out = np.zeros(ps.shape[:-1] + (4,), dtype=ps.dtype)
        # Default branch: south pole fallback (0,0,1,0).
        out[..., 2] = 1.0
        # Main branch: w > 0 generic.
        np.divide(w, n, out=out[..., 0], where=safe)
        np.divide(px, n, out=out[..., 2], where=safe)
        np.divide(py, n, out=out[..., 3], where=safe)
        return out

    def act(self, obs: np.ndarray, theta: np.ndarray) -> int:
        # Lightweight, unrolled forward. Tensor shapes are tiny (8 quaternions
        # at most), so numpy dispatch dominates over arithmetic; we keep array
        # ops to a minimum.
        rotors = theta[: self.n_rotor].reshape(self.K, self.n_eigspaces, 3)
        readout = theta[self.n_rotor :].reshape(self.readout_out, self.readout_in)

        q = acrobot_obs_to_quat(obs)

        speed = math.sqrt(obs[4] * obs[4] + obs[5] * obs[5])
        kappa = self.KAPPA_BASE + self.KAPPA_VEL * min(speed, 8.0)
        f = vmf_activation(q, kappa=kappa)

        # Pack both eigenspaces into one (8, 4) tensor of "quaternion coefficients".
        # E_packed: (120, 8, 4) = stack of [E3_basis, E7_basis] along the copy axis.
        E_packed = self._E_packed
        coeffs = np.einsum("i,ick->ck", f, E_packed)  # (8, 4)

        # Pre-compute rotor quaternions for both stages (4 rotors total).
        # Each stage has 2 rotors (one per eigenspace), broadcast over 4 copies.
        for stage in range(self.K):
            mags = np.sqrt((coeffs * coeffs).sum(axis=-1, keepdims=True))  # (8, 1)
            safe = (mags[:, 0] > 1e-10)
            inv_mag = np.where(safe[:, None], 1.0 / np.where(safe[:, None], mags, 1.0), 0.0)
            qs = coeffs * inv_mag  # (8, 4)

            # Apply rotors: rotor[0] to qs[:4], rotor[1] to qs[4:].
            r0 = self._exp_so3_to_quat(rotors[stage, 0])
            r1 = self._exp_so3_to_quat(rotors[stage, 1])
            qs[:4] = self._qmul_batch(r0, qs[:4])
            qs[4:] = self._qmul_batch(r1, qs[4:])

            # Hopf-project and section-lift.
            w = qs[:, 0]; a = qs[:, 1]; b = qs[:, 2]; c = qs[:, 3]
            px = 2 * (w * b + a * c)
            py = 2 * (w * c - a * b)
            pz = w * w + a * a - b * b - c * c

            sw = 1.0 + pz
            sn = np.sqrt(sw * sw + px * px + py * py)
            sn_safe = sn > 1e-10
            inv_sn = np.where(sn_safe, 1.0 / np.where(sn_safe, sn, 1.0), 0.0)
            sec = np.empty_like(qs)
            sec[:, 0] = sw * inv_sn
            sec[:, 1] = 0.0
            sec[:, 2] = px * inv_sn
            sec[:, 3] = py * inv_sn
            # Fallback for sn ~= 0: south pole.
            sec[~sn_safe] = (0.0, 0.0, 1.0, 0.0)

            warp = 2.0 * np.tanh(mags * 0.5)  # (8, 1)
            coeffs = sec * warp
            coeffs[~safe] = 0.0

        # Final feature: Hopf-project unit form, scale by magnitude.
        mags = np.sqrt((coeffs * coeffs).sum(axis=-1))  # (8,)
        safe = mags > 1e-10
        inv = np.where(safe, 1.0 / np.where(safe, mags, 1.0), 0.0)
        w = coeffs[:, 0] * inv
        a = coeffs[:, 1] * inv
        b = coeffs[:, 2] * inv
        c = coeffs[:, 3] * inv
        feat = np.empty(self.readout_in, dtype=np.float64)
        feat[0::3] = 2 * (w * b + a * c) * mags
        feat[1::3] = 2 * (w * c - a * b) * mags
        feat[2::3] = (w * w + a * a - b * b - c * c) * mags

        return int(np.argmax(readout @ feat))


# ---------------------------------------------------------------------------
# MLP policy: 6 -> 8 -> 3 tanh
# ---------------------------------------------------------------------------


class MLPPolicy:
    """Standard small MLP for parameter-matched comparison.

    Parameters: 6*8 + 8 + 8*3 + 3 = 48 + 8 + 24 + 3 = 83.
    """

    OBS_DIM = 6
    HIDDEN = 8
    N_ACTIONS = 3

    def __init__(self) -> None:
        self.W1_shape = (self.HIDDEN, self.OBS_DIM)
        self.b1_shape = (self.HIDDEN,)
        self.W2_shape = (self.N_ACTIONS, self.HIDDEN)
        self.b2_shape = (self.N_ACTIONS,)
        self.n_W1 = int(np.prod(self.W1_shape))
        self.n_b1 = int(np.prod(self.b1_shape))
        self.n_W2 = int(np.prod(self.W2_shape))
        self.n_b2 = int(np.prod(self.b2_shape))
        self.n_params = self.n_W1 + self.n_b1 + self.n_W2 + self.n_b2

    def init_params(self, rng: np.random.Generator) -> np.ndarray:
        # Xavier-ish init for tanh.
        s1 = math.sqrt(1.0 / self.OBS_DIM)
        s2 = math.sqrt(1.0 / self.HIDDEN)
        W1 = rng.normal(scale=s1, size=self.W1_shape)
        b1 = np.zeros(self.b1_shape)
        W2 = rng.normal(scale=s2, size=self.W2_shape)
        b2 = np.zeros(self.b2_shape)
        return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

    def split(self, theta: np.ndarray):
        i = 0
        W1 = theta[i : i + self.n_W1].reshape(self.W1_shape); i += self.n_W1
        b1 = theta[i : i + self.n_b1].reshape(self.b1_shape); i += self.n_b1
        W2 = theta[i : i + self.n_W2].reshape(self.W2_shape); i += self.n_W2
        b2 = theta[i : i + self.n_b2].reshape(self.b2_shape); i += self.n_b2
        return W1, b1, W2, b2

    def forward_logits(self, obs: np.ndarray, theta: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2 = self.split(theta)
        h = np.tanh(W1 @ obs + b1)
        return W2 @ h + b2

    def act(self, obs: np.ndarray, theta: np.ndarray) -> int:
        return int(np.argmax(self.forward_logits(obs, theta)))

    def softmax_probs(self, obs: np.ndarray, theta: np.ndarray) -> np.ndarray:
        z = self.forward_logits(obs, theta)
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()


# ---------------------------------------------------------------------------
# Random policy (no parameters)
# ---------------------------------------------------------------------------


class RandomPolicy:
    n_params = 0

    def init_params(self, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(0)

    def act(self, obs: np.ndarray, theta: np.ndarray) -> int:
        # We deliberately use the global RNG to avoid threading a generator
        # through the rollout API. Random search reproducibility is via the
        # rollout seed, not the action-selection RNG.
        return int(np.random.randint(3))
