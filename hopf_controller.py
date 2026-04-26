# ================================================================
# Hopf Geometric Feature Extractor
#
# This module contains geometric primitives on S³ / S² and three
# controller classes built on them. None of the classes implement a
# multi-stage Hopf architecture with Hopf projection *between*
# computational stages and accumulated holonomy across passes — that
# architecture is described in the original v2 sketch and has not
# been built. The current MNIST production accuracy path is
# ADEHopfController (v8/v9) wrapped by the multi-scale kappa /
# Nystrom polynomial kernel ridge in `train_ade_hopf.py` (v10).
# Best MNIST result on record: 97.39% (v10). Adding the Ω² / Ω³
# face and cell eigenspaces (v11 in `train_v11.py`, v12 in
# `train_v12.py`) does not improve MNIST accuracy over v10.
#
# Primitives:
#   hopf_project        — S³ → S² quadratic map. THIS IS THE NONLINEARITY.
#   hopf_section        — S² → S³ section (gauge-dependent)
#   hopf_lift           — section + fiber rotation
#   pancharatnam_phase  — discrete parallel-transport phase between two S²
#   solid_angle_triangle— unsigned spherical triangle area on S²
#   holonomy_triangle   — signed Berry phase + transport phase for a
#                         closed S² triangle (Euler–Eriksson route)
#   triangle_berry_clifford — independent signed Berry phase via actual
#                         Cl(3,0) rotor composition. Cross-checked
#                         against Euler–Eriksson to machine precision
#                         (see verify_berry_phase).
#   poincare_warp       — radial conformal compression r → 2·tanh(r/2)
#
# Controllers:
#   HopfController       (v6) — per-eigenspace Hopf + McKay E₈ message
#                               passing on rotor parameters that are
#                               LEARNED by GA/ES. Used as the optional
#                               `controller_type="hopf"` path inside
#                               `genreg_genome.py` for the Snake IDE,
#                               and as a v6 MNIST trainer entry point
#                               via `train_hopf_mnist.py`. Not the
#                               current MNIST accuracy path.
#   VertexHopfController (v7) — all 120 vertex activations through a
#                               Hopf hidden layer with rotors LEARNED
#                               by ES (`train_hopf_es.py`,
#                               `train_hopf_rotor_es.py`). Reference
#                               implementation; superseded for MNIST
#                               by ADEHopfController.
#   ADEHopfController    (v8/v9) — ADE-structured feature extraction
#                               with CG cross products and E₈ edge
#                               features. ZERO non-convex parameters:
#                               the features are fixed geometric
#                               functions of the input and only the
#                               linear readout is fit (closed-form
#                               ridge regression). v10 wraps this in
#                               `train_ade_hopf.py` with multi-scale
#                               kappa and polynomial kernel ridge via
#                               Nystrom approximation. Current MNIST
#                               production accuracy path. v11/v12
#                               (separate trainers) extend the
#                               feature set with Ω² face and Ω³ cell
#                               eigenspaces; both saturate at v10's
#                               accuracy and do not improve on it.
#
# Retraction notice. Earlier writeups (deleted per-version FINDINGS.md
# files, now superseded by the top-level FINDINGS.md) interpreted the
# signed Berry phase as a chirality-specific causal signal for MNIST
# cross-digit transfer. Rigorous follow-up runs do not support that
# interpretation: unit weights on the face eigenspace basis match or
# beat signed-Berry-phase weights in every cell tested. The Berry
# phase math here is correct and reusable; the chirality claim made
# on top of it was not. See FINDINGS.md for the full tables and the
# explicit retraction.
#
# Note on terminology: the irrep decomposition is a symmetry constraint
# (it restricts the function class); Hopf projection is the actual
# nonlinearity (it adds representational capacity). Those are not
# interchangeable, and claims that the "irrep boundaries are the
# nonlinearity" are not accurate in this code — `hopf_project` is, and
# is labeled as such at its definition below.
# ================================================================

import math
import random
import numpy as np

from cell600 import get_geometry


# ================================================================
# Geometric operations (pure numpy, verified against clifford Cl(3,0))
# Rotor convention: R = w + a·e12 + b·e13 + c·e23, stored as [w,a,b,c]
# ================================================================

def qmul(a, b):
    """Hamilton product of rotor components [w, e12, e13, e23]."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def qnormalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def hopf_project(q):
    """
    Hopf projection S³ → S² : R·e3·~R expanded in Cl(3,0).
    px = 2(wb + ac), py = 2(wc - ab), pz = w² + a² - b² - c².
    THIS IS THE NONLINEARITY.
    """
    w, a, b, c = q
    return np.array([2*(w*b + a*c), 2*(w*c - a*b), w*w + a*a - b*b - c*c])


def hopf_section(px, py, pz):
    """Section: (1 + p·e3)/|1 + p·e3| in Cl(3,0) → [w, 0, px/n, py/n]."""
    w = 1.0 + pz
    n = math.sqrt(w*w + px*px + py*py)
    if n < 1e-10:
        return np.array([0.0, 0.0, 1.0, 0.0])
    return np.array([w/n, 0.0, px/n, py/n])


def hopf_lift(px, py, pz, phase):
    """Section lift with fiber rotation."""
    sec = hopf_section(px, py, pz)
    fiber = np.array([math.cos(phase/2), math.sin(phase/2), 0.0, 0.0])
    return qnormalize(qmul(sec, fiber))


def pancharatnam_phase(p1, p2):
    """Discrete parallel transport phase between two S² points."""
    r1 = hopf_section(*p1)
    r2 = hopf_section(*p2)
    r1_rev = np.array([r1[0], -r1[1], -r1[2], -r1[3]])
    ov = qmul(r1_rev, r2)
    return math.atan2(-ov[1], ov[0])


def solid_angle_triangle(a, b, c):
    """Solid angle via Euler-Eriksson."""
    triple = np.dot(a, np.cross(b, c))
    denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    if abs(denom) < 1e-12:
        return math.pi
    return 2.0 * math.atan2(abs(triple), denom)


def holonomy_triangle(p1, p2, p_ref):
    """Berry phase + transport phase from closed S² triangle.

    Returns (berry, transport) where `berry` is -½ times the signed
    solid angle of the oriented triangle (v1, v2, vr) on S², and
    `transport` is the sum of three Pancharatnam phases around the
    triangle.

    The math is correct and reusable. An earlier hypothesis that the
    sign of `berry` carries chirality-specific signal for MNIST
    cross-digit transfer via a face-eigenspace weighting (v11/v12) was
    not supported by rigorous follow-up runs; see FINDINGS.md for the
    retraction and the full table of results. The function is kept
    because the geometry is sound and because it remains available for
    any future test on a task where chirality is the only signal.

    See also `triangle_berry_clifford` for an independent
    Clifford-algebraic computation of the same Berry phase, used by
    v11 and v12 to cross-check the Euler–Eriksson route.
    """
    v1, v2, vr = p1.copy(), p2.copy(), p_ref.copy()
    for v in [v1, v2, vr]:
        n = np.linalg.norm(v)
        if n > 1e-10:
            v /= n
    berry = -solid_angle_triangle(v1, v2, vr) / 2.0
    transport = (pancharatnam_phase(v1, v2) +
                 pancharatnam_phase(v2, vr) +
                 pancharatnam_phase(vr, v1))
    return berry, transport


def poincare_warp_scalar(x):
    """Poincaré warp on a scalar: 2·tanh(x/2)."""
    return 2.0 * math.tanh(x / 2.0)


def poincare_warp(v):
    """Radial conformal compression on a vector."""
    r = np.linalg.norm(v)
    if r < 1e-10:
        return v.copy()
    return v * (2.0 * math.tanh(r / 2.0) / r)


def random_unit_quat():
    v = np.random.randn(4)
    return v / np.linalg.norm(v)


# ================================================================
# Mutation helpers
# ================================================================

def mutate_quat(q, rate, scale):
    if random.random() > rate:
        return q.copy()
    v = np.array([random.gauss(0, scale) for _ in range(3)])
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return q.copy()
    delta = np.array([math.cos(nv), math.sin(nv)/nv * v[0],
                      math.sin(nv)/nv * v[1], math.sin(nv)/nv * v[2]])
    return qnormalize(qmul(q, delta))


def mutate_scale(s, rate, scale):
    return s + random.gauss(0, scale * (abs(s) + 1e-9)) if random.random() < rate else s


def mutate_flat_2d(arr, rate, scale):
    mask = np.random.random(arr.shape) < rate
    arr[mask] += np.random.randn(mask.sum()) * scale


def mutate_flat(arr, rate, scale):
    mask = np.random.random(arr.shape) < rate
    arr[mask] += np.random.randn(mask.sum()) * scale


# ================================================================
# Pixel-to-vertex kernel (cached)
# ================================================================

_GEO = None
_PIXEL_KERNEL = None


def _get_geo():
    global _GEO
    if _GEO is None:
        _GEO = get_geometry()
    return _GEO


def _build_pixel_kernel(input_size, kappa=10.0):
    geo = _get_geo()
    vertices = geo["vertices"]
    rows, cols = (28, 28) if input_size == 784 else (
        int(math.ceil(math.sqrt(input_size))),) * 2

    pixel_quats = []
    for idx in range(input_size):
        r, c = divmod(idx, 28) if input_size == 784 else divmod(idx, cols)
        x = 2.0 * c / max(cols - 1, 1) - 1.0
        y = 2.0 * r / max(rows - 1, 1) - 1.0
        denom = 1.0 + x*x + y*y
        X, Y, Z = 2*x/denom, 2*y/denom, (x*x + y*y - 1)/denom
        if Z > -0.999:
            s = math.sqrt(2*(1+Z))
            q = np.array([s/2, -Y/s, X/s, 0.0])
        else:
            q = np.array([0.0, 0.0, 1.0, 0.0])
        pixel_quats.append(q / np.linalg.norm(q))

    pixel_quats = np.array(pixel_quats)
    dots = np.abs(pixel_quats @ vertices.T)
    scaled = kappa * dots - (kappa * dots).max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float64)


_PIXEL_KAPPA = None

def _get_pixel_kernel(input_size, kappa=10.0):
    global _PIXEL_KERNEL, _PIXEL_KAPPA
    if _PIXEL_KERNEL is None or _PIXEL_KERNEL.shape[0] != input_size or _PIXEL_KAPPA != kappa:
        _PIXEL_KERNEL = _build_pixel_kernel(input_size, kappa)
        _PIXEL_KAPPA = kappa
    return _PIXEL_KERNEL


# ================================================================
# HopfController v6 — Per-Eigenspace Hopf Projections
# ================================================================

class HopfController:
    """
    v6: Every eigenspace contributes DIRECTION, not just energy.

    v5 crushed 8 of 9 eigenspaces to scalar norms, losing all spatial
    phase information. v6 Hopf-projects each eigenspace's first 4D
    coefficients through a learned rotor, extracting 3 S² directional
    features per eigenspace. The fine-grained spatial detail that
    distinguishes "1" from "7" now reaches the classifier.

    50 features from:
      - 1 scalar (E₀)
      - 24 Hopf S² coords (8 eigenspaces × 3)
      - 9 McKay message-passed energies
      - 12 curl Hopf S² coords (4 curl eigenspaces × 3)
      - 4 curl norms (Poincaré warped)

    ~582 parameters.
    """

    N_MCKAY_ROUNDS = 3

    def __init__(self, input_size=784, hidden_size=16, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        geo = _get_geo()
        _get_pixel_kernel(input_size)

        # Per-eigenspace rotors for scalar eigenspaces (right-multiply before Hopf)
        # E₀ (dim 1) gets None; E₁-E₈ (dim ≥ 4) each get a learned rotor
        self.sc_rotors = []
        for es in geo["scalar_eigenspaces"]:
            if es["multiplicity"] >= 4:
                self.sc_rotors.append(random_unit_quat())
            else:
                self.sc_rotors.append(None)

        # Per-eigenspace rotors for curl eigenspaces (all have dim ≥ 6)
        self.cu_rotors = [random_unit_quat()
                          for _ in geo["curl_eigenspaces"]]

        # McKay coupling weights on 8 E₈ edges × K rounds
        self.mckay_couplings = np.random.randn(
            self.N_MCKAY_ROUNDS, 8) * 0.1

        # Feature count
        n_sc_hopf = sum(1 for r in self.sc_rotors if r is not None)
        n_sc_scalar = sum(1 for r in self.sc_rotors if r is None)
        n_cu = len(self.cu_rotors)
        self.n_features = (
            n_sc_scalar +          # E₀ scalar (Poincaré warped)
            n_sc_hopf * 3 +        # Hopf S² per eigenspace (scaled by norm)
            9 +                    # McKay message-passed node energies
            n_cu * 3 +             # curl Hopf S² coords (scaled by norm)
            n_cu                   # curl norms (Poincaré warped)
        )

        xavier = math.sqrt(2.0 / (self.n_features + output_size))
        self.W_out = np.random.randn(output_size, self.n_features) * xavier
        self.b_out = np.zeros(output_size)

    def _mckay_message_pass(self, node_energies):
        """
        Message passing on the extended E₈ Dynkin diagram.
        9 node energies propagate along 8 edges for K rounds.
        Poincaré warp after each round.
        """
        geo = _get_geo()
        e8_edges = geo["e8_edges"]
        energies = node_energies.copy()

        for rnd in range(self.N_MCKAY_ROUNDS):
            new_e = energies.copy()
            for edge_idx, (i, j) in enumerate(e8_edges):
                w = self.mckay_couplings[rnd, edge_idx]
                new_e[j] += energies[i] * w
                new_e[i] += energies[j] * w
            energies = np.array([poincare_warp_scalar(e) for e in new_e])

        return energies

    def features_from_activation(self, f):
        """Build the geometric feature vector from a 120-vertex activation
        using the per-eigenspace Hopf projections + McKay message passing.

        Exposed so other controllers (SnakeHopfController) can substitute
        their own activation construction while reusing the eigenspace
        machinery.
        """
        geo = _get_geo()
        sc = [es["vectors"].T @ f for es in geo["scalar_eigenspaces"]]
        df = geo["d0"] @ f
        cu = [es["vectors"].T @ df for es in geo["curl_eigenspaces"]]

        features = []

        # --- Scalar eigenspaces ---
        norms = []
        for i, coeffs in enumerate(sc):
            norm = np.linalg.norm(coeffs)
            norms.append(norm)

            if self.sc_rotors[i] is not None:
                c4 = coeffs[:4]
                n4 = np.linalg.norm(c4)
                if n4 > 1e-10:
                    q = c4 / n4
                    q_rot = qnormalize(qmul(q, self.sc_rotors[i]))
                    p = hopf_project(q_rot)
                    scale = min(norm, 10.0)
                    features.extend((p * scale).tolist())
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.append(poincare_warp_scalar(coeffs[0]))

        # --- McKay message passing on eigenspace norms ---
        eigenspace_to_e8 = geo["eigenspace_to_e8"]
        node_energies = np.zeros(9)
        for i, norm in enumerate(norms):
            node_energies[eigenspace_to_e8[i]] = norm
        mckay_out = self._mckay_message_pass(node_energies)
        features.extend(mckay_out.tolist())

        # --- Curl eigenspaces: Hopf projection + norm ---
        for i, coeffs in enumerate(cu):
            norm = np.linalg.norm(coeffs)
            c4 = coeffs[:4]
            n4 = np.linalg.norm(c4)
            if n4 > 1e-10:
                q = c4 / n4
                q_rot = qnormalize(qmul(q, self.cu_rotors[i]))
                p = hopf_project(q_rot)
                scale = min(norm, 10.0)
                features.extend((p * scale).tolist())
            else:
                features.extend([0.0, 0.0, 0.0])
            features.append(poincare_warp_scalar(norm))

        return np.array(features)

    def forward(self, inputs):
        """Forward pass: pixel kernel → activation → features → readout."""
        kernel = _get_pixel_kernel(self.input_size)
        x = np.asarray(inputs, dtype=np.float64)
        if len(x) < self.input_size:
            x = np.pad(x, (0, self.input_size - len(x)))
        x = x[:self.input_size]
        f = x @ kernel  # (120,)
        features = self.features_from_activation(f)
        logits = self.W_out @ features + self.b_out
        return logits.tolist()

    def select_action(self, signals, signal_order):
        inputs = [signals.get(k, 0.0) for k in signal_order]
        logits = self.forward(inputs)
        return logits.index(max(logits))

    def mutate(self, rate=0.1, scale=0.3):
        # Geometric params: full mutation scale
        for i in range(len(self.sc_rotors)):
            if self.sc_rotors[i] is not None:
                self.sc_rotors[i] = mutate_quat(self.sc_rotors[i], rate, scale)
        for i in range(len(self.cu_rotors)):
            self.cu_rotors[i] = mutate_quat(self.cu_rotors[i], rate, scale)
        for rnd in range(self.N_MCKAY_ROUNDS):
            for e in range(8):
                self.mckay_couplings[rnd, e] = mutate_scale(
                    self.mckay_couplings[rnd, e], rate, scale)
        # Readout: half scale to preserve learned structure
        mutate_flat_2d(self.W_out, rate, scale * 0.5)
        mutate_flat(self.b_out, rate, scale * 0.5)

    def crossover(self, other):
        """Uniform crossover: randomly pick params from self or other."""
        child = self.clone()
        for i in range(len(child.sc_rotors)):
            if child.sc_rotors[i] is not None and random.random() < 0.5:
                child.sc_rotors[i] = other.sc_rotors[i].copy()
        for i in range(len(child.cu_rotors)):
            if random.random() < 0.5:
                child.cu_rotors[i] = other.cu_rotors[i].copy()
        mask = np.random.random(child.mckay_couplings.shape) < 0.5
        child.mckay_couplings[mask] = other.mckay_couplings[mask]
        mask = np.random.random(child.W_out.shape) < 0.5
        child.W_out[mask] = other.W_out[mask]
        mask = np.random.random(child.b_out.shape) < 0.5
        child.b_out[mask] = other.b_out[mask]
        return child

    def clone(self):
        c = HopfController.__new__(HopfController)
        c.input_size = self.input_size
        c.hidden_size = self.hidden_size
        c.output_size = self.output_size
        c.N_MCKAY_ROUNDS = self.N_MCKAY_ROUNDS
        c.n_features = self.n_features
        c.sc_rotors = [q.copy() if q is not None else None
                       for q in self.sc_rotors]
        c.cu_rotors = [q.copy() for q in self.cu_rotors]
        c.mckay_couplings = self.mckay_couplings.copy()
        c.W_out = self.W_out.copy()
        c.b_out = self.b_out.copy()
        return c

    def to_flat(self):
        """Flatten all parameters to a single vector for ES optimization."""
        parts = []
        for r in self.sc_rotors:
            if r is not None:
                parts.append(r)
        for r in self.cu_rotors:
            parts.append(r)
        parts.append(self.mckay_couplings.ravel())
        parts.append(self.W_out.ravel())
        parts.append(self.b_out)
        return np.concatenate(parts)

    def from_flat(self, flat):
        """Load parameters from a flat vector. Normalizes rotors to S³."""
        idx = 0
        for i in range(len(self.sc_rotors)):
            if self.sc_rotors[i] is not None:
                self.sc_rotors[i] = qnormalize(flat[idx:idx+4].copy())
                idx += 4
        for i in range(len(self.cu_rotors)):
            self.cu_rotors[i] = qnormalize(flat[idx:idx+4].copy())
            idx += 4
        n_mc = self.mckay_couplings.size
        self.mckay_couplings = flat[idx:idx+n_mc].reshape(
            self.mckay_couplings.shape).copy()
        idx += n_mc
        n_w = self.W_out.size
        self.W_out = flat[idx:idx+n_w].reshape(self.W_out.shape).copy()
        idx += n_w
        self.b_out = flat[idx:idx+self.b_out.size].copy()

    def param_count(self):
        n_sc = sum(4 for r in self.sc_rotors if r is not None)
        n_cu = len(self.cu_rotors) * 4
        return (
            n_sc + n_cu +
            self.mckay_couplings.size +
            self.W_out.size +
            self.b_out.size
        )

    def effective_dof(self):
        # Rotors on S³ have 3 DOF each
        n_sc = sum(3 for r in self.sc_rotors if r is not None)
        n_cu = len(self.cu_rotors) * 3
        return (
            n_sc + n_cu +
            self.mckay_couplings.size +
            self.W_out.size +
            self.b_out.size
        )

    def to_dict(self):
        return {
            "type": "hopf", "version": 6,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "n_mckay_rounds": self.N_MCKAY_ROUNDS,
            "sc_rotors": [q.tolist() if q is not None else None
                          for q in self.sc_rotors],
            "cu_rotors": [q.tolist() for q in self.cu_rotors],
            "mckay_couplings": self.mckay_couplings.tolist(),
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        version = d.get("version", 5)
        if version < 6:
            raise ValueError(
                f"Cannot load v{version} checkpoint into v6 controller. "
                f"Architecture changed: per-eigenspace Hopf projections."
            )
        c = cls.__new__(cls)
        c.input_size = d["input_size"]
        c.hidden_size = d.get("hidden_size", 16)
        c.output_size = d["output_size"]
        c.N_MCKAY_ROUNDS = d.get("n_mckay_rounds", 3)
        c.sc_rotors = [np.array(q) if q is not None else None
                       for q in d["sc_rotors"]]
        c.cu_rotors = [np.array(q) for q in d["cu_rotors"]]
        c.mckay_couplings = np.array(d["mckay_couplings"], dtype=np.float64)
        c.W_out = np.array(d["W_out"], dtype=np.float64)
        c.b_out = np.array(d["b_out"], dtype=np.float64)
        # Recompute n_features
        n_sc_hopf = sum(1 for r in c.sc_rotors if r is not None)
        n_sc_scalar = sum(1 for r in c.sc_rotors if r is None)
        n_cu = len(c.cu_rotors)
        c.n_features = n_sc_scalar + n_sc_hopf * 3 + 9 + n_cu * 3 + n_cu
        return c


# ================================================================
# VertexHopfController v7 — Full Vertex Activations + Hopf Hidden Layer
# ================================================================

class VertexHopfController:
    """
    v7: Uses ALL 120 vertex activations through a Hopf hidden layer.

    Diagnostic showed: optimal linear on 120 vertex activations = 76%.
    v5/v6 threw away info compressing to 36/50 features, capping at ~59%.

    v7: No compression. All 120 vertex activations are grouped into
    30 × 4D, each processed by a learned rotor + Hopf projection.
    This gives 30 × 4 = 120 features (3 S² + 1 magnitude per group).
    Linear readout classifies from these 120 nonlinear features.

    The Hopf projection is a quadratic nonlinearity — each unit extracts
    a specific orientation-dependent feature from its 4D vertex group.
    30 different rotors = 30 different quadratic feature detectors.

    ~1330 params (still 38x fewer than MLP baseline).
    """

    N_GROUPS = 30  # 120 vertices / 4 = 30 groups

    def __init__(self, input_size=784, hidden_size=16, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        _get_geo()
        _get_pixel_kernel(input_size)

        # 30 right-rotors for Hopf hidden layer
        self.rotors = [random_unit_quat() for _ in range(self.N_GROUPS)]

        # Features: 30 × (3 S² + 1 magnitude) = 120
        self.n_features = self.N_GROUPS * 4

        xavier = math.sqrt(2.0 / (self.n_features + output_size))
        self.W_out = np.random.randn(output_size, self.n_features) * xavier
        self.b_out = np.zeros(output_size)

    def forward(self, inputs):
        kernel = _get_pixel_kernel(self.input_size)
        x = np.asarray(inputs, dtype=np.float64)
        if len(x) < self.input_size:
            x = np.pad(x, (0, self.input_size - len(x)))
        x = x[:self.input_size]

        f = x @ kernel  # (120,)

        features = np.empty(self.n_features)
        for g in range(self.N_GROUPS):
            c4 = f[g*4:(g+1)*4]
            mag = np.linalg.norm(c4)
            base = g * 4
            if mag > 1e-10:
                q = c4 / mag
                q_rot = qnormalize(qmul(q, self.rotors[g]))
                p = hopf_project(q_rot)
                scale = min(mag, 10.0)
                features[base] = p[0] * scale
                features[base+1] = p[1] * scale
                features[base+2] = p[2] * scale
            else:
                features[base] = 0.0
                features[base+1] = 0.0
                features[base+2] = 0.0
            features[base+3] = poincare_warp_scalar(mag)

        logits = self.W_out @ features + self.b_out
        return logits.tolist()

    def select_action(self, signals, signal_order):
        inputs = [signals.get(k, 0.0) for k in signal_order]
        logits = self.forward(inputs)
        return logits.index(max(logits))

    def mutate(self, rate=0.1, scale=0.3):
        for i in range(self.N_GROUPS):
            self.rotors[i] = mutate_quat(self.rotors[i], rate, scale)
        mutate_flat_2d(self.W_out, rate, scale * 0.5)
        mutate_flat(self.b_out, rate, scale * 0.5)

    def crossover(self, other):
        child = self.clone()
        for i in range(self.N_GROUPS):
            if random.random() < 0.5:
                child.rotors[i] = other.rotors[i].copy()
        mask = np.random.random(child.W_out.shape) < 0.5
        child.W_out[mask] = other.W_out[mask]
        mask = np.random.random(child.b_out.shape) < 0.5
        child.b_out[mask] = other.b_out[mask]
        return child

    def clone(self):
        c = VertexHopfController.__new__(VertexHopfController)
        c.input_size = self.input_size
        c.hidden_size = self.hidden_size
        c.output_size = self.output_size
        c.N_GROUPS = self.N_GROUPS
        c.n_features = self.n_features
        c.rotors = [q.copy() for q in self.rotors]
        c.W_out = self.W_out.copy()
        c.b_out = self.b_out.copy()
        return c

    def to_flat(self):
        parts = [r for r in self.rotors]
        parts.append(self.W_out.ravel())
        parts.append(self.b_out)
        return np.concatenate(parts)

    def from_flat(self, flat):
        idx = 0
        for i in range(self.N_GROUPS):
            self.rotors[i] = qnormalize(flat[idx:idx+4].copy())
            idx += 4
        n_w = self.W_out.size
        self.W_out = flat[idx:idx+n_w].reshape(self.W_out.shape).copy()
        idx += n_w
        self.b_out = flat[idx:idx+self.b_out.size].copy()

    def param_count(self):
        return self.N_GROUPS * 4 + self.W_out.size + self.b_out.size

    def effective_dof(self):
        return self.N_GROUPS * 3 + self.W_out.size + self.b_out.size

    def to_dict(self):
        return {
            "type": "hopf", "version": 7,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "n_groups": self.N_GROUPS,
            "rotors": [q.tolist() for q in self.rotors],
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        c = cls.__new__(cls)
        c.input_size = d["input_size"]
        c.hidden_size = d.get("hidden_size", 16)
        c.output_size = d["output_size"]
        c.N_GROUPS = d.get("n_groups", 30)
        c.n_features = c.N_GROUPS * 4
        c.rotors = [np.array(q) for q in d["rotors"]]
        c.W_out = np.array(d["W_out"], dtype=np.float64)
        c.b_out = np.array(d["b_out"], dtype=np.float64)
        return c


# ================================================================
# ADEHopfController v8 — ADE-Structured Spectral Hopf
# ================================================================

class ADEHopfController:
    """
    v8: ADE-structured feature extraction with principled nonlinearities.

    Key improvements over v7:
    - Eigenspace decomposition respects ADE structure (not arbitrary 4-tuples)
    - Irrep copy decomposition via group orbit method
    - Hopf projection ONLY on spin-1/2 eigenspaces (E1, E8) where exact
    - CG cross products (character projector to V_1) for real-type irreps
    - Hopf on 4-tuples for eigenspaces without CG (practical nonlinearity)
    - E8 Dynkin edge features (norm products/ratios)

    Zero non-convex parameters: all features are fixed geometric functions
    of the input. Only the linear readout is learned (via ridge regression).
    """

    def __init__(self, input_size=784, hidden_size=16, output_size=10):
        from ade_geometry import get_ade
        self.input_size = input_size
        self.output_size = output_size

        ade = get_ade()
        _get_pixel_kernel(input_size)

        # Count features
        self.n_features = self._count_features(ade)

        xavier = math.sqrt(2.0 / (self.n_features + output_size))
        self.W_out = np.random.randn(output_size, self.n_features) * xavier
        self.b_out = np.zeros(output_size)

    @staticmethod
    def _count_features(ade):
        n = 0
        for idx, aes in enumerate(ade["ade_eigenspaces"]):
            d = aes["d"]
            d2 = aes["d2"]
            copies = aes["copies"]
            cg = aes["cg_v1"]

            if d == 1:
                n += 1
            elif d == 2:
                # Hopf on full 4D: 3 S2 + 1 mag
                n += 4
            else:
                # Copy magnitudes
                n += len(copies)
                # CG cross products (all pairs)
                if cg is not None:
                    n_pairs = len(copies) * (len(copies) - 1) // 2
                    n += n_pairs * cg.shape[0]
                else:
                    # Fallback: Hopf on 4-tuples of eigenspace coefficients
                    n_hopf = d2 // 4
                    n_leftover = d2 % 4
                    n += n_hopf * 4 + n_leftover

        # Curl eigenspace features (co-exact 1-forms, Theorem 5)
        # Edge/differential reading: how signals flow through the net
        for ces in ade.get("curl_eigenspaces", []):
            mult = ces["multiplicity"]
            n_hopf = mult // 4
            n_leftover = mult % 4
            n += n_hopf * 4 + n_leftover

        # E8 edge features: 2 norm + 2 directional per edge
        n += len(ade["e8_edges"]) * 4
        return n

    def extract_features(self, inputs):
        """Extract the full feature vector from raw pixel inputs."""
        from ade_geometry import get_ade
        ade = get_ade()
        kernel = _get_pixel_kernel(self.input_size)

        x = np.asarray(inputs, dtype=np.float64)
        if len(x) < self.input_size:
            x = np.pad(x, (0, self.input_size - len(x)))
        x = x[:self.input_size]

        f = x @ kernel  # (120,) vertex activations

        features = []
        es_norms = []
        es_hopf_vecs = {}  # eigenspace index -> S2 vector for E8 interactions

        for idx, aes in enumerate(ade["ade_eigenspaces"]):
            V = aes["V"]
            d = aes["d"]
            d2 = aes["d2"]
            copies = aes["copies"]
            cg = aes["cg_v1"]

            # Project to eigenspace
            coeffs = V.T @ f  # (d2,)
            es_norms.append(np.linalg.norm(coeffs))

            if d == 1:
                features.append(poincare_warp_scalar(coeffs[0]))

            elif d == 2:
                # Hopf on full 4D eigenspace (exact for spin-1/2)
                q = coeffs
                mag = np.linalg.norm(q)
                if mag > 1e-10:
                    p = hopf_project(q / mag)
                    scale = min(mag, 10.0)
                    features.extend((p * scale).tolist())
                    es_hopf_vecs[idx] = p  # store unit S2 vector
                else:
                    features.extend([0.0, 0.0, 0.0])
                features.append(poincare_warp_scalar(mag))

            else:
                # Decompose into copies
                copy_vecs = [cb.T @ coeffs for cb in copies]

                # Store first copy direction as S2 representative
                if len(copy_vecs[0]) >= 4:
                    c4 = copy_vecs[0][:4]
                    c4_mag = np.linalg.norm(c4)
                    if c4_mag > 1e-10:
                        es_hopf_vecs[idx] = hopf_project(c4 / c4_mag)

                # Copy magnitudes
                for v in copy_vecs:
                    features.append(poincare_warp_scalar(np.linalg.norm(v)))

                if cg is not None:
                    # CG cross products: all pairs -> V_1 projection
                    nc = len(copy_vecs)
                    for a in range(nc):
                        for b in range(a + 1, nc):
                            kron_ab = np.kron(copy_vecs[a], copy_vecs[b])
                            w = cg @ kron_ab
                            features.extend(w.tolist())
                else:
                    # No CG: Hopf on 4-tuples of eigenspace coeffs
                    n_hopf = d2 // 4
                    for g in range(n_hopf):
                        c4 = coeffs[g*4:(g+1)*4]
                        mag = np.linalg.norm(c4)
                        if mag > 1e-10:
                            p = hopf_project(c4 / mag)
                            scale = min(mag, 10.0)
                            features.extend((p * scale).tolist())
                        else:
                            features.extend([0.0, 0.0, 0.0])
                        features.append(poincare_warp_scalar(mag))
                    # Leftover coefficients
                    leftover = d2 % 4
                    if leftover > 0:
                        for k in range(leftover):
                            features.append(
                                poincare_warp_scalar(coeffs[n_hopf*4 + k]))

        # --- Curl eigenspace features (edge/differential reading) ---
        # Co-exact 1-forms (Theorem 5): how signals flow through the net.
        # By Hodge orthogonality, d0@f is exact and ⊥ co-exact eigenspaces.
        # Use multiplicative edge signal h_e = f_i * f_j to access curl modes:
        # this quadratic interaction breaks the exact/co-exact barrier.
        edge_list = ade["edges"]
        h_mult = np.array([f[i] * f[j] for i, j in edge_list])

        for ces in ade.get("curl_eigenspaces", []):
            V_curl = ces["vectors"]        # (720, mult)
            mult = ces["multiplicity"]
            curl_coeffs = V_curl.T @ h_mult  # (mult,)

            n_hopf = mult // 4
            for g in range(n_hopf):
                c4 = curl_coeffs[g*4:(g+1)*4]
                mag = np.linalg.norm(c4)
                if mag > 1e-10:
                    p = hopf_project(c4 / mag)
                    scale = min(mag, 10.0)
                    features.extend((p * scale).tolist())
                else:
                    features.extend([0.0, 0.0, 0.0])
                features.append(poincare_warp_scalar(mag))
            leftover = mult % 4
            if leftover > 0:
                for k in range(leftover):
                    features.append(
                        poincare_warp_scalar(curl_coeffs[n_hopf*4 + k]))

        # E8 edge features: norm interactions + directional interactions
        # The E8 edges encode which irrep transitions are geometrically allowed
        e8_edges = ade["e8_edges"]
        e8_to_es = ade["e8_to_eigenspace"]
        for ni, nj in e8_edges:
            ei = e8_to_es[ni]
            ej = e8_to_es[nj]
            ni_val = es_norms[ei]
            nj_val = es_norms[ej]
            # Norm interactions
            features.append(poincare_warp_scalar(ni_val * nj_val))
            features.append(poincare_warp_scalar(
                ni_val / (nj_val + 1e-6) - nj_val / (ni_val + 1e-6)))
            # Directional interactions along lawful E8 transitions
            if ei in es_hopf_vecs and ej in es_hopf_vecs:
                hi = es_hopf_vecs[ei]
                hj = es_hopf_vecs[ej]
                features.append(np.dot(hi, hj))                     # alignment
                features.append(np.linalg.norm(np.cross(hi, hj)))   # orthogonality
            else:
                features.append(0.0)
                features.append(0.0)

        return np.array(features)

    def forward(self, inputs):
        features = self.extract_features(inputs)
        logits = self.W_out @ features + self.b_out
        return logits.tolist()

    def select_action(self, signals, signal_order):
        inputs = [signals.get(k, 0.0) for k in signal_order]
        logits = self.forward(inputs)
        return logits.index(max(logits))

    def to_flat(self):
        return np.concatenate([self.W_out.ravel(), self.b_out])

    def from_flat(self, flat):
        n_w = self.W_out.size
        self.W_out = flat[:n_w].reshape(self.W_out.shape).copy()
        self.b_out = flat[n_w:n_w + self.b_out.size].copy()

    def param_count(self):
        return self.W_out.size + self.b_out.size

    def effective_dof(self):
        return self.param_count()

    def clone(self):
        c = ADEHopfController.__new__(ADEHopfController)
        c.input_size = self.input_size
        c.output_size = self.output_size
        c.n_features = self.n_features
        c.W_out = self.W_out.copy()
        c.b_out = self.b_out.copy()
        return c

    def to_dict(self):
        return {
            "type": "hopf", "version": 8,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_features": self.n_features,
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        c = cls.__new__(cls)
        c.input_size = d["input_size"]
        c.output_size = d["output_size"]
        c.n_features = d["n_features"]
        c.W_out = np.array(d["W_out"], dtype=np.float64)
        c.b_out = np.array(d["b_out"], dtype=np.float64)
        return c


# ================================================================
# Clifford verification (import-time)
# ================================================================

def verify_geometric_ops(n_trials=20):
    """Verify numpy formulas against clifford Cl(3,0)."""
    import clifford as cf
    layout, blades = cf.Cl(3)
    ce3 = blades['e3']
    ce12, ce13, ce23 = blades['e12'], blades['e13'], blades['e23']

    for _ in range(n_trials):
        v = np.random.randn(4)
        v = v / np.linalg.norm(v)
        R = v[0] + v[1]*ce12 + v[2]*ce13 + v[3]*ce23
        p_cf = np.array([float((R*ce3*~R)(1).value[i]) for i in [1,2,3]])
        p_np = hopf_project(v)
        assert np.allclose(p_cf, p_np, atol=1e-10)
    return n_trials

try:
    _n_verified = verify_geometric_ops(20)
except ImportError:
    _n_verified = 0


# ================================================================
# v11: Cl(3,0) rotor-composition Berry phase for triangle holonomy
#
# For a closed triangular loop (p_a, p_b, p_c) on S², parallel transport
# of a tangent frame around the loop picks up a rotation whose angle
# equals the signed solid angle of the spherical triangle
# (Hannay/Berry theorem; the Hopf bundle has curvature 2-form equal to
# twice the S² area form, so the holonomy integral is the solid angle).
#
# We implement this via actual Cl(3,0) rotor composition:
#   R_pq = cos(θ/2) + sin(θ/2) B̂_pq
# where θ = arccos(p·q) and B̂_pq is the unit bivector dual to
# (p × q)/|p × q|. Composing R_ca · R_bc · R_ab yields a rotor whose
# axis is parallel to p_a (the base point of the loop) and whose
# rotation angle is the Berry phase — extractable from the dot product
# of the bivector part with the bivector dual of p_a.
#
# This is genuinely Clifford-algebraic: rotor composition in the
# even subalgebra Cl(3,0)⁺ (which IS the quaternion algebra via
# {1, e12, e13, e23}), using qmul for the geometric product.
# ================================================================

# Convention (matching hopf_project):
#   rotor [w, a, b, c] = w + a·e12 + b·e13 + c·e23
#   axis-to-bivector map: axis (nx, ny, nz) ↔ (nz, -ny, nx)
#   (so that axis e3 ↔ bivector e12, consistent with rotation of e3
#    by rotor sandwich R e3 ~R being the identity when R is a
#    rotation around e3)


def rotor_from_axis_angle(axis, angle):
    """Build a rotor for rotation by `angle` around unit `axis` in Cl(3,0)."""
    w = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    return np.array([w, s * axis[2], -s * axis[1], s * axis[0]])


def rotor_transport(p, q):
    """
    Cl(3,0) rotor for parallel transport from S² point p to q
    along the shortest geodesic.

    Returns [w, a, b, c] such that applying R v ~R rotates any vector
    v by the geodesic angle between p and q about the axis (p × q).
    When p and q are (anti)parallel the axis is undefined; we return
    the identity (for p = q) or a π rotation around an arbitrary
    perpendicular direction (for p = -q).
    """
    cos_th = float(np.dot(p, q))
    cos_th = max(-1.0, min(1.0, cos_th))
    theta = math.acos(cos_th)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    cr = np.cross(p, q)
    n = np.linalg.norm(cr)
    if n < 1e-12:
        # Antipodal: pick any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(p[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(p, perp)
        axis /= np.linalg.norm(axis)
    else:
        axis = cr / n
    return rotor_from_axis_angle(axis, theta)


def triangle_berry_clifford(p_a, p_b, p_c):
    """
    Compute the Berry phase of parallel transport around the spherical
    triangle (p_a → p_b → p_c → p_a) by composing three Cl(3,0)
    rotor-valued parallel transports and extracting the net rotation
    angle about the base point p_a.

    Pure Cl(3,0) computation: the only primitives used are qmul
    (the geometric product in the even subalgebra) and dot products
    to align the axis of the composed rotor with the base point.
    """
    R_ab = rotor_transport(p_a, p_b)
    R_bc = rotor_transport(p_b, p_c)
    R_ca = rotor_transport(p_c, p_a)
    # Compose right-to-left: apply R_ab first, then R_bc, then R_ca
    R_total = qmul(R_ca, qmul(R_bc, R_ab))

    # The composed rotor rotates about an axis parallel to p_a.
    # Its bivector part should be aligned with the bivector dual of p_a:
    #   axis p_a = (p_a_x, p_a_y, p_a_z) ↔ bivector (p_a_z, -p_a_y, p_a_x)
    biv_expected = np.array([p_a[2], -p_a[1], p_a[0]])
    biv_actual = R_total[1:]
    biv_mag = np.linalg.norm(biv_actual)
    sign = 1.0
    dot = float(np.dot(biv_actual, biv_expected))
    if dot < 0:
        sign = -1.0
    # angle = 2 * atan2(|bivector|, scalar), with sign from axis alignment
    return sign * 2.0 * math.atan2(biv_mag, R_total[0])


def verify_berry_phase(n_trials=50, rng=None):
    """
    Verify that the signed Euler-Eriksson solid-angle formula used in
    cell600.py agrees with an independent Cl(3,0) rotor-composition
    Berry phase calculation.

    Both routes compute the same invariant two ways:
      (a) Euler-Eriksson: 2·atan2(a·(b×c), 1 + a·b + b·c + c·a)
      (b) Cl(3,0) rotor composition of three geodesic transports.
    If they disagree the geometric substrate is broken.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(n_trials):
        # Sample three random points on S², avoiding degeneracies
        pts = rng.standard_normal((3, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        a, b, c = pts
        # Signed solid angle via Euler-Eriksson (matches cell600.py)
        triple = float(np.dot(a, np.cross(b, c)))
        denom = 1.0 + float(np.dot(a, b)) + float(np.dot(b, c)) + float(np.dot(c, a))
        berry_ee = math.atan2(triple, denom)
        # Signed Berry phase via Cl(3,0) rotor composition. Composing
        # three transports produces a rotation by 2·Berry about the
        # base point (the factor-of-two is the usual spin/rotor double
        # cover), so compare triangle_berry_clifford / 2 to berry_ee.
        berry_cf = triangle_berry_clifford(a, b, c) / 2.0
        err = abs(berry_ee - berry_cf)
        # Handle wrap-around (both values live in (-π, π))
        err = min(err, abs(err - 2 * math.pi), abs(err + 2 * math.pi))
        max_err = max(max_err, err)
    return max_err


try:
    _berry_err = verify_berry_phase(50)
    _berry_verified = _berry_err < 1e-8
except Exception:
    _berry_err = None
    _berry_verified = False
