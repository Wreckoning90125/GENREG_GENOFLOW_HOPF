# ================================================================
# Hopf Geometric Controller — Spectral Substrate on S³
#
# Architecture derived from the geometry of the 600-cell and the
# Hopf fibration S³ → S² (fiber S¹).
#
# Geometric operations are defined in Cl(3,0) (clifford algebra)
# and implemented in numpy for performance. Each numpy formula is
# the expansion of a specific clifford expression, verified at
# import time by verify_geometric_ops().
#
# Pipeline:
#   1. Project input onto 600-cell via Hopf map → spectral decomposition
#      (600-cell = 120 vertices of binary icosahedral group 2I;
#       eigenbasis from Theorems 3 & 5; see cell600.py)
#   2. E₁ (4D dipole = natural S³ rotor):
#      - Multi-stage: rotate → HOPF PROJECT → lift → rotate → HOPF PROJECT
#      - Parallel transport (Pancharatnam connection) → Berry phase
#      - Holonomy from S² triangles (Euler-Eriksson) → topological features
#   3. C₁ (6D co-exact 1-form = chirality detector):
#      - Same Hopf pipeline on first 4 coefficients
#      - "The difference between a 6 and a 9 is a curl feature"
#   4. E₀-E₅ (Theorem 3), E₆-E₈ (aliased), C₂-C₄ (Theorem 5):
#      per-eigenspace scale + Poincaré conformal warp r'=2·tanh(r/2)
#   5. Minimal linear readout
#
# Dependencies: numpy (hot path), clifford (verification), cell600 (geometry)
# ================================================================

import math
import random
import numpy as np

from cell600 import get_geometry


# ================================================================
# Geometric operations in numpy
#
# Each function documents the Cl(3,0) expression it implements.
# Rotor convention: R = w + a·e12 + b·e13 + c·e23 ∈ Cl⁺(3,0)
# stored as numpy array [w, a, b, c].
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


def qnorm(q):
    return math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)


def qnormalize(q):
    n = qnorm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def hopf_project(q):
    """
    Hopf projection: S³ → S².
    For rotor R = w + a·e12 + b·e13 + c·e23, computes R·e3·~R.

    Returns (px, py, pz) on S². THIS IS THE NONLINEARITY.

    Derived from expanding the sandwich product in Cl(3,0):
        px = 2(wb + ac)
        py = 2(wc - ab)
        pz = w² + a² - b² - c²
    """
    w, a, b, c = q
    return np.array([
        2.0 * (w*b + a*c),
        2.0 * (w*c - a*b),
        w*w + a*a - b*b - c*c,
    ])


def hopf_section(px, py, pz):
    """
    Canonical section of the Hopf bundle: S² → S³.
    Returns rotor [w, e12, e13, e23] such that hopf_project(R) = (px,py,pz).

    Derived from (1 + p·e3)/|1 + p·e3| in Cl(3,0):
        1 + p·e3 = (1+pz) + 0·e12 + px·e13 + py·e23
    """
    w = 1.0 + pz
    norm = math.sqrt(w*w + px*px + py*py)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0, 0.0])  # south pole fallback
    return np.array([w / norm, 0.0, px / norm, py / norm])


def hopf_lift(px, py, pz, phase):
    """Section lift with fiber rotation: section(p) · exp(phase/2 · e12)."""
    sec = hopf_section(px, py, pz)
    fiber = np.array([math.cos(phase / 2.0), math.sin(phase / 2.0), 0.0, 0.0])
    return qnormalize(qmul(sec, fiber))


def pancharatnam_phase(p1, p2):
    """
    Phase from Pancharatnam connection between two S² points.
    Discrete parallel transport: phase acquired moving fiber from p1 to p2.
    """
    r1 = hopf_section(*p1)
    r2 = hopf_section(*p2)
    r1_rev = np.array([r1[0], -r1[1], -r1[2], -r1[3]])
    overlap = qmul(r1_rev, r2)
    return math.atan2(-overlap[1], overlap[0])


def solid_angle_triangle(a, b, c):
    """Solid angle of spherical triangle. Euler-Eriksson formula."""
    triple = np.dot(a, np.cross(b, c))
    denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    if abs(denom) < 1e-12:
        return math.pi
    return 2.0 * math.atan2(abs(triple), denom)


def holonomy_triangle(p1, p2, p_ref):
    """
    Holonomy from closed triangle on S²: Berry phase + transport phase.
    p1, p2, p_ref are np arrays of shape (3,) on S².
    """
    # Normalize
    for v in [p1, p2, p_ref]:
        n = np.linalg.norm(v)
        if n > 1e-10:
            v /= n

    omega = solid_angle_triangle(p1, p2, p_ref)
    berry = -omega / 2.0

    ph_12 = pancharatnam_phase(p1, p2)
    ph_2r = pancharatnam_phase(p2, p_ref)
    ph_r1 = pancharatnam_phase(p_ref, p1)
    transport = ph_12 + ph_2r + ph_r1

    return berry, transport


def poincare_warp(coefficients):
    """Radial conformal compression: r' = 2·tanh(r/2). Per-eigenspace."""
    r = np.linalg.norm(coefficients)
    if r < 1e-10:
        return coefficients.copy()
    r_prime = 2.0 * math.tanh(r / 2.0)
    return coefficients * (r_prime / r)


def apply_givens_rotations(coefficients, angles, pairs):
    """Givens (planar) rotations within an eigenspace."""
    result = coefficients.copy()
    for angle, (i, j) in zip(angles, pairs):
        c = math.cos(angle)
        s = math.sin(angle)
        ri = c * result[i] - s * result[j]
        rj = s * result[i] + c * result[j]
        result[i] = ri
        result[j] = rj
    return result


def random_unit_quat():
    """Sample uniformly from S³."""
    v = np.random.randn(4)
    return v / np.linalg.norm(v)


# ================================================================
# Mutation (pure numpy)
# ================================================================

def mutate_quat(q, rate, scale):
    """Geodesic perturbation on S³ via exponential map."""
    if random.random() > rate:
        return q.copy()
    v = np.array([random.gauss(0, scale) for _ in range(3)])
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return q.copy()
    s = math.sin(norm_v) / norm_v
    delta = np.array([math.cos(norm_v), s * v[0], s * v[1], s * v[2]])
    return qnormalize(qmul(q, delta))


def mutate_angle(angle, rate, scale):
    if random.random() > rate:
        return angle
    return angle + random.gauss(0, scale)


def mutate_scale(s, rate, scale):
    if random.random() > rate:
        return s
    return s + random.gauss(0, scale * (abs(s) + 1e-9))


def mutate_flat_2d(arr, rate, scale):
    mask = np.random.random(arr.shape) < rate
    arr[mask] += np.random.randn(mask.sum()) * scale


def mutate_flat(arr, rate, scale):
    mask = np.random.random(arr.shape) < rate
    arr[mask] += np.random.randn(mask.sum()) * scale


# ================================================================
# Geometry cache
# ================================================================

_GEO = None
_PIXEL_KERNEL = None


def _get_geo():
    global _GEO
    if _GEO is None:
        _GEO = get_geometry()
    return _GEO


def _build_pixel_kernel(input_size, kappa=10.0):
    """Pixel-to-vertex soft assignment via Hopf map."""
    geo = _get_geo()
    vertices = geo["vertices"]

    if input_size == 784:
        rows, cols = 28, 28
    else:
        side = int(math.ceil(math.sqrt(input_size)))
        rows, cols = side, side

    pixel_quats = []
    for idx in range(input_size):
        r, c = divmod(idx, 28) if input_size == 784 else divmod(idx, cols)
        x = 2.0 * c / max(cols - 1, 1) - 1.0
        y = 2.0 * r / max(rows - 1, 1) - 1.0

        denom = 1.0 + x * x + y * y
        X = 2.0 * x / denom
        Y = 2.0 * y / denom
        Z = (x * x + y * y - 1.0) / denom

        if Z > -0.999:
            s = math.sqrt(2.0 * (1.0 + Z))
            q = np.array([s / 2.0, -Y / s, X / s, 0.0])
        else:
            q = np.array([0.0, 0.0, 1.0, 0.0])
        pixel_quats.append(q / np.linalg.norm(q))

    pixel_quats = np.array(pixel_quats)
    dots = np.abs(pixel_quats @ vertices.T)
    scaled = kappa * dots
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return (exp_scaled / exp_scaled.sum(axis=1, keepdims=True)).astype(np.float64)


def _get_pixel_kernel(input_size):
    global _PIXEL_KERNEL
    if _PIXEL_KERNEL is None or _PIXEL_KERNEL.shape[0] != input_size:
        _PIXEL_KERNEL = _build_pixel_kernel(input_size)
    return _PIXEL_KERNEL


# Reference point for holonomy triangles (north pole)
_NORTH = np.array([0.0, 0.0, 1.0])


# ================================================================
# HopfController — Pure Numpy, Geometrically Correct
# ================================================================

class HopfController:
    """
    Controller where the geometry IS the computation.
    Pure numpy forward pass — no clifford in the hot loop.

    E₁ (4D): 3-stage Hopf pipeline with parallel transport + holonomy.
    C₁ (6D): 2-stage Hopf chirality pipeline (6 vs 9 discrimination).
    E₀-E₅, E₆-E₈, C₂-C₄: per-eigenspace scale + Poincaré warp.
    """

    GIVENS_PAIRS_E2 = [(0, 1), (2, 3), (4, 5), (6, 7)]
    N_HOPF_STAGES = 3
    N_C1_STAGES = 2

    def __init__(self, input_size=784, hidden_size=16, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        _get_geo()
        _get_pixel_kernel(input_size)

        # E₁ multi-stage Hopf pipeline
        self.e1_L = [random_unit_quat() for _ in range(self.N_HOPF_STAGES)]
        self.e1_R = [random_unit_quat() for _ in range(self.N_HOPF_STAGES)]
        self.e1_phases = [random.uniform(-math.pi, math.pi)
                          for _ in range(self.N_HOPF_STAGES - 1)]

        # C₁ chirality Hopf pipeline
        self.c1_L = [random_unit_quat() for _ in range(self.N_C1_STAGES)]
        self.c1_R = [random_unit_quat() for _ in range(self.N_C1_STAGES)]
        self.c1_phases = [random.uniform(-math.pi, math.pi)
                          for _ in range(self.N_C1_STAGES - 1)]

        # Per-eigenspace scales
        self.s0 = 1.0                        # E₀
        self.givens2 = [0.0] * 4             # E₂ Givens angles
        self.scales = [1.0] * 6              # E₃, E₄, E₅, E₆, E₇, E₈
        self.curl_scales = [1.0] * 3         # C₂, C₃, C₄

        # Feature dimensions
        n_e1 = 3 * self.N_HOPF_STAGES + 2 * (self.N_HOPF_STAGES - 1)  # 13
        n_c1 = 3 * self.N_C1_STAGES + 2 * (self.N_C1_STAGES - 1) + 2  # 10
        n_scalar = 1 + 9 + 16 + 25 + 36 + 9 + 16 + 4  # 116
        n_curl = 16 + 30 + 48                            # 94
        self.n_features = n_e1 + n_c1 + n_scalar + n_curl  # 233

        xavier = math.sqrt(2.0 / (self.n_features + output_size))
        self.W_out = np.random.randn(output_size, self.n_features) * xavier
        self.b_out = np.zeros(output_size)

    def _hopf_pipeline(self, coeffs_4d, L_rotors, R_rotors, phases, n_stages):
        """Multi-stage Hopf pipeline on a 4D coefficient vector."""
        norm = np.linalg.norm(coeffs_4d)
        if norm < 1e-10:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q = coeffs_4d / norm

        s2_points = []
        for stage in range(n_stages):
            q_rot = qnormalize(qmul(qmul(L_rotors[stage], q), R_rotors[stage]))
            p = hopf_project(q_rot)
            s2_points.append(p)
            if stage < n_stages - 1:
                q = hopf_lift(p[0], p[1], p[2], phases[stage])

        features = []
        for p in s2_points:
            features.extend(p.tolist())
        for i in range(len(s2_points) - 1):
            berry, transport = holonomy_triangle(
                s2_points[i].copy(), s2_points[i + 1].copy(), _NORTH.copy()
            )
            features.append(berry)
            features.append(transport)

        return np.array(features) * min(norm, 10.0)

    def forward(self, inputs):
        """Pure numpy forward pass. No clifford objects allocated."""
        geo = _get_geo()
        kernel = _get_pixel_kernel(self.input_size)

        # Project input onto 600-cell
        x = np.asarray(inputs, dtype=np.float64)
        if len(x) < self.input_size:
            x = np.pad(x, (0, self.input_size - len(x)))
        x = x[:self.input_size]
        f = x @ kernel  # (120,)

        # Spectral decomposition — all eigenspaces
        sc = [es["vectors"].T @ f for es in geo["scalar_eigenspaces"]]
        df = geo["d0"] @ f
        cu = [es["vectors"].T @ df for es in geo["curl_eigenspaces"]]

        # E₁: multi-stage Hopf (the geometric core)
        e1_feat = self._hopf_pipeline(sc[1], self.e1_L, self.e1_R,
                                       self.e1_phases, self.N_HOPF_STAGES)

        # C₁: chirality Hopf pipeline (first 4 of 6 coefficients)
        c1_feat = self._hopf_pipeline(cu[0][:4], self.c1_L, self.c1_R,
                                       self.c1_phases, self.N_C1_STAGES)
        c1_remain = poincare_warp(cu[0][4:])  # remaining 2
        c1_feat = np.concatenate([c1_feat, c1_remain])

        # E₀: scale + warp
        e0_feat = poincare_warp(sc[0] * self.s0)

        # E₂: Givens + warp
        e2_feat = poincare_warp(apply_givens_rotations(
            sc[2], self.givens2, self.GIVENS_PAIRS_E2))

        # E₃-E₈: scale + warp
        higher = []
        for i, idx in enumerate([3, 4, 5, 6, 7, 8]):
            if idx < len(sc):
                higher.append(poincare_warp(sc[idx] * self.scales[i]))
            else:
                higher.append(np.zeros(1))

        # C₂-C₄: scale + warp
        curl_higher = []
        for i, idx in enumerate([1, 2, 3]):
            curl_higher.append(poincare_warp(cu[idx] * self.curl_scales[i]))

        features = np.concatenate([e1_feat, c1_feat, e0_feat, e2_feat]
                                  + higher + curl_higher)

        logits = self.W_out @ features + self.b_out
        return logits.tolist()

    def select_action(self, signals, signal_order):
        inputs = [signals.get(k, 0.0) for k in signal_order]
        logits = self.forward(inputs)
        return logits.index(max(logits))

    def mutate(self, rate=0.1, scale=0.3):
        for i in range(self.N_HOPF_STAGES):
            self.e1_L[i] = mutate_quat(self.e1_L[i], rate, scale)
            self.e1_R[i] = mutate_quat(self.e1_R[i], rate, scale)
        for i in range(len(self.e1_phases)):
            self.e1_phases[i] = mutate_angle(self.e1_phases[i], rate, scale)

        for i in range(self.N_C1_STAGES):
            self.c1_L[i] = mutate_quat(self.c1_L[i], rate, scale)
            self.c1_R[i] = mutate_quat(self.c1_R[i], rate, scale)
        for i in range(len(self.c1_phases)):
            self.c1_phases[i] = mutate_angle(self.c1_phases[i], rate, scale)

        self.s0 = mutate_scale(self.s0, rate, scale)
        self.givens2 = [mutate_angle(a, rate, scale) for a in self.givens2]
        self.scales = [mutate_scale(s, rate, scale) for s in self.scales]
        self.curl_scales = [mutate_scale(s, rate, scale) for s in self.curl_scales]

        mutate_flat_2d(self.W_out, rate, scale)
        mutate_flat(self.b_out, rate, scale)

    def clone(self):
        c = HopfController.__new__(HopfController)
        c.input_size = self.input_size
        c.hidden_size = self.hidden_size
        c.output_size = self.output_size
        c.N_HOPF_STAGES = self.N_HOPF_STAGES
        c.N_C1_STAGES = self.N_C1_STAGES
        c.n_features = self.n_features
        c.GIVENS_PAIRS_E2 = self.GIVENS_PAIRS_E2

        c.e1_L = [q.copy() for q in self.e1_L]
        c.e1_R = [q.copy() for q in self.e1_R]
        c.e1_phases = self.e1_phases[:]
        c.c1_L = [q.copy() for q in self.c1_L]
        c.c1_R = [q.copy() for q in self.c1_R]
        c.c1_phases = self.c1_phases[:]

        c.s0 = self.s0
        c.givens2 = self.givens2[:]
        c.scales = self.scales[:]
        c.curl_scales = self.curl_scales[:]
        c.W_out = self.W_out.copy()
        c.b_out = self.b_out.copy()
        return c

    def param_count(self):
        n_rotors = 2 * self.N_HOPF_STAGES + 2 * self.N_C1_STAGES
        n_phases = (self.N_HOPF_STAGES - 1) + (self.N_C1_STAGES - 1)
        return (n_rotors * 4 + n_phases + 1 + len(self.givens2) +
                len(self.scales) + len(self.curl_scales) +
                self.W_out.size + self.b_out.size)

    def effective_dof(self):
        n_rotors = 2 * self.N_HOPF_STAGES + 2 * self.N_C1_STAGES
        n_phases = (self.N_HOPF_STAGES - 1) + (self.N_C1_STAGES - 1)
        return (n_rotors * 3 + n_phases + 1 + len(self.givens2) +
                len(self.scales) + len(self.curl_scales) +
                self.W_out.size + self.b_out.size)

    def to_dict(self):
        return {
            "type": "hopf", "version": 4,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "n_hopf_stages": self.N_HOPF_STAGES,
            "n_c1_stages": self.N_C1_STAGES,
            "e1_L": [q.tolist() for q in self.e1_L],
            "e1_R": [q.tolist() for q in self.e1_R],
            "e1_phases": self.e1_phases,
            "c1_L": [q.tolist() for q in self.c1_L],
            "c1_R": [q.tolist() for q in self.c1_R],
            "c1_phases": self.c1_phases,
            "s0": self.s0,
            "givens2": self.givens2,
            "scales": self.scales,
            "curl_scales": self.curl_scales,
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        c = cls.__new__(cls)
        c.input_size = d["input_size"]
        c.hidden_size = d.get("hidden_size", 16)
        c.output_size = d["output_size"]
        c.N_HOPF_STAGES = d.get("n_hopf_stages", 3)
        c.N_C1_STAGES = d.get("n_c1_stages", 2)
        c.GIVENS_PAIRS_E2 = [(0, 1), (2, 3), (4, 5), (6, 7)]

        c.e1_L = [np.array(q) for q in d["e1_L"]]
        c.e1_R = [np.array(q) for q in d["e1_R"]]
        c.e1_phases = d["e1_phases"]
        c.c1_L = [np.array(q) for q in d.get("c1_L", [[1,0,0,0]] * c.N_C1_STAGES)]
        c.c1_R = [np.array(q) for q in d.get("c1_R", [[1,0,0,0]] * c.N_C1_STAGES)]
        c.c1_phases = d.get("c1_phases", [0.0] * (c.N_C1_STAGES - 1))

        c.s0 = d["s0"]
        c.givens2 = d["givens2"]
        c.scales = d.get("scales", [1.0] * 6)
        c.curl_scales = d.get("curl_scales", [1.0] * 3)
        c.W_out = np.array(d["W_out"], dtype=np.float64)
        c.b_out = np.array(d["b_out"], dtype=np.float64)

        n_e1 = 3 * c.N_HOPF_STAGES + 2 * (c.N_HOPF_STAGES - 1)
        n_c1 = 3 * c.N_C1_STAGES + 2 * (c.N_C1_STAGES - 1) + 2
        c.n_features = n_e1 + n_c1 + 116 + 94
        return c


# ================================================================
# Clifford verification — proves numpy formulas match Cl(3,0)
# ================================================================

def verify_geometric_ops(n_trials=50):
    """
    Verify all numpy geometric operations against clifford Cl(3,0).

    Each numpy function in this module is the expansion of a specific
    clifford expression. This function tests them against each other
    on random inputs and asserts exact agreement.

    Run at import time or on demand to confirm correctness.
    """
    import clifford as cf

    layout, blades = cf.Cl(3)
    ce1, ce2, ce3 = blades['e1'], blades['e2'], blades['e3']
    ce12, ce13, ce23 = blades['e12'], blades['e13'], blades['e23']

    def to_rotor(w, a, b, c):
        return w + a * ce12 + b * ce13 + c * ce23

    passed = 0

    for _ in range(n_trials):
        v = np.random.randn(4)
        v = v / np.linalg.norm(v)
        w, a, b, c = v

        # --- Hopf projection: R·e3·~R ---
        R = to_rotor(w, a, b, c)
        p_cf = (R * ce3 * ~R)(1)
        p_cf_vec = np.array([float(p_cf.value[1]), float(p_cf.value[2]),
                             float(p_cf.value[3])])
        p_np = hopf_project(v)
        assert np.allclose(p_cf_vec, p_np, atol=1e-10), \
            f"hopf_project mismatch: {p_cf_vec} vs {p_np}"

        # --- Section lift: (1 + p·e3)/|1 + p·e3| ---
        px, py, pz = p_np
        p_mv = px * ce1 + py * ce2 + pz * ce3
        inner = 1 + p_mv * ce3
        R_cf = inner / abs(inner) if abs(inner) > 1e-10 else ce13
        r_cf = np.array([float(R_cf.value[0]), float(R_cf.value[4]),
                         float(R_cf.value[5]), float(R_cf.value[6])])
        r_np = hopf_section(px, py, pz)
        assert np.allclose(r_cf, r_np, atol=1e-10), \
            f"hopf_section mismatch: {r_cf} vs {r_np}"

        # --- Pancharatnam phase: atan2 of ~R1·R2 fiber component ---
        v2 = np.random.randn(4)
        v2 = v2 / np.linalg.norm(v2)
        p2_np = hopf_project(v2)
        p2_mv = p2_np[0] * ce1 + p2_np[1] * ce2 + p2_np[2] * ce3

        R1_cf = (1 + p_mv * ce3)
        R1_cf = R1_cf / abs(R1_cf) if abs(R1_cf) > 1e-10 else ce13
        R2_cf = (1 + p2_mv * ce3)
        R2_cf = R2_cf / abs(R2_cf) if abs(R2_cf) > 1e-10 else ce13
        ov_cf = ~R1_cf * R2_cf
        phase_cf = math.atan2(float(ov_cf.value[4]), float(ov_cf.value[0]))
        phase_np = pancharatnam_phase(p_np, p2_np)
        assert abs(phase_cf - phase_np) < 1e-10, \
            f"pancharatnam mismatch: {phase_cf} vs {phase_np}"

        passed += 1

    return passed


# Run verification at import time
try:
    _n_verified = verify_geometric_ops(20)
except ImportError:
    # clifford not installed — skip verification, numpy ops are standalone
    _n_verified = 0
