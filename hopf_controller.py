# ================================================================
# Hopf Geometric Controller v5 — ADE-Structured Computation
#
# The ADE classification IS the architecture.
#
# The 9 eigenspaces of the 600-cell graph Laplacian are the 9 irreps
# of the binary icosahedral group 2I. Via the McKay correspondence,
# they map to the 9 nodes of the extended E₈ Dynkin diagram.
#
# Information flows along E₈ edges. Adjacent irreps exchange energy;
# non-adjacent ones can't. The Hopf projection is the nonlinearity.
# Parallel transport carries phase. Holonomy provides topology.
#
# Pipeline:
#   1. Input → 600-cell → spectral decomposition into 9 irreps
#   2. E₁ (4D): multi-stage Hopf pipeline (geometry IS computation)
#   3. C₁ (6D): Hopf chirality pipeline (6 vs 9 discrimination)
#   4. Irrep energies → McKay message passing on E₈ Dynkin diagram
#   5. Poincaré warp per node (nonlinearity)
#   6. Small readout from 36 geometric features
#
# ~441 params. The E₈ topology does the work the linear map was doing.
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
    """Berry phase + transport phase from closed S² triangle."""
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


def apply_givens(coeffs, angles, pairs):
    """Givens rotations within an eigenspace."""
    result = coeffs.copy()
    for angle, (i, j) in zip(angles, pairs):
        c, s = math.cos(angle), math.sin(angle)
        ri = c * result[i] - s * result[j]
        rj = s * result[i] + c * result[j]
        result[i], result[j] = ri, rj
    return result


def random_unit_quat():
    v = np.random.randn(4)
    return v / np.linalg.norm(v)


# ================================================================
# Mutation
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


def mutate_angle(a, rate, scale):
    return a + random.gauss(0, scale) if random.random() < rate else a


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
_NORTH = np.array([0.0, 0.0, 1.0])


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


def _get_pixel_kernel(input_size):
    global _PIXEL_KERNEL
    if _PIXEL_KERNEL is None or _PIXEL_KERNEL.shape[0] != input_size:
        _PIXEL_KERNEL = _build_pixel_kernel(input_size)
    return _PIXEL_KERNEL


# ================================================================
# HopfController v5 — ADE-Structured Geometric Computation
# ================================================================

class HopfController:
    """
    The ADE classification IS the architecture.

    9 eigenspaces → 9 nodes of extended E₈ Dynkin diagram (McKay).
    Information propagates along E₈ edges with learned couplings.
    Hopf projection is the nonlinearity. Berry phase carries topology.
    ~441 parameters.
    """

    GIVENS_PAIRS_E2 = [(0, 1), (2, 3), (4, 5), (6, 7)]
    N_HOPF_STAGES = 3   # E₁ Hopf stages
    N_C1_STAGES = 2      # C₁ chirality stages
    N_MCKAY_ROUNDS = 3   # message passing rounds on E₈

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

        # E₂ Givens
        self.givens2 = [0.0] * 4

        # McKay message passing: coupling weights on 8 E₈ edges × K rounds
        self.mckay_couplings = np.random.randn(
            self.N_MCKAY_ROUNDS, 8) * 0.1

        # Feature dimensions:
        #   9 McKay-processed node energies
        #   4 curl energies (Poincaré warped)
        #   13 E₁ Hopf features
        #   10 C₁ chirality features
        self.n_features = 9 + 4 + 13 + 10  # = 36

        xavier = math.sqrt(2.0 / (self.n_features + output_size))
        self.W_out = np.random.randn(output_size, self.n_features) * xavier
        self.b_out = np.zeros(output_size)

    def _hopf_pipeline(self, coeffs_4d, L_rotors, R_rotors, phases, n_stages):
        """Multi-stage Hopf on a 4D coefficient vector → geometric features."""
        norm = np.linalg.norm(coeffs_4d)
        q = coeffs_4d / norm if norm > 1e-10 else np.array([1., 0., 0., 0.])

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
            b, t = holonomy_triangle(s2_points[i].copy(),
                                     s2_points[i+1].copy(), _NORTH.copy())
            features.append(b)
            features.append(t)

        return np.array(features) * min(norm, 10.0)

    def _mckay_message_pass(self, node_energies):
        """
        Message passing on the extended E₈ Dynkin diagram.

        9 node energies propagate along 8 edges for K rounds.
        Adjacent irreps exchange energy with learned coupling weights.
        Poincaré warp after each round (the nonlinearity).

        The E₈ topology constrains information flow:
        node 4 (5D irrep, branching point) is the hub.
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
            # Poincaré warp per node (scalar version)
            energies = np.array([poincare_warp_scalar(e) for e in new_e])

        return energies

    def forward(self, inputs):
        """Forward pass: spectral decomposition → Hopf pipelines → McKay E₈ → readout."""
        geo = _get_geo()
        kernel = _get_pixel_kernel(self.input_size)

        # Project input onto 600-cell
        x = np.asarray(inputs, dtype=np.float64)
        if len(x) < self.input_size:
            x = np.pad(x, (0, self.input_size - len(x)))
        x = x[:self.input_size]
        f = x @ kernel  # (120,)

        # Spectral decomposition into 9 irreps + 4 curl modes
        sc = [es["vectors"].T @ f for es in geo["scalar_eigenspaces"]]
        df = geo["d0"] @ f
        cu = [es["vectors"].T @ df for es in geo["curl_eigenspaces"]]

        # --- E₁ Hopf pipeline (4D dipole = natural S³) ---
        e1_feat = self._hopf_pipeline(sc[1], self.e1_L, self.e1_R,
                                       self.e1_phases, self.N_HOPF_STAGES)

        # --- C₁ chirality pipeline (first 4 of 6 curl coefficients) ---
        c1_feat = self._hopf_pipeline(cu[0][:4], self.c1_L, self.c1_R,
                                       self.c1_phases, self.N_C1_STAGES)
        c1_remain = poincare_warp(cu[0][4:])
        c1_feat = np.concatenate([c1_feat, c1_remain])

        # --- E₂ Givens rotation (applied before energy extraction) ---
        sc[2] = apply_givens(sc[2], self.givens2, self.GIVENS_PAIRS_E2)

        # --- Irrep energies → E₈ nodes ---
        # Map eigenspace norms onto the 9 nodes of the E₈ Dynkin diagram
        eigenspace_to_e8 = geo["eigenspace_to_e8"]
        node_energies = np.zeros(9)
        for es_idx in range(len(sc)):
            e8_node = eigenspace_to_e8[es_idx]
            node_energies[e8_node] = np.linalg.norm(sc[es_idx])

        # --- McKay message passing on E₈ ---
        mckay_out = self._mckay_message_pass(node_energies)

        # --- Curl energies (Poincaré warped) ---
        curl_energies = np.array([poincare_warp_scalar(np.linalg.norm(cu[i]))
                                  for i in range(4)])

        # --- Concatenate all geometric features ---
        features = np.concatenate([
            mckay_out,      # 9 (E₈ node energies after message passing)
            curl_energies,  # 4
            e1_feat,        # 13 (Hopf S² coords + Berry + transport)
            c1_feat,        # 10 (chirality Hopf + remaining)
        ])  # total: 36

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
        self.givens2 = [mutate_angle(a, rate, scale) for a in self.givens2]
        # McKay couplings: scale-relative mutation
        for rnd in range(self.N_MCKAY_ROUNDS):
            for e in range(8):
                self.mckay_couplings[rnd, e] = mutate_scale(
                    self.mckay_couplings[rnd, e], rate, scale)
        mutate_flat_2d(self.W_out, rate, scale)
        mutate_flat(self.b_out, rate, scale)

    def clone(self):
        c = HopfController.__new__(HopfController)
        c.input_size = self.input_size
        c.hidden_size = self.hidden_size
        c.output_size = self.output_size
        c.N_HOPF_STAGES = self.N_HOPF_STAGES
        c.N_C1_STAGES = self.N_C1_STAGES
        c.N_MCKAY_ROUNDS = self.N_MCKAY_ROUNDS
        c.GIVENS_PAIRS_E2 = self.GIVENS_PAIRS_E2
        c.n_features = self.n_features
        c.e1_L = [q.copy() for q in self.e1_L]
        c.e1_R = [q.copy() for q in self.e1_R]
        c.e1_phases = self.e1_phases[:]
        c.c1_L = [q.copy() for q in self.c1_L]
        c.c1_R = [q.copy() for q in self.c1_R]
        c.c1_phases = self.c1_phases[:]
        c.givens2 = self.givens2[:]
        c.mckay_couplings = self.mckay_couplings.copy()
        c.W_out = self.W_out.copy()
        c.b_out = self.b_out.copy()
        return c

    def param_count(self):
        return (
            2 * self.N_HOPF_STAGES * 4 +      # E₁ rotors
            (self.N_HOPF_STAGES - 1) +          # E₁ phases
            2 * self.N_C1_STAGES * 4 +          # C₁ rotors
            (self.N_C1_STAGES - 1) +            # C₁ phases
            len(self.givens2) +                 # E₂ Givens
            self.mckay_couplings.size +         # McKay couplings
            self.W_out.size +                   # readout
            self.b_out.size
        )

    def effective_dof(self):
        return (
            2 * self.N_HOPF_STAGES * 3 +       # rotors: 3 DOF each on S³
            (self.N_HOPF_STAGES - 1) +
            2 * self.N_C1_STAGES * 3 +
            (self.N_C1_STAGES - 1) +
            len(self.givens2) +
            self.mckay_couplings.size +
            self.W_out.size +
            self.b_out.size
        )

    def to_dict(self):
        return {
            "type": "hopf", "version": 5,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "n_hopf_stages": self.N_HOPF_STAGES,
            "n_c1_stages": self.N_C1_STAGES,
            "n_mckay_rounds": self.N_MCKAY_ROUNDS,
            "e1_L": [q.tolist() for q in self.e1_L],
            "e1_R": [q.tolist() for q in self.e1_R],
            "e1_phases": self.e1_phases,
            "c1_L": [q.tolist() for q in self.c1_L],
            "c1_R": [q.tolist() for q in self.c1_R],
            "c1_phases": self.c1_phases,
            "givens2": self.givens2,
            "mckay_couplings": self.mckay_couplings.tolist(),
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
        c.N_MCKAY_ROUNDS = d.get("n_mckay_rounds", 3)
        c.GIVENS_PAIRS_E2 = [(0, 1), (2, 3), (4, 5), (6, 7)]
        c.n_features = 36
        c.e1_L = [np.array(q) for q in d["e1_L"]]
        c.e1_R = [np.array(q) for q in d["e1_R"]]
        c.e1_phases = d["e1_phases"]
        c.c1_L = [np.array(q) for q in d.get("c1_L", [[1,0,0,0]]*c.N_C1_STAGES)]
        c.c1_R = [np.array(q) for q in d.get("c1_R", [[1,0,0,0]]*c.N_C1_STAGES)]
        c.c1_phases = d.get("c1_phases", [0.0]*(c.N_C1_STAGES - 1))
        c.givens2 = d["givens2"]
        c.mckay_couplings = np.array(d["mckay_couplings"], dtype=np.float64)
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
