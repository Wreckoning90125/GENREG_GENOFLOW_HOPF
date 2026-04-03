# ================================================================
# Hopf Geometric Controller v3 — Multi-Stage Geometric Computation
#
# The geometry IS the computation. Not preprocessing for a linear model.
#
# Architecture:
#   1. Project input onto 600-cell via Hopf map → spectral decomposition
#   2. E₁ (4D dipole = natural S³ rotor):
#      - Stage 1: learned rotation → HOPF PROJECT → S² point p₁
#      - Lift back to S³ with learned phase
#      - Stage 2: learned rotation → HOPF PROJECT → S² point p₂
#      - Parallel transport p₁→p₂ → accumulated Berry phase
#      - Holonomy from triangle (p₁, p₂, reference) → topological phase
#   3. E₀, E₂, C₁: eigenspace-respecting operations + Poincaré warp
#   4. Minimal readout from geometric features
#
# The Hopf projection appears BETWEEN computational stages.
# Parallel transport CARRIES information along paths on S².
# Holonomy PROVIDES topological features.
# The clifford library handles ALL geometric algebra operations.
# ================================================================

import math
import random
import numpy as np
import clifford as cf

from cell600 import get_geometry

# ================================================================
# Clifford algebra setup: Cl(3,0)
#
# Even subalgebra = quaternions = rotors on S³
# Rotor R = w + x*e12 + y*e13 + z*e23
# Hopf projection: R * e3 * ~R → grade-1 vector on S²
# Section lift: (1 + p*e3) / |1 + p*e3|
# Fiber rotation: right-multiply by exp(θ/2 * e12)
# ================================================================

_layout, _blades = cf.Cl(3)
e1 = _blades['e1']
e2 = _blades['e2']
e3 = _blades['e3']
e12 = _blades['e12']
e13 = _blades['e13']
e23 = _blades['e23']

# Blade index map for value array extraction
_IDX_SCALAR = 0
_IDX_E1 = 1
_IDX_E2 = 2
_IDX_E3 = 3
_IDX_E12 = 4
_IDX_E13 = 5
_IDX_E23 = 6


# ================================================================
# Clifford geometric operations
# ================================================================

def rotor_from_quaternion(w, x, y, z):
    """Quaternion [w,x,y,z] → Cl(3) rotor (even-grade multivector)."""
    return w + x * e12 + y * e13 + z * e23


def rotor_to_array(R):
    """Extract [w, x, y, z] from rotor."""
    return np.array([
        float(R.value[_IDX_SCALAR]),
        float(R.value[_IDX_E12]),
        float(R.value[_IDX_E13]),
        float(R.value[_IDX_E23]),
    ])


def random_rotor():
    """Sample uniformly from S³."""
    v = np.random.randn(4)
    v = v / np.linalg.norm(v)
    return rotor_from_quaternion(v[0], v[1], v[2], v[3])


def normalize_rotor(R):
    """Project rotor back to unit norm (numerical safety)."""
    n = abs(R)
    if n < 1e-12:
        return rotor_from_quaternion(1, 0, 0, 0)
    return R / n


def hopf_project(R):
    """
    Hopf projection: S³ → S² via sandwich product.
    R * e3 * ~R → grade-1 multivector on S².

    THIS IS THE NONLINEARITY. It appears between computational stages.
    """
    return (R * e3 * ~R)(1)


def hopf_section(p_mv):
    """
    Canonical section of the Hopf bundle: S² → S³.
    Given point p on S², returns rotor R such that R*e3*~R = p.
    """
    inner = 1 + p_mv * e3
    if abs(inner) < 1e-10:
        return e1 * (1.0 / abs(e1))  # south pole section
    return inner / abs(inner)


def hopf_lift(p_mv, phase):
    """
    Section lift with fiber rotation.
    S² point + phase → rotor on S³.
    """
    R_sec = hopf_section(p_mv)
    fiber_rot = math.cos(phase / 2) + math.sin(phase / 2) * e12
    return R_sec * fiber_rot


def s2_extract(p_mv):
    """Extract R³ coordinates from grade-1 multivector."""
    return np.array([
        float(p_mv.value[_IDX_E1]),
        float(p_mv.value[_IDX_E2]),
        float(p_mv.value[_IDX_E3]),
    ])


def pancharatnam_phase(p1_mv, p2_mv):
    """
    Phase from Pancharatnam connection between two S² points.
    This is the discrete parallel transport: the phase acquired
    by moving a fiber element from p1 to p2 along the geodesic.
    """
    R1 = hopf_section(p1_mv)
    R2 = hopf_section(p2_mv)
    overlap = ~R1 * R2
    w = float(overlap.value[_IDX_SCALAR])
    b12 = float(overlap.value[_IDX_E12])
    return math.atan2(b12, w)


def solid_angle_triangle(a, b, c):
    """
    Solid angle of spherical triangle with unit vector vertices.
    Euler-Eriksson formula.
    """
    triple = np.dot(a, np.cross(b, c))
    denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    if abs(denom) < 1e-12:
        return math.pi
    return 2.0 * math.atan2(abs(triple), denom)


def holonomy_triangle(p1_mv, p2_mv, p_ref_mv):
    """
    Holonomy from closed triangle on S²: Berry phase = -Ω/2
    where Ω is the solid angle of the triangle (p1, p2, p_ref).

    Also computes Pancharatnam transport phase along the path.
    Returns (berry_phase, transport_phase).
    """
    v1 = s2_extract(p1_mv)
    v2 = s2_extract(p2_mv)
    v_ref = s2_extract(p_ref_mv)

    # Normalize
    for v in [v1, v2, v_ref]:
        n = np.linalg.norm(v)
        if n > 1e-10:
            v /= n

    omega = solid_angle_triangle(v1, v2, v_ref)
    berry = -omega / 2.0

    # Pancharatnam transport phase along each leg
    ph_12 = pancharatnam_phase(p1_mv, p2_mv)
    ph_2r = pancharatnam_phase(p2_mv, p_ref_mv)
    ph_r1 = pancharatnam_phase(p_ref_mv, p1_mv)
    transport = ph_12 + ph_2r + ph_r1

    return berry, transport


def poincare_warp(coefficients):
    """
    Radial conformal compression: r' = 2·tanh(r/2).
    Applied to coefficient vector AS A WHOLE, not per-component.
    """
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


# ================================================================
# Mutation operations (all via clifford)
# ================================================================

def mutate_rotor(R, rate, scale):
    """Geodesic perturbation on S³ via exponential map in Cl(3)."""
    if random.random() > rate:
        return R
    # Random tangent bivector
    v = [random.gauss(0, scale) for _ in range(3)]
    # Bivector = v[0]*e12 + v[1]*e13 + v[2]*e23
    B = v[0] * e12 + v[1] * e13 + v[2] * e23
    # Exponential map: exp(B) = cos(|B|) + sin(|B|)/|B| * B
    norm_B = abs(B)
    if norm_B < 1e-12:
        return R
    delta = math.cos(norm_B) + (math.sin(norm_B) / norm_B) * B
    result = R * delta
    return normalize_rotor(result)


def mutate_angle(angle, rate, scale):
    """Additive Gaussian on an angle."""
    if random.random() > rate:
        return angle
    return angle + random.gauss(0, scale)


def mutate_scale(s, rate, scale):
    """Scale-relative mutation."""
    if random.random() > rate:
        return s
    return s + random.gauss(0, scale * (abs(s) + 1e-9))


def mutate_flat_2d(arr, rate, scale):
    """Standard GENREG mutation on 2D array."""
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if random.random() < rate:
                arr[i, j] += random.gauss(0, scale)


def mutate_flat(arr, rate, scale):
    """Standard GENREG mutation on 1D array."""
    for i in range(len(arr)):
        if random.random() < rate:
            arr[i] += random.gauss(0, scale)


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
        r, c = divmod(idx, cols) if input_size != 784 else divmod(idx, 28)

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

        q = q / np.linalg.norm(q)
        pixel_quats.append(q)

    pixel_quats = np.array(pixel_quats)
    dots = np.abs(pixel_quats @ vertices.T)
    scaled = kappa * dots
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    kernel = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    return kernel.astype(np.float64)


def _get_pixel_kernel(input_size):
    global _PIXEL_KERNEL
    if _PIXEL_KERNEL is None or _PIXEL_KERNEL.shape[0] != input_size:
        _PIXEL_KERNEL = _build_pixel_kernel(input_size)
    return _PIXEL_KERNEL


# ================================================================
# HopfController v3 — Multi-Stage Geometric Computation
# ================================================================

class HopfController:
    """
    Controller where the geometry IS the computation.

    The E₁ eigenspace (4D = natural S³) goes through a multi-stage
    Hopf pipeline where:
    - Hopf projection is the nonlinearity BETWEEN stages
    - Section lifting with learned phase carries information back to S³
    - Parallel transport accumulates Berry phase along the S² path
    - Holonomy extracts topological features from closed loops

    Other eigenspaces get eigenspace-respecting operations
    (Givens rotations, scales, Poincaré warp).
    """

    GIVENS_PAIRS_E2 = [(0, 1), (2, 3), (4, 5), (6, 7)]
    N_HOPF_STAGES = 3   # Hopf project→lift cycles for E₁
    N_C1_STAGES = 2      # Hopf stages for C₁ chirality detection

    def __init__(self, input_size=784, hidden_size=16, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        _get_geo()
        _get_pixel_kernel(input_size)

        # --- E₁ multi-stage Hopf pipeline (the geometric core) ---
        self.e1_left_rotors = [random_rotor() for _ in range(self.N_HOPF_STAGES)]
        self.e1_right_rotors = [random_rotor() for _ in range(self.N_HOPF_STAGES)]
        self.e1_lift_phases = [random.uniform(-math.pi, math.pi)
                               for _ in range(self.N_HOPF_STAGES - 1)]

        # --- C₁ chirality Hopf pipeline ---
        # "The difference between a 6 and a 9 is a curl feature."
        # C₁ is 6D. First 4 coefficients → quaternion → Hopf stages.
        # Remaining 2 get Poincaré warped separately.
        self.c1_left_rotors = [random_rotor() for _ in range(self.N_C1_STAGES)]
        self.c1_right_rotors = [random_rotor() for _ in range(self.N_C1_STAGES)]
        self.c1_lift_phases = [random.uniform(-math.pi, math.pi)
                               for _ in range(self.N_C1_STAGES - 1)]

        # --- E₀ (1D): scale ---
        self.s0 = 1.0

        # --- E₂ (9D): Givens rotations ---
        self.givens2 = [0.0] * len(self.GIVENS_PAIRS_E2)

        # --- E₃ (16D), E₄ (25D), E₅ (36D): per-eigenspace learned scales ---
        self.scale_e3 = 1.0
        self.scale_e4 = 1.0
        self.scale_e5 = 1.0

        # --- E₆ (9D), E₇ (16D), E₈ (4D): aliased modes ---
        # Nyquist folding of high-frequency content. Multiplicities mirror
        # lower eigenspaces. Previously discarded; now included.
        self.scale_e6 = 1.0
        self.scale_e7 = 1.0
        self.scale_e8 = 1.0

        # --- C₂ (16D), C₃ (30D), C₄ (48D): curl scales ---
        self.curl_scales = [1.0, 1.0, 1.0]  # C₂, C₃, C₄ only (C₁ gets Hopf)

        # --- Readout ---
        # E₁ pipeline: 3*N_STAGES + 2*(N_STAGES-1) = 13
        n_e1_features = 3 * self.N_HOPF_STAGES + 2 * (self.N_HOPF_STAGES - 1)
        # C₁ chirality pipeline: 3*N_C1_STAGES + 2*(N_C1_STAGES-1) + 2 (remaining)
        n_c1_features = 3 * self.N_C1_STAGES + 2 * (self.N_C1_STAGES - 1) + 2
        # Scalar eigenspaces: E₀(1) + E₂(9) + E₃(16) + E₄(25) + E₅(36)
        #                   + E₆(9) + E₇(16) + E₈(4) = 116
        n_scalar_features = 1 + 9 + 16 + 25 + 36 + 9 + 16 + 4
        # Curl eigenspaces: C₂(16) + C₃(30) + C₄(48) = 94
        n_curl_features = 16 + 30 + 48
        self.n_features = n_e1_features + n_c1_features + n_scalar_features + n_curl_features

        xavier_scale = math.sqrt(2.0 / (self.n_features + output_size))
        self.W_out = np.random.randn(output_size, self.n_features) * xavier_scale
        self.b_out = np.zeros(output_size)

    def _e1_pipeline(self, a1):
        """
        Multi-stage Hopf computation on the E₁ dipole eigenspace.

        a₁ ∈ R⁴ (coefficients from spectral decomposition).
        These ARE a natural quaternion/rotor on S³.

        Returns feature vector: S² coordinates from each stage,
        plus Berry phases and transport phases from holonomy computation.
        """
        # Embed coefficient vector as rotor on S³
        norm = np.linalg.norm(a1)
        if norm < 1e-10:
            a1_normed = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            a1_normed = a1 / norm
        q = rotor_from_quaternion(*a1_normed)

        s2_points = []      # S² points from each projection (grade-1 mvs)
        s2_coords = []      # R³ coordinates from each projection

        for stage in range(self.N_HOPF_STAGES):
            # Apply learned rotation: Lk * q * Rk
            q_rotated = self.e1_left_rotors[stage] * q * self.e1_right_rotors[stage]
            q_rotated = normalize_rotor(q_rotated)

            # HOPF PROJECT: S³ → S² (the nonlinearity)
            p = hopf_project(q_rotated)
            s2_points.append(p)
            s2_coords.append(s2_extract(p))

            # LIFT back to S³ for next stage (except last)
            if stage < self.N_HOPF_STAGES - 1:
                q = hopf_lift(p, self.e1_lift_phases[stage])

        # Compute holonomy between consecutive S² points
        berry_phases = []
        transport_phases = []
        for i in range(len(s2_points) - 1):
            berry, transport = holonomy_triangle(
                s2_points[i], s2_points[i + 1], e3  # reference: north pole
            )
            berry_phases.append(berry)
            transport_phases.append(transport)

        # Assemble feature vector
        features = []
        for coords in s2_coords:
            features.extend(coords.tolist())  # 3 per stage
        features.extend(berry_phases)         # N_STAGES - 1
        features.extend(transport_phases)     # N_STAGES - 1

        # Scale by input magnitude (orbit radius = representation-theoretic weight)
        return np.array(features) * min(norm, 10.0)

    def _c1_pipeline(self, c1_coeffs):
        """
        Hopf pipeline for C₁ curl eigenspace (chirality detection).

        C₁ ∈ R⁶: the 6D co-exact 1-form eigenspace detects oriented
        structure — loops, circulation, handedness. The difference between
        a 6 and a 9 is a curl feature (§55 of the dependency index).

        First 4 coefficients → quaternion → multi-stage Hopf pipeline.
        Remaining 2 → Poincaré warp (preserves their information).
        """
        # Split: first 4 as quaternion, remaining 2 as scalar features
        q_part = c1_coeffs[:4]
        remaining = c1_coeffs[4:]

        norm = np.linalg.norm(q_part)
        if norm < 1e-10:
            q_normed = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q_normed = q_part / norm
        q = rotor_from_quaternion(*q_normed)

        s2_points = []
        s2_coords = []

        for stage in range(self.N_C1_STAGES):
            q_rotated = self.c1_left_rotors[stage] * q * self.c1_right_rotors[stage]
            q_rotated = normalize_rotor(q_rotated)
            p = hopf_project(q_rotated)
            s2_points.append(p)
            s2_coords.append(s2_extract(p))
            if stage < self.N_C1_STAGES - 1:
                q = hopf_lift(p, self.c1_lift_phases[stage])

        berry_phases = []
        transport_phases = []
        for i in range(len(s2_points) - 1):
            berry, transport = holonomy_triangle(
                s2_points[i], s2_points[i + 1], e3
            )
            berry_phases.append(berry)
            transport_phases.append(transport)

        features = []
        for coords in s2_coords:
            features.extend(coords.tolist())
        features.extend(berry_phases)
        features.extend(transport_phases)

        # Append Poincaré-warped remaining coefficients
        remaining_warped = poincare_warp(remaining)
        features.extend(remaining_warped.tolist())

        return np.array(features) * min(norm, 10.0)

    def forward(self, inputs):
        """
        Forward pass through the spectral geometric pipeline.

        Args:
            inputs: list of floats (784 for MNIST, 11 for Snake, etc.)

        Returns:
            list of output_size logits
        """
        geo = _get_geo()
        kernel = _get_pixel_kernel(self.input_size)

        # --- Project input onto 600-cell vertices ---
        x = np.array(inputs, dtype=np.float64)
        if len(x) < self.input_size:
            x = np.pad(x, (0, self.input_size - len(x)))
        x = x[:self.input_size]

        f = x @ kernel  # signal on 120 vertices

        # --- Spectral decomposition: ALL eigenspaces including aliased ---
        scalar_coeffs = []
        for es in geo["scalar_eigenspaces"]:  # E0..E5 (clean) + E6..E8 (aliased)
            scalar_coeffs.append(es["vectors"].T @ f)

        d0 = geo["d0"]
        df = d0 @ f  # discrete gradient on edges
        curl_coeffs = []
        for es in geo["curl_eigenspaces"]:  # C1..C4
            curl_coeffs.append(es["vectors"].T @ df)

        # Unpack — 9 scalar eigenspaces (6 clean + 3 aliased), 4 curl
        a0 = scalar_coeffs[0]   # E₀: 1D
        a1 = scalar_coeffs[1]   # E₁: 4D (→ Hopf pipeline)
        a2 = scalar_coeffs[2]   # E₂: 9D
        a3 = scalar_coeffs[3]   # E₃: 16D
        a4 = scalar_coeffs[4]   # E₄: 25D
        a5 = scalar_coeffs[5]   # E₅: 36D
        # Aliased modes: Nyquist folding of high-frequency S³ harmonics
        a6 = scalar_coeffs[6] if len(scalar_coeffs) > 6 else np.zeros(9)   # E₆: 9D
        a7 = scalar_coeffs[7] if len(scalar_coeffs) > 7 else np.zeros(16)  # E₇: 16D
        a8 = scalar_coeffs[8] if len(scalar_coeffs) > 8 else np.zeros(4)   # E₈: 4D

        c1, c2, c3, c4 = curl_coeffs

        # --- E₁ (4D): Multi-stage Hopf pipeline (the geometric core) ---
        e1_features = self._e1_pipeline(a1)

        # --- C₁ (6D): Hopf chirality pipeline ---
        # "The difference between a 6 and a 9 is a curl feature"
        c1_features = self._c1_pipeline(c1)

        # --- E₀ (1D): scale + warp ---
        e0_features = poincare_warp(a0 * self.s0)

        # --- E₂ (9D): Givens rotations + warp ---
        a2_rotated = apply_givens_rotations(a2, self.givens2, self.GIVENS_PAIRS_E2)
        e2_features = poincare_warp(a2_rotated)

        # --- E₃–E₅ (clean higher harmonics): scale + warp ---
        e3_features = poincare_warp(a3 * self.scale_e3)
        e4_features = poincare_warp(a4 * self.scale_e4)
        e5_features = poincare_warp(a5 * self.scale_e5)

        # --- E₆–E₈ (aliased modes): scale + warp ---
        e6_features = poincare_warp(a6 * self.scale_e6)
        e7_features = poincare_warp(a7 * self.scale_e7)
        e8_features = poincare_warp(a8 * self.scale_e8)

        # --- C₂–C₄: scale + warp ---
        c2_features = poincare_warp(c2 * self.curl_scales[0])
        c3_features = poincare_warp(c3 * self.curl_scales[1])
        c4_features = poincare_warp(c4 * self.curl_scales[2])

        # --- Concatenate ALL geometric features ---
        features = np.concatenate([
            e1_features,   # 13  (3 Hopf stages + holonomy)
            c1_features,   # 10  (2 Hopf stages + holonomy + 2 remaining)
            e0_features,   # 1
            e2_features,   # 9
            e3_features,   # 16
            e4_features,   # 25
            e5_features,   # 36
            e6_features,   # 9   (aliased)
            e7_features,   # 16  (aliased)
            e8_features,   # 4   (aliased)
            c2_features,   # 16
            c3_features,   # 30
            c4_features,   # 48
        ])

        # --- Minimal readout ---
        logits = self.W_out @ features + self.b_out
        return logits.tolist()

    def select_action(self, signals, signal_order):
        """Select action from signal dictionary."""
        inputs = [signals.get(k, 0.0) for k in signal_order]
        logits = self.forward(inputs)
        return logits.index(max(logits))

    def mutate(self, rate=0.1, scale=0.3):
        """
        Mutate evolved parameters.
        Rotors: geodesic on S³. Phases: additive. Scales: relative.
        Readout: standard GENREG.
        """
        # E₁ pipeline rotors and phases
        for i in range(self.N_HOPF_STAGES):
            self.e1_left_rotors[i] = mutate_rotor(self.e1_left_rotors[i], rate, scale)
            self.e1_right_rotors[i] = mutate_rotor(self.e1_right_rotors[i], rate, scale)
        for i in range(len(self.e1_lift_phases)):
            self.e1_lift_phases[i] = mutate_angle(self.e1_lift_phases[i], rate, scale)

        # C₁ chirality pipeline
        for i in range(self.N_C1_STAGES):
            self.c1_left_rotors[i] = mutate_rotor(self.c1_left_rotors[i], rate, scale)
            self.c1_right_rotors[i] = mutate_rotor(self.c1_right_rotors[i], rate, scale)
        for i in range(len(self.c1_lift_phases)):
            self.c1_lift_phases[i] = mutate_angle(self.c1_lift_phases[i], rate, scale)

        # Scalar eigenspaces
        self.s0 = mutate_scale(self.s0, rate, scale)
        self.givens2 = [mutate_angle(a, rate, scale) for a in self.givens2]
        self.scale_e3 = mutate_scale(self.scale_e3, rate, scale)
        self.scale_e4 = mutate_scale(self.scale_e4, rate, scale)
        self.scale_e5 = mutate_scale(self.scale_e5, rate, scale)
        self.scale_e6 = mutate_scale(self.scale_e6, rate, scale)
        self.scale_e7 = mutate_scale(self.scale_e7, rate, scale)
        self.scale_e8 = mutate_scale(self.scale_e8, rate, scale)
        self.curl_scales = [mutate_scale(s, rate, scale) for s in self.curl_scales]

        # Readout
        mutate_flat_2d(self.W_out, rate, scale)
        mutate_flat(self.b_out, rate, scale)

    def clone(self):
        """Deep copy."""
        c = HopfController.__new__(HopfController)
        c.input_size = self.input_size
        c.hidden_size = self.hidden_size
        c.output_size = self.output_size
        c.N_HOPF_STAGES = self.N_HOPF_STAGES
        c.n_features = self.n_features

        c.e1_left_rotors = [normalize_rotor(R + 0) for R in self.e1_left_rotors]
        c.e1_right_rotors = [normalize_rotor(R + 0) for R in self.e1_right_rotors]
        c.e1_lift_phases = self.e1_lift_phases[:]

        c.c1_left_rotors = [normalize_rotor(R + 0) for R in self.c1_left_rotors]
        c.c1_right_rotors = [normalize_rotor(R + 0) for R in self.c1_right_rotors]
        c.c1_lift_phases = self.c1_lift_phases[:]

        c.s0 = self.s0
        c.givens2 = self.givens2[:]
        c.scale_e3 = self.scale_e3
        c.scale_e4 = self.scale_e4
        c.scale_e5 = self.scale_e5
        c.scale_e6 = self.scale_e6
        c.scale_e7 = self.scale_e7
        c.scale_e8 = self.scale_e8
        c.curl_scales = self.curl_scales[:]
        c.W_out = self.W_out.copy()
        c.b_out = self.b_out.copy()
        return c

    def param_count(self):
        """Total scalar parameters (evolved only)."""
        n_e1_rotors = 2 * self.N_HOPF_STAGES
        n_c1_rotors = 2 * self.N_C1_STAGES
        return (
            n_e1_rotors * 4 +                     # E₁ rotors
            (self.N_HOPF_STAGES - 1) +             # E₁ lift phases
            n_c1_rotors * 4 +                      # C₁ rotors
            (self.N_C1_STAGES - 1) +               # C₁ lift phases
            1 +                                    # s0
            len(self.givens2) +                    # Givens angles
            6 +                                    # scale_e3..e8
            len(self.curl_scales) +                # curl_scales (C₂,C₃,C₄)
            self.W_out.size +
            self.b_out.size
        )

    def effective_dof(self):
        """Effective DOF (rotors have 3 DOF each on S³)."""
        n_e1_rotors = 2 * self.N_HOPF_STAGES
        n_c1_rotors = 2 * self.N_C1_STAGES
        return (
            n_e1_rotors * 3 +
            (self.N_HOPF_STAGES - 1) +
            n_c1_rotors * 3 +
            (self.N_C1_STAGES - 1) +
            1 +
            len(self.givens2) +
            6 +
            len(self.curl_scales) +
            self.W_out.size +
            self.b_out.size
        )

    def to_dict(self):
        """Serialize."""
        return {
            "type": "hopf",
            "version": 3,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "n_hopf_stages": self.N_HOPF_STAGES,
            "e1_left_rotors": [rotor_to_array(R).tolist()
                               for R in self.e1_left_rotors],
            "e1_right_rotors": [rotor_to_array(R).tolist()
                                for R in self.e1_right_rotors],
            "e1_lift_phases": self.e1_lift_phases,
            "c1_left_rotors": [rotor_to_array(R).tolist() for R in self.c1_left_rotors],
            "c1_right_rotors": [rotor_to_array(R).tolist() for R in self.c1_right_rotors],
            "c1_lift_phases": self.c1_lift_phases,
            "n_c1_stages": self.N_C1_STAGES,
            "s0": self.s0,
            "givens2": self.givens2,
            "scale_e3": self.scale_e3,
            "scale_e4": self.scale_e4,
            "scale_e5": self.scale_e5,
            "scale_e6": self.scale_e6,
            "scale_e7": self.scale_e7,
            "scale_e8": self.scale_e8,
            "curl_scales": self.curl_scales,
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize."""
        c = cls.__new__(cls)
        c.input_size = d["input_size"]
        c.hidden_size = d.get("hidden_size", 16)
        c.output_size = d["output_size"]
        c.N_HOPF_STAGES = d.get("n_hopf_stages", 3)
        c.GIVENS_PAIRS_E2 = [(0, 1), (2, 3), (4, 5), (6, 7)]

        c.e1_left_rotors = [rotor_from_quaternion(*q)
                            for q in d["e1_left_rotors"]]
        c.e1_right_rotors = [rotor_from_quaternion(*q)
                             for q in d["e1_right_rotors"]]
        c.e1_lift_phases = d["e1_lift_phases"]

        c.N_C1_STAGES = d.get("n_c1_stages", 2)
        c.c1_left_rotors = [rotor_from_quaternion(*q) for q in d.get("c1_left_rotors", [[1,0,0,0]]*c.N_C1_STAGES)]
        c.c1_right_rotors = [rotor_from_quaternion(*q) for q in d.get("c1_right_rotors", [[1,0,0,0]]*c.N_C1_STAGES)]
        c.c1_lift_phases = d.get("c1_lift_phases", [0.0] * (c.N_C1_STAGES - 1))

        c.s0 = d["s0"]
        c.givens2 = d["givens2"]
        c.scale_e3 = d.get("scale_e3", 1.0)
        c.scale_e4 = d.get("scale_e4", 1.0)
        c.scale_e5 = d.get("scale_e5", 1.0)
        c.scale_e6 = d.get("scale_e6", 1.0)
        c.scale_e7 = d.get("scale_e7", 1.0)
        c.scale_e8 = d.get("scale_e8", 1.0)
        c.curl_scales = d.get("curl_scales", [1.0, 1.0, 1.0])
        c.W_out = np.array(d["W_out"], dtype=np.float64)
        c.b_out = np.array(d["b_out"], dtype=np.float64)

        n_e1_features = 3 * c.N_HOPF_STAGES + 2 * (c.N_HOPF_STAGES - 1)
        n_c1_features = 3 * c.N_C1_STAGES + 2 * (c.N_C1_STAGES - 1) + 2
        c.n_features = (n_e1_features + n_c1_features +
                        1 + 9 + 16 + 25 + 36 + 9 + 16 + 4 +  # scalars E0-E8
                        16 + 30 + 48)  # curls C2-C4
        return c
