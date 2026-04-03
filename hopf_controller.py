# ================================================================
# Hopf Geometric Controller v2 — Spectral Substrate on S³
#
# Architecture derived from the geometry of the 600-cell and the
# Hopf fibration. NOT a quaternion MLP.
#
# Pipeline:
#   1. Project input onto 120 vertices of 600-cell via Hopf map
#   2. Spectral decomposition into eigenbasis (Theorem 3 + 5)
#   3. Irrep-respecting rotations within each eigenspace
#   4. Poincaré conformal warp per eigenspace (the nonlinearity)
#   5. Linear readout to action logits
#
# Fixed (geometric, not learned):
#   - 600-cell vertices, eigenbasis, curl basis, projection kernel
#
# Evolved (the only parameters evolution touches):
#   - Per-eigenspace rotations (quaternions, Givens angles)
#   - Per-eigenspace scales
#   - Readout weights
# ================================================================

import math
import random
import numpy as np
import quaternion as npquat

from cell600 import get_geometry


# ================================================================
# Geometry cache (computed once on first import)
# ================================================================

_GEO = None
_PIXEL_KERNEL = None  # (784, 120) soft assignment for MNIST


def _get_geo():
    global _GEO
    if _GEO is None:
        _GEO = get_geometry()
    return _GEO


# ================================================================
# Pixel-to-S³ mapping and soft-assignment kernel (MNIST)
# ================================================================

def _build_pixel_kernel(input_size, kappa=10.0):
    """
    Build the soft-assignment kernel mapping input positions to 600-cell vertices.

    For MNIST (784 pixels on a 28×28 grid):
      1. Map pixel (i,j) to normalized coords (x,y) ∈ [-1, 1]²
      2. Inverse stereographic projection: (x,y) → S²
      3. Hopf section lift: S² → S³
      4. Soft assignment to 120 vertices via K[pixel, vertex] = softmax(κ · q_pixel · q_vertex)

    Returns:
        np.ndarray of shape (input_size, 120)
    """
    geo = _get_geo()
    vertices = geo["vertices"]  # (120, 4)

    if input_size == 784:
        # MNIST: 28×28 grid
        rows, cols = 28, 28
    else:
        # Generic: assume square, or fall back to 1D distribution on S³
        side = int(math.ceil(math.sqrt(input_size)))
        rows, cols = side, side

    pixel_quats = []
    for idx in range(input_size):
        if input_size == 784:
            r, c = divmod(idx, 28)
        else:
            r, c = divmod(idx, cols)

        # Normalize to [-1, 1]²
        x = 2.0 * c / (cols - 1) - 1.0 if cols > 1 else 0.0
        y = 2.0 * r / (rows - 1) - 1.0 if rows > 1 else 0.0

        # Inverse stereographic projection: (x, y) → S²
        denom = 1.0 + x * x + y * y
        X = 2.0 * x / denom
        Y = 2.0 * y / denom
        Z = (x * x + y * y - 1.0) / denom

        # Hopf section lift: S² → S³ (canonical section, phase θ=0)
        if Z > -0.999:
            s = math.sqrt(2.0 * (1.0 + Z))
            q = np.array([s / 2.0, -Y / s, X / s, 0.0])
        else:
            q = np.array([0.0, 0.0, 1.0, 0.0])

        q = q / np.linalg.norm(q)
        pixel_quats.append(q)

    pixel_quats = np.array(pixel_quats)  # (input_size, 4)

    # Soft assignment: K[pixel, vertex] = softmax(κ · q_pixel · q_vertex)
    # Raw dot products (not absolute — these are oriented on S³)
    dots = pixel_quats @ vertices.T  # (input_size, 120)
    # Use absolute value for assignment (Hopf fibers are circles, both orientations valid)
    dots = np.abs(dots)
    # Softmax along vertex axis with temperature κ
    scaled = kappa * dots
    scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability
    exp_scaled = np.exp(scaled)
    kernel = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

    return kernel.astype(np.float64)


def _get_pixel_kernel(input_size):
    """Get or compute cached pixel kernel."""
    global _PIXEL_KERNEL
    if _PIXEL_KERNEL is None or _PIXEL_KERNEL.shape[0] != input_size:
        _PIXEL_KERNEL = _build_pixel_kernel(input_size)
    return _PIXEL_KERNEL


# ================================================================
# Poincaré conformal warp (the nonlinearity)
# ================================================================

def poincare_warp(coefficients):
    """
    Radial conformal compression: r' = 2·tanh(r/2).
    Applied to a coefficient vector AS A WHOLE, not per-component.
    Preserves angles between components. Bounded output (radius < 2).
    Invertible via 2·atanh(r'/2).

    This IS the nonlinearity. There is no other.
    """
    r = np.linalg.norm(coefficients)
    if r < 1e-10:
        return coefficients.copy()
    r_prime = 2.0 * math.tanh(r / 2.0)
    return coefficients * (r_prime / r)


# ================================================================
# Within-eigenspace rotation operations
# ================================================================

def apply_so4_rotation(coefficients, qL, qR):
    """
    Rotate a 4D coefficient vector via SO(4) parameterized as
    a pair of unit quaternions: v → qL * v * conj(qR).

    This is the proper isometry of S³ acting on the 4D dipole eigenspace.
    Uses numpy-quaternion for the operation.
    """
    # Convert coefficient vector to quaternion
    v = np.quaternion(coefficients[0], coefficients[1],
                      coefficients[2], coefficients[3])
    # Apply double-cover rotation
    result = qL * v * qR.conjugate()
    return np.array([result.w, result.x, result.y, result.z])


def apply_givens_rotations(coefficients, angles, pairs):
    """
    Apply a sequence of Givens (planar) rotations to a coefficient vector.
    Each rotation acts in a 2D plane specified by (i, j) index pair.

    For the 9D quadrupole eigenspace, this parameterizes a subgroup of SO(9).
    """
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
# Mutation operations
# ================================================================

def mutate_quaternion(q, rate, scale):
    """Geodesic perturbation on S³ via exponential map."""
    if random.random() > rate:
        return q
    # Random tangent vector
    v = np.array([random.gauss(0, scale) for _ in range(3)])
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return q
    half_angle = norm_v
    s = math.sin(half_angle) / norm_v
    delta = np.quaternion(math.cos(half_angle), s * v[0], s * v[1], s * v[2])
    result = q * delta
    # Normalize
    result = result / abs(result)
    return result


def mutate_angle(angle, rate, scale):
    """Additive Gaussian mutation on an angle (wraps naturally)."""
    if random.random() > rate:
        return angle
    return angle + random.gauss(0, scale)


def mutate_scale(s, rate, scale):
    """Scale-relative mutation matching GENREG protein style."""
    if random.random() > rate:
        return s
    return s + random.gauss(0, scale * (abs(s) + 1e-9))


def mutate_flat(arr, rate, scale):
    """Standard GENREG mutation on flat weight array."""
    for i in range(len(arr)):
        if random.random() < rate:
            arr[i] += random.gauss(0, scale)


def mutate_flat_2d(arr, rate, scale):
    """Standard GENREG mutation on 2D weight array."""
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if random.random() < rate:
                arr[i, j] += random.gauss(0, scale)


# ================================================================
# HopfController — The Spectral Geometric Substrate
# ================================================================

class HopfController:
    """
    Controller that computes via the spectral geometry of the 600-cell.

    Fixed (determined by geometry, never mutated):
        - 600-cell vertices (120 unit quaternions of 2I)
        - Scalar eigenbasis E0-E5 (Theorem 3)
        - Curl eigenbasis C1-C4 (Theorem 5)
        - Pixel-to-vertex soft-assignment kernel

    Evolved (the parameters GENREG mutates):
        - s0: scale on DC component (E0, 1D) → 1 param
        - qL1, qR1: SO(4) rotation on dipole space (E1, 4D) → 2 quaternions
        - givens2: Givens angles on quadrupole space (E2, 9D) → 4 angles
        - curl_scale: scale on curl features (C1, 6D) → 1 param
        - W_out, b_out: linear readout → output_size × n_features + output_size

    Total evolved params (minimal, MNIST): ~225
    """

    # Eigenspaces used (minimal version: E0, E1, E2, C1)
    # Feature dims: 1 + 4 + 9 + 6 = 20
    N_FEATURES = 20

    # Givens rotation pairs for 9D quadrupole space
    GIVENS_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

    def __init__(self, input_size=784, hidden_size=16, output_size=10):
        # Store sizes for interface compatibility
        self.input_size = input_size
        self.hidden_size = hidden_size  # unused internally, kept for interface
        self.output_size = output_size

        # Ensure geometry is loaded
        _get_geo()

        # Pre-compute pixel kernel for this input size
        _get_pixel_kernel(input_size)

        # --- Evolved parameters ---

        # E0 (1D): scalar scale
        self.s0 = 1.0

        # E1 (4D): SO(4) rotation via pair of unit quaternions
        self.qL1 = np.quaternion(1, 0, 0, 0)
        self.qR1 = np.quaternion(1, 0, 0, 0)

        # E2 (9D): Givens rotations (4 angles for 4 rotation planes)
        self.givens2 = [0.0] * len(self.GIVENS_PAIRS)

        # C1 (6D): scalar scale
        self.curl_scale = 1.0

        # Readout: linear map from features to logits
        # Xavier initialization
        fan_in = self.N_FEATURES
        fan_out = output_size
        xavier_scale = math.sqrt(2.0 / (fan_in + fan_out))
        self.W_out = np.random.randn(output_size, self.N_FEATURES) * xavier_scale
        self.b_out = np.zeros(output_size)

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

        # --- Step 2: Project input onto 600-cell vertices ---
        x = np.array(inputs, dtype=np.float64)
        if len(x) < self.input_size:
            x = np.pad(x, (0, self.input_size - len(x)))
        x = x[:self.input_size]

        # Soft assignment: (input_size,) @ (input_size, 120) → (120,)
        f = x @ kernel  # signal on 120 vertices

        # --- Step 3: Spectral decomposition ---

        # Scalar eigenspaces
        E0_vecs = geo["scalar_eigenspaces"][0]["vectors"]  # (120, 1)
        E1_vecs = geo["scalar_eigenspaces"][1]["vectors"]  # (120, 4)
        E2_vecs = geo["scalar_eigenspaces"][2]["vectors"]  # (120, 9)

        a0 = E0_vecs.T @ f  # (1,)
        a1 = E1_vecs.T @ f  # (4,)
        a2 = E2_vecs.T @ f  # (9,)

        # Curl eigenspace: compute discrete gradient on edges, then project
        d0 = geo["d0"]  # (720, 120)
        df = d0 @ f      # discrete gradient, (720,)

        C1_vecs = geo["curl_eigenspaces"][0]["vectors"]  # (720, 6)
        c1 = C1_vecs.T @ df  # (6,)

        # --- Step 4: Irrep-respecting rotations + Poincaré warp ---

        # E0 (1D): just scale
        a0_transformed = poincare_warp(a0 * self.s0)

        # E1 (4D): SO(4) rotation via quaternion pair, then warp
        a1_rotated = apply_so4_rotation(a1, self.qL1, self.qR1)
        a1_transformed = poincare_warp(a1_rotated)

        # E2 (9D): Givens rotations, then warp
        a2_rotated = apply_givens_rotations(a2, self.givens2, self.GIVENS_PAIRS)
        a2_transformed = poincare_warp(a2_rotated)

        # C1 (6D): scale, then warp
        c1_transformed = poincare_warp(c1 * self.curl_scale)

        # --- Step 5: Concatenate and readout ---
        features = np.concatenate([
            a0_transformed,   # 1
            a1_transformed,   # 4
            a2_transformed,   # 9
            c1_transformed,   # 6
        ])  # total: 20

        logits = self.W_out @ features + self.b_out

        return logits.tolist()

    def select_action(self, signals, signal_order):
        """
        Select action from signal dictionary.
        Same interface as Controller.
        """
        inputs = [signals.get(k, 0.0) for k in signal_order]
        logits = self.forward(inputs)
        return logits.index(max(logits))

    def mutate(self, rate=0.1, scale=0.3):
        """
        Mutate evolved parameters.

        Quaternions: geodesic perturbation on S³.
        Angles: additive Gaussian (wraps naturally).
        Scales: scale-relative Gaussian.
        Readout weights: standard GENREG additive Gaussian.
        """
        self.s0 = mutate_scale(self.s0, rate, scale)
        self.qL1 = mutate_quaternion(self.qL1, rate, scale)
        self.qR1 = mutate_quaternion(self.qR1, rate, scale)
        self.givens2 = [mutate_angle(a, rate, scale) for a in self.givens2]
        self.curl_scale = mutate_scale(self.curl_scale, rate, scale)
        mutate_flat_2d(self.W_out, rate, scale)
        mutate_flat(self.b_out, rate, scale)

    def clone(self):
        """Create a deep copy of this controller."""
        c = HopfController.__new__(HopfController)
        c.input_size = self.input_size
        c.hidden_size = self.hidden_size
        c.output_size = self.output_size
        c.s0 = self.s0
        c.qL1 = np.quaternion(self.qL1.w, self.qL1.x, self.qL1.y, self.qL1.z)
        c.qR1 = np.quaternion(self.qR1.w, self.qR1.x, self.qR1.y, self.qR1.z)
        c.givens2 = self.givens2[:]
        c.curl_scale = self.curl_scale
        c.W_out = self.W_out.copy()
        c.b_out = self.b_out.copy()
        return c

    def param_count(self):
        """Total scalar parameters (evolved only)."""
        return (
            1 +                              # s0
            4 + 4 +                          # qL1, qR1 (4 each, 3 DOF each)
            len(self.givens2) +              # Givens angles
            1 +                              # curl_scale
            self.W_out.size +                # readout weights
            self.b_out.size                  # readout bias
        )

    def effective_dof(self):
        """Effective degrees of freedom (accounting for S³ constraints)."""
        return (
            1 +                              # s0
            3 + 3 +                          # qL1, qR1 (3 DOF each on S³)
            len(self.givens2) +              # Givens angles
            1 +                              # curl_scale
            self.W_out.size +                # readout (unconstrained)
            self.b_out.size
        )

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "type": "hopf",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "s0": self.s0,
            "qL1": [self.qL1.w, self.qL1.x, self.qL1.y, self.qL1.z],
            "qR1": [self.qR1.w, self.qR1.x, self.qR1.y, self.qR1.z],
            "givens2": self.givens2,
            "curl_scale": self.curl_scale,
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize from dictionary."""
        c = cls.__new__(cls)
        c.input_size = d["input_size"]
        c.hidden_size = d.get("hidden_size", 16)
        c.output_size = d["output_size"]
        c.s0 = d["s0"]
        q = d["qL1"]
        c.qL1 = np.quaternion(q[0], q[1], q[2], q[3])
        q = d["qR1"]
        c.qR1 = np.quaternion(q[0], q[1], q[2], q[3])
        c.givens2 = d["givens2"]
        c.curl_scale = d["curl_scale"]
        c.W_out = np.array(d["W_out"], dtype=np.float64)
        c.b_out = np.array(d["b_out"], dtype=np.float64)
        return c
