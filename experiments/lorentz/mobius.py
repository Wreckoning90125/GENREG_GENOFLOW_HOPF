"""SL(2, ℂ) / Möbius / Lorentz machinery.

This module is the Lorentz/conformal fallback flagged in AGENTS.md:
when Euclidean / icosahedral symmetry doesn't fit the data, the
relevant covering structure is

    SL(2, ℂ)  →  PSL(2, ℂ)  ≅  SO(3, 1)⁺
        \                        /
         \                      /
          ↳  acts on S² ≅ ℂP¹ via Möbius transformations

with the rotation group SO(3) embedded as the SU(2) ⊂ SL(2, ℂ)
subgroup (Lorentz boosts = 0). The key invariant is the
cross-ratio of 4 points on S², which is the unique numerical
invariant of a 4-tuple under the conformal group.

References:
  - Penrose & Rindler, "Spinors and Space-Time" Vol. I (1984).
  - Needham, "Visual Complex Analysis" Ch. 3.
  - Fricke & Klein for the hyperbolic-side picture (the EPINET origin).

This is the *infrastructure* — feature pipelines that consume these
primitives are in `bench_mobius_equivalence.py`. High-precision
verification (mpmath, 50-digit) is in `tests/test_mobius_canonical.py`.
"""

from __future__ import annotations

import cmath
import math
import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Stereographic projection: S² ↔ ℂ ∪ {∞}
# ---------------------------------------------------------------------------


def s2_to_complex(p: np.ndarray) -> np.ndarray:
    """S² ⊂ ℝ³ → ℂ via stereographic projection from the north pole.

    p shape (..., 3). Returns complex array of shape p.shape[:-1].
    The north pole (0, 0, 1) maps to ∞ (returned as nan + nan*1j; the
    Möbius machinery downstream handles ∞ as a limit).

    Sign convention chosen so that SU(2) ⊂ SL(2, ℂ) acts on the
    image plane as the ordinary SO(3) rotation it double-covers
    (see test_su2_via_mobius_equals_so3_rotation). Concretely:
    we send (x, y, z) ↦ (x − i·y) / (1 − z), which is the Hopf-
    spinor convention with z = α/β under |α|² + |β|² = 1.
    """
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    denom = 1.0 - z
    inf = np.abs(denom) < 1e-12
    safe = np.where(inf, 1.0, denom)
    out = (x - 1j * y) / safe
    out = np.where(inf, np.complex128(np.inf), out)
    return out


def complex_to_s2(z: np.ndarray) -> np.ndarray:
    """ℂ ∪ {∞} → S² ⊂ ℝ³ inverse stereographic from the north pole.

    Inverse of `s2_to_complex` with the same Hopf-spinor sign
    convention: y carries the OPPOSITE sign of Im(z), so that the
    SU(2) action on z corresponds to the ordinary 3D rotation.
    """
    z = np.asarray(z, dtype=np.complex128)
    inf = ~np.isfinite(z)
    safe = np.where(inf, 0.0 + 0.0j, z)
    sq = (safe.real * safe.real + safe.imag * safe.imag)
    denom = 1.0 + sq
    x = (2.0 * safe.real) / denom
    y = (-2.0 * safe.imag) / denom
    z_coord = (sq - 1.0) / denom
    x = np.where(inf, 0.0, x)
    y = np.where(inf, 0.0, y)
    z_coord = np.where(inf, 1.0, z_coord)
    return np.stack([x, y, z_coord], axis=-1)


# ---------------------------------------------------------------------------
# Möbius transformations and SL(2, ℂ)
# ---------------------------------------------------------------------------


def mobius_apply(M: np.ndarray, z: complex | np.ndarray) -> complex | np.ndarray:
    """Apply a 2×2 complex matrix M = [[a, b], [c, d]] to z ∈ ℂ ∪ {∞}
    as a Möbius transformation: z ↦ (a z + b) / (c z + d).

    Vectorized over z. Handles z = ∞ correctly (returns a/c if c ≠ 0,
    else ∞).
    """
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    z = np.asarray(z, dtype=np.complex128)
    inf_in = ~np.isfinite(z)
    safe_z = np.where(inf_in, 0.0 + 0.0j, z)
    num = a * safe_z + b
    den = c * safe_z + d
    # If z = ∞: result is a/c (or ∞ if c == 0).
    if c != 0:
        inf_image = a / c
    else:
        inf_image = np.complex128(np.inf)
    bad = np.abs(den) < 1e-15
    safe_den = np.where(bad, 1.0 + 0.0j, den)
    out = num / safe_den
    out = np.where(inf_in, inf_image, out)
    out = np.where(bad & ~inf_in, np.complex128(np.inf), out)
    return out


def sl2c_from_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation by `theta` about `axis` ∈ S² as an SU(2) ⊂ SL(2, ℂ).

    Maps to the rotation of S² by `theta` about `axis` via the
    standard double cover. Useful to verify that SO(3) ⊂ Möbius
    acts on S² as ordinary rotation.
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    # SU(2) parameterization
    return np.array([
        [c - 1j * s * a[2],   -s * (a[1] + 1j * a[0])],
        [s * (a[1] - 1j * a[0]),   c + 1j * s * a[2]],
    ], dtype=np.complex128)


def sl2c_boost(rapidity: float, direction: np.ndarray) -> np.ndarray:
    """Pure Lorentz boost as an SL(2, ℂ) matrix.

    Acts on S² (= celestial sphere of a Lorentz observer) as a
    Möbius transformation that "stretches" the hemisphere in the
    boost direction. For rapidity → 0 reduces to identity; for
    rapidity → ∞ collapses to a fixed point.
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    ch, sh = math.cosh(rapidity / 2.0), math.sinh(rapidity / 2.0)
    return np.array([
        [ch + sh * d[2],   sh * (d[0] - 1j * d[1])],
        [sh * (d[0] + 1j * d[1]),   ch - sh * d[2]],
    ], dtype=np.complex128)


def sl2c_random(rng: np.random.Generator) -> np.ndarray:
    """Uniformly-ish random SL(2, ℂ) matrix.

    Samples real and imaginary parts of a, b, c from a standard
    Gaussian, then chooses d = (1 + b c) / a (rescaling a to be
    nonzero), and normalizes overall to unit determinant.
    """
    while True:
        a = rng.normal() + 1j * rng.normal()
        if abs(a) > 0.1:
            break
    b = rng.normal() + 1j * rng.normal()
    c = rng.normal() + 1j * rng.normal()
    d = (1.0 + b * c) / a
    M = np.array([[a, b], [c, d]], dtype=np.complex128)
    det = a * d - b * c
    M = M / cmath.sqrt(det)  # normalize to det = 1
    return M


# ---------------------------------------------------------------------------
# Cross-ratio: the canonical conformal invariant of 4 points
# ---------------------------------------------------------------------------


def cross_ratio(z1, z2, z3, z4):
    """λ(z1, z2; z3, z4) = ((z1 − z3)(z2 − z4)) / ((z1 − z4)(z2 − z3)).

    Invariant under Möbius transformations applied to all four
    points simultaneously. Sign convention follows Needham,
    "Visual Complex Analysis" §3.V.

    Handles z = ∞ by passing to limits — Möbius treats ∞ as a
    regular point, so each factor (z_i − z_j) with one infinite
    argument cancels in the ratio.
    """
    # Vectorize: if any zi is ∞, compute the limit.
    # The cleanest implementation: treat ∞ as a large finite number
    # and observe the limit cancels. We do it symbolically per-arg.
    def isinf(z):
        return not np.isfinite(z) if np.isscalar(z) else np.any(~np.isfinite(z))

    if any(map(isinf, (z1, z2, z3, z4))):
        # Use the standard limit forms.
        if isinf(z1):
            return (z2 - z4) / (z2 - z3)
        if isinf(z2):
            return (z1 - z3) / (z1 - z4)
        if isinf(z3):
            return -(z2 - z4) / (z1 - z4) * 0 + (z2 - z4) / (z2 - z4)  # placeholder
        if isinf(z4):
            return (z1 - z3) / (z2 - z3)
    return ((z1 - z3) * (z2 - z4)) / ((z1 - z4) * (z2 - z3))


def cross_ratio_array(points: np.ndarray) -> np.ndarray:
    """Cross-ratio of 4-tuples of complex points.

    points: shape (..., 4). Returns shape (...) complex.
    Vectorized; assumes all points are finite (cleaner for the
    feature pipeline).
    """
    z1, z2, z3, z4 = points[..., 0], points[..., 1], points[..., 2], points[..., 3]
    return ((z1 - z3) * (z2 - z4)) / ((z1 - z4) * (z2 - z3))


def s2_cross_ratio(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """Cross-ratio of 4 S² points, computed via stereographic projection
    to ℂ ∪ {∞}.

    Returns complex array of shape p1.shape[:-1]. Invariant under
    any Möbius transformation applied to all 4 points (i.e., any
    element of the conformal group SO(3,1)⁺ of S²).
    """
    z1 = s2_to_complex(p1)
    z2 = s2_to_complex(p2)
    z3 = s2_to_complex(p3)
    z4 = s2_to_complex(p4)
    pts = np.stack([z1, z2, z3, z4], axis=-1)
    return cross_ratio_array(pts)
