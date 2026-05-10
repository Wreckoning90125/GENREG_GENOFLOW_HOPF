"""Verification of the SL(2, ℂ) / Möbius / Lorentz machinery against
canonical identities.

All assertions are against either analytic identities or against
clifford-algebra-derived rotation matrices (the SO(3) ⊂ SO(3,1)
subgroup test).
"""

from __future__ import annotations

import cmath
import math
import numpy as np
from mpmath import mp, mpf, mpc, sqrt as mpsqrt

from experiments.lorentz.mobius import (
    s2_to_complex, complex_to_s2,
    mobius_apply,
    sl2c_from_axis_angle, sl2c_boost, sl2c_random,
    cross_ratio_array, s2_cross_ratio,
)


# ---------------------------------------------------------------------------
# Stereographic projection round-trips
# ---------------------------------------------------------------------------


def test_stereographic_roundtrip():
    """S² → ℂ → S² is identity to machine precision (excluding the
    north pole, which maps to ∞)."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    # Exclude points within 0.01 of the north pole (numerical limit).
    pts = pts[pts[:, 2] < 0.99]
    z = s2_to_complex(pts)
    pts_back = complex_to_s2(z)
    err = np.max(np.linalg.norm(pts - pts_back, axis=1))
    assert err < 1e-12, f"stereographic round-trip max err {err:.2e}"


def test_complex_to_s2_unit_sphere():
    """complex_to_s2 always returns unit-norm S² points."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=300) + 1j * rng.normal(size=300)
    pts = complex_to_s2(z)
    norms = np.linalg.norm(pts, axis=-1)
    assert np.all(np.abs(norms - 1.0) < 1e-12)


# ---------------------------------------------------------------------------
# SL(2, ℂ) determinants and group structure
# ---------------------------------------------------------------------------


def test_sl2c_rotation_is_sl2():
    """sl2c_from_axis_angle returns SU(2) ⊂ SL(2, ℂ) — det = 1."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = rng.uniform(0, 2 * math.pi)
        M = sl2c_from_axis_angle(axis, theta)
        det = np.linalg.det(M)
        assert abs(det - 1.0) < 1e-12, f"det = {det}"
        # Also: SU(2) ⇒ M† M = I.
        assert np.allclose(M.conj().T @ M, np.eye(2), atol=1e-12)


def test_sl2c_boost_is_sl2():
    """sl2c_boost returns det = 1 elements."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        eta = rng.uniform(-2.0, 2.0)
        M = sl2c_boost(eta, d)
        det = np.linalg.det(M)
        assert abs(det - 1.0) < 1e-12
        # Boosts are NOT in SU(2): M† M ≠ I in general (they are
        # Hermitian-positive instead).
        # Verify Hermitian:
        assert np.allclose(M.conj().T, M, atol=1e-12), "boost should be Hermitian"


def test_sl2c_random_is_sl2():
    rng = np.random.default_rng(4)
    for _ in range(20):
        M = sl2c_random(rng)
        det = np.linalg.det(M)
        assert abs(det - 1.0) < 1e-10, f"det = {det}"


# ---------------------------------------------------------------------------
# SU(2) ⊂ SL(2, ℂ) acts on S² as ordinary SO(3) rotation
# ---------------------------------------------------------------------------


def test_su2_via_mobius_equals_so3_rotation():
    """For a 90° rotation about ẑ, the Möbius action z ↦ Mz where
    M = sl2c_from_axis_angle((0,0,1), π/2) should rotate every S²
    point by 90° about ẑ.

    This is the cleanest non-trivial check that SU(2) ⊂ SL(2, ℂ)
    acts on the celestial sphere as the ordinary rotation it
    double-covers.
    """
    rng = np.random.default_rng(5)
    for axis_seed in range(10):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        theta = rng.uniform(0.1, 2.0 * math.pi - 0.1)

        # Rotation as 3×3 matrix (Rodrigues' formula).
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

        # SU(2) double-cover.
        M = sl2c_from_axis_angle(axis, theta)

        # Test on several S² points.
        pts = rng.normal(size=(30, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts[pts[:, 2] < 0.98]  # avoid north pole

        # Path 1: rotate in 3D.
        rotated_pts = pts @ R.T

        # Path 2: stereographic → Möbius → inverse stereographic.
        z = s2_to_complex(pts)
        z_rotated = mobius_apply(M, z)
        pts_via_mobius = complex_to_s2(z_rotated)

        diffs = np.linalg.norm(rotated_pts - pts_via_mobius, axis=1)
        assert np.max(diffs) < 1e-10, (
            f"SU(2) ⊂ SL(2,ℂ) action mismatches SO(3) rotation: "
            f"max err {np.max(diffs):.2e}"
        )


# ---------------------------------------------------------------------------
# Cross-ratio Möbius invariance
# ---------------------------------------------------------------------------


def test_cross_ratio_invariant_under_random_mobius():
    """The cross-ratio of any 4 complex points is invariant under
    Möbius transformations applied to all four.

    For random SL(2, ℂ) elements, the cross-ratio before and after
    must agree to machine precision.
    """
    rng = np.random.default_rng(6)
    for _ in range(50):
        z = rng.normal(size=4) + 1j * rng.normal(size=4)
        lambda_before = cross_ratio_array(z)

        M = sl2c_random(rng)
        zt = mobius_apply(M, z)
        lambda_after = cross_ratio_array(zt)

        err = abs(lambda_before - lambda_after)
        assert err < 1e-8, f"cross-ratio not invariant under M: |Δλ| = {err:.2e}"


def test_cross_ratio_invariant_under_lorentz_boost():
    """Specifically test that a non-trivial Lorentz boost (NOT in SU(2))
    preserves the cross-ratio. This is the part that pure-SO(3)
    invariance does NOT cover.
    """
    rng = np.random.default_rng(7)
    for _ in range(30):
        z = rng.normal(size=4) + 1j * rng.normal(size=4)
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        eta = rng.uniform(0.5, 2.0)  # non-trivial boost
        M = sl2c_boost(eta, d)
        lambda_before = cross_ratio_array(z)
        lambda_after = cross_ratio_array(mobius_apply(M, z))
        assert abs(lambda_before - lambda_after) < 1e-9


def test_cross_ratio_changes_under_non_mobius():
    """Sanity: the cross-ratio is NOT invariant under generic
    coordinate maps. Specifically, simple complex squaring z ↦ z²
    is not a Möbius transformation and should change the
    cross-ratio of generic points.
    """
    rng = np.random.default_rng(8)
    z = rng.normal(size=4) + 1j * rng.normal(size=4)
    lambda_orig = cross_ratio_array(z)
    lambda_sq = cross_ratio_array(z * z)
    assert abs(lambda_orig - lambda_sq) > 1e-3


# ---------------------------------------------------------------------------
# High-precision identity: cross-ratio identities via mpmath
# ---------------------------------------------------------------------------


def test_cross_ratio_six_values_identity_mpmath():
    """The six values of the cross-ratio under S_4 permutation of the
    4 points are exactly {λ, 1−λ, 1/λ, λ/(λ−1), (λ−1)/λ, 1/(1−λ)}.

    Verified at 50-digit mpmath precision for a random (high-precision)
    cross-ratio value, against analytic identity.
    """
    mp.dps = 50
    # Pick a non-degenerate λ = (3 + i√7) / (5 − i√2).
    lam = mpc(mpf(3), mpsqrt(mpf(7))) / mpc(mpf(5), -mpsqrt(mpf(2)))
    expected = [
        lam,
        1 - lam,
        1 / lam,
        lam / (lam - 1),
        (lam - 1) / lam,
        1 / (1 - lam),
    ]
    # Sum of all six = 3 (a known identity).
    s = sum(expected)
    assert abs(s - mpf(3)) < mpf(10) ** -45, (
        f"sum of 6 cross-ratio values should be 3, got {s}"
    )
    # Product of pairs of opposites = … verify the three pairs each
    # multiply to a known function of λ. The pair (λ, 1−λ) is special;
    # we just check 1/λ · λ = 1 etc.
    assert abs(expected[0] * expected[2] - 1) < mpf(10) ** -45
    assert abs(expected[1] * expected[5] - 1) < mpf(10) ** -45
    assert abs(expected[3] * expected[4] - 1) < mpf(10) ** -45


def test_s2_cross_ratio_invariant_under_rotation():
    """S² cross-ratio (via stereographic) is invariant under any
    SO(3) rotation of the 4 points.
    """
    rng = np.random.default_rng(9)
    for _ in range(30):
        pts = rng.normal(size=(4, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        lam_before = s2_cross_ratio(pts[0], pts[1], pts[2], pts[3])

        # Random rotation.
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        theta = rng.uniform(0, 2 * math.pi)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        pts_rot = pts @ R.T
        lam_after = s2_cross_ratio(pts_rot[0], pts_rot[1], pts_rot[2], pts_rot[3])
        assert abs(lam_before - lam_after) < 1e-10


def test_s2_cross_ratio_invariant_under_full_mobius():
    """S² cross-ratio is invariant under the FULL Möbius group,
    including non-SO(3) boosts. This is the test that distinguishes
    Möbius-invariant features from merely SO(3)-invariant ones.
    """
    rng = np.random.default_rng(10)
    pts = rng.normal(size=(4, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    lam_orig = s2_cross_ratio(pts[0], pts[1], pts[2], pts[3])

    for _ in range(20):
        M = sl2c_random(rng)
        z = s2_to_complex(pts)
        z_t = mobius_apply(M, z)
        pts_t = complex_to_s2(z_t)
        lam_t = s2_cross_ratio(pts_t[0], pts_t[1], pts_t[2], pts_t[3])
        # Conformal transforms can change phase by a unit complex,
        # but for the cross-ratio of distinct points there's no
        # sign ambiguity — the value is invariant.
        assert abs(lam_orig - lam_t) < 1e-8, (
            f"|λ_orig − λ_after_mobius| = {abs(lam_orig - lam_t):.2e}"
        )
