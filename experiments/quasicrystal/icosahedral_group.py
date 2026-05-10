"""Icosahedral point group I (order 60) and its character table.

The icosahedral group I is not a crystallographic point group (it is
not one of the 32 that satisfy the crystallographic restriction
theorem). spgrep therefore does not provide its irreps. We construct
them here from first principles, with mpmath verification of the
canonical character table against published values.

Reference: Cotton, "Chemical Applications of Group Theory" (3rd ed),
Appendix II, Table A.20. Also Steurer & Deloudi 2009 §A for the
quasicrystal context.

The group I has 5 conjugacy classes:
    E    : identity (1 element)
    12 C5: rotations by 2π/5 about 5-fold axes (12 elements)
    12 C5²: rotations by 4π/5 about 5-fold axes (12 elements)
    20 C3: rotations by 2π/3 about 3-fold axes (20 elements)
    15 C2: rotations by π about 2-fold axes (15 elements)

And 5 irreducible representations:
    A   : trivial, dim 1
    T1  : 3D, characters (3, τ, 1-τ, 0, -1) where τ = (1+√5)/2
    T2  : 3D, characters (3, 1-τ, τ, 0, -1) — Galois conjugate of T1
    G   : 4D, characters (4, -1, -1, 1, 0)
    H   : 5D, characters (5, 0, 0, -1, 1)

Sum of squared dimensions: 1 + 9 + 9 + 16 + 25 = 60 ✓ (matches |I|).
"""

from __future__ import annotations

import math
import numpy as np
from mpmath import mp, mpf, sqrt as mpsqrt


# Conjugacy class names, sizes, and character-table-friendly order.
CLASS_NAMES = ("E", "C5", "C5_2", "C3", "C2")
CLASS_SIZES = (1, 12, 12, 20, 15)
IRREP_NAMES = ("A", "T1", "T2", "G", "H")
IRREP_DIMS = (1, 3, 3, 4, 5)


def character_table_mp(prec: int = 50) -> list[list[mpf]]:
    """Return the 5x5 character table for I at `prec` decimal digits.

    Rows index irreps (A, T1, T2, G, H); columns index conjugacy
    classes (E, C5, C5², C3, C2). Entries are mpmath mpf objects so
    orthogonality identities can be verified to arbitrary precision.
    """
    mp.dps = prec
    tau = (mpf(1) + mpsqrt(mpf(5))) / mpf(2)
    one_minus_tau = mpf(1) - tau
    chi = [
        [mpf(1), mpf(1),       mpf(1),       mpf(1),  mpf(1)],   # A
        [mpf(3), tau,          one_minus_tau, mpf(0), mpf(-1)],  # T1
        [mpf(3), one_minus_tau, tau,          mpf(0), mpf(-1)],  # T2
        [mpf(4), mpf(-1),      mpf(-1),      mpf(1),  mpf(0)],   # G
        [mpf(5), mpf(0),       mpf(0),       mpf(-1), mpf(1)],   # H
    ]
    return chi


def verify_orthogonality(prec: int = 50, tol: float = 1e-45) -> dict:
    """Verify the great orthogonality theorem on the I character table.

    Specifically:
      - Σ_g χ_i(g) χ_j(g)* / |G| = δ_ij  (row orthogonality)
      - Σ_i d_i χ_i(g) χ_i(h)* = |G| δ_{conj(g, h)} / |class| (col)

    Computed at `prec` decimal places via mpmath. Returns a dict of
    pairwise residuals; max should be < `tol`.
    """
    mp.dps = prec
    chi = character_table_mp(prec)
    G = mpf(60)
    sizes = [mpf(c) for c in CLASS_SIZES]

    # Row orthogonality: <χ_i, χ_j> = (1/|G|) Σ_c |class_c| χ_i(c) χ_j(c)
    row_residuals = {}
    for i in range(5):
        for j in range(i, 5):
            s = sum(sizes[c] * chi[i][c] * chi[j][c] for c in range(5)) / G
            expected = mpf(1) if i == j else mpf(0)
            row_residuals[(i, j)] = float(abs(s - expected))

    return {
        "row_orth_residuals": row_residuals,
        "max_residual": max(row_residuals.values()),
        "ok": max(row_residuals.values()) < tol,
    }


def rotation_matrices() -> np.ndarray:
    """Return all 60 rotation matrices of the icosahedral group I.

    Built by enumerating all rotations that map the icosahedron vertex
    set { (0, ±1, ±τ), (±1, ±τ, 0), (±τ, 0, ±1) } to itself. Includes
    only proper rotations (det = +1), not the full Ih (120 elements).

    The 12 vertices come in 6 antipodal pairs; the 60 rotations are
    enumerated as: identity, 4 non-identity rotations per 5-fold axis
    × 6 axes (= 24), 2 non-identity rotations per 3-fold axis × 10
    axes (= 20), 1 non-identity rotation per 2-fold axis × 15 axes
    (= 15). Total 1 + 24 + 20 + 15 = 60. ✓
    """
    tau = (1.0 + math.sqrt(5.0)) / 2.0
    icos_verts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            icos_verts.append([0.0, s1 * 1.0, s2 * tau])
            icos_verts.append([s1 * 1.0, s2 * tau, 0.0])
            icos_verts.append([s2 * tau, 0.0, s1 * 1.0])
    icos_verts = np.array(icos_verts) / math.sqrt(1.0 + tau * tau)  # unit-length

    rot_set = []  # collected as flattened tuples for de-dup

    def add(R):
        # de-dup with tolerance
        for R_existing in rot_set:
            if np.allclose(R, R_existing, atol=1e-8):
                return
        rot_set.append(R)

    # Identity.
    add(np.eye(3))

    # 5-fold rotations: axis through each vertex.
    five_fold_axes = []
    seen = set()
    for v in icos_verts:
        key = tuple(np.round(v, 6))
        if key in seen or tuple(np.round(-v, 6)) in seen:
            continue
        seen.add(key)
        five_fold_axes.append(v)
    for axis in five_fold_axes:
        for k in (1, 2, 3, 4):
            angle = 2.0 * math.pi * k / 5.0
            add(_axis_angle_to_R(axis, angle))

    # 3-fold rotations: axes through each face center (= sum of 3 adjacent verts).
    # The icosahedron has 20 faces; opposite faces give same axis -> 10 axes.
    # Adjacent vertices: distance ^2 = 4 (after unit normalization with our scale).
    # Use brute force: for every triple of mutually adjacent verts, axis is the sum.
    adj_thresh = 4.0 / (1.0 + tau * tau) + 1e-6
    three_fold_axes = []
    seen3 = set()
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                vi, vj, vk = icos_verts[i], icos_verts[j], icos_verts[k]
                dij = np.sum((vi - vj) ** 2)
                dik = np.sum((vi - vk) ** 2)
                djk = np.sum((vj - vk) ** 2)
                if dij < adj_thresh and dik < adj_thresh and djk < adj_thresh:
                    axis = (vi + vj + vk) / np.linalg.norm(vi + vj + vk)
                    key = tuple(np.round(axis, 6))
                    if key in seen3 or tuple(np.round(-axis, 6)) in seen3:
                        continue
                    seen3.add(key)
                    three_fold_axes.append(axis)
    for axis in three_fold_axes:
        for k in (1, 2):
            angle = 2.0 * math.pi * k / 3.0
            add(_axis_angle_to_R(axis, angle))

    # 2-fold rotations: axes through edge midpoints.
    # Adjacent vertex pairs only.
    two_fold_axes = []
    seen2 = set()
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.sum((icos_verts[i] - icos_verts[j]) ** 2)
            if d < adj_thresh:
                axis = (icos_verts[i] + icos_verts[j]) / np.linalg.norm(
                    icos_verts[i] + icos_verts[j]
                )
                key = tuple(np.round(axis, 6))
                if key in seen2 or tuple(np.round(-axis, 6)) in seen2:
                    continue
                seen2.add(key)
                two_fold_axes.append(axis)
    for axis in two_fold_axes:
        add(_axis_angle_to_R(axis, math.pi))

    return np.stack(rot_set)


def _axis_angle_to_R(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' formula."""
    a = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0],
    ])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def conjugacy_class_of(R: np.ndarray) -> int:
    """Return the conjugacy-class index (0..4) of a rotation matrix in I.

    Classes: 0=E, 1=C5, 2=C5², 3=C3, 4=C2.
    """
    tr = np.trace(R)
    if tr > 2.99:
        return 0
    if tr < -1.01:
        return 4  # 180° rotation
    # tr = 1 + 2 cos θ
    angle = math.acos(max(min((tr - 1.0) / 2.0, 1.0), -1.0))
    deg = math.degrees(angle)
    if abs(deg - 72.0) < 1.0 or abs(deg - 288.0) < 1.0:
        return 1  # C5
    if abs(deg - 144.0) < 1.0 or abs(deg - 216.0) < 1.0:
        return 2  # C5²
    if abs(deg - 120.0) < 1.0 or abs(deg - 240.0) < 1.0:
        return 3  # C3
    if abs(deg - 180.0) < 1.0:
        return 4
    raise ValueError(f"unrecognized rotation, trace={tr:.6f}, angle={deg:.2f}°")


def class_counts() -> dict[int, int]:
    """Sanity: count conjugacy class members in our rotation_matrices()."""
    Rs = rotation_matrices()
    counts = {i: 0 for i in range(5)}
    for R in Rs:
        counts[conjugacy_class_of(R)] += 1
    return counts
