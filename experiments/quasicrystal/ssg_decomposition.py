"""Superspace group (SSG) of the primitive icosahedral quasicrystal.

This module makes explicit the (3+3)-dimensional superspace structure
that the cut-and-project construction in `cnp.py` uses implicitly.
The relevant SSG is the icosahedral rotation group I (order 60)
acting on Z⁶ via signed permutations of the 6 lattice basis vectors —
each basis vector projecting to one of the 6 distinct 5-fold axes
of the icosahedron in physical 3D.

This is the "roll our own" part flagged in `AGENTS.md` for SSGs:
no Python library covers (3+d)D superspace group decompositions
(JANA2020 is GUI-only, Bilbao is web-only, ISODISTORT is web/closed-
source). We construct the 60 6×6 integer matrices, verify they form
a faithful representation of I, and use the character-formula
decomposition (with `mpmath` for high-precision arithmetic over the
golden-ratio number ring ℤ[τ]) to confirm

    ρ_6D = T1 ⊕ T2

i.e., the 6D rep splits exactly as one copy of each 3D irrep of I.
This is what makes the cut-and-project a clean physical/perp split:
physical 3D carries T1, perp 3D carries T2 (or vice versa,
depending on Galois conjugation convention).

References:
  - Elser, "Indexing problems in quasicrystal diffraction" (PRB 32,
    1985) — the 6D icosahedral lattice formulation.
  - Janssen, Janner, Looijenga-Vos, "Incommensurate and commensurate
    modulated structures", Int. Tables for Crystallography Vol. C.
  - Stokes, Campbell, van Smaalen, "Generation of (3+d)-dimensional
    superspace groups for incommensurate crystals", Acta Cryst A67
    (2011), 45-55.
"""

from __future__ import annotations

import math
import numpy as np

from experiments.quasicrystal.cnp import _projection_matrices, TAU
from experiments.quasicrystal.icosahedral_group import (
    rotation_matrices,
    conjugacy_class_of,
    character_table_mp,
    CLASS_SIZES,
    IRREP_NAMES,
    IRREP_DIMS,
)


def ssg_rotation_matrices() -> np.ndarray:
    """The 60 6×6 integer SSG matrices for icosahedral I acting on Z⁶.

    Each R ∈ I (3×3 rotation) permutes the 6 columns of the 6×6
    orthogonal matrix M = [P_phys; P_perp] (with possible signs).
    Specifically, since R acts diagonally on the 6D space split as
    physical ⊕ perp,

        R · v_phys^i = ±v_phys^{σ(i)}
        R · v_perp^i = ±v_perp^{σ(i)}

    where σ is the same signed permutation for both halves (because
    the 6D rep is irreducible-by-irreducible and R commutes with the
    splitting). The resulting 6×6 integer matrix is this signed
    permutation in the original Z⁶ basis.

    Returns (60, 6, 6) integer array. We extract them by direct
    matching of R·P_phys_columns to P_phys_columns (up to sign).
    """
    P_phys, P_perp = _projection_matrices()
    Rs_3d = rotation_matrices()
    # 6 column vectors of P_phys are the 6 physical-space 5-fold axes
    # (each of length 1/√2). For each R ∈ I, we match R · col_i to ±col_j.
    cols_phys = P_phys.T  # (6, 3): each row is a physical-space basis image
    SSG = []
    for R in Rs_3d:
        rotated = (R @ P_phys).T  # (6, 3): each row is R · col_i
        M_int = np.zeros((6, 6), dtype=np.int32)
        for i in range(6):
            # find j and sign s.t. rotated[i] ≈ s * cols_phys[j]
            best_j = -1
            best_sign = 0
            for j in range(6):
                for s in (+1, -1):
                    if np.allclose(rotated[i], s * cols_phys[j], atol=1e-9):
                        best_j = j
                        best_sign = s
                        break
                if best_j >= 0:
                    break
            if best_j < 0:
                raise RuntimeError(
                    f"could not match column {i} of rotated P_phys; "
                    f"icosahedral lattice basis violated"
                )
            # M_int has entry +1 or -1 at (i, j)? Convention: we want
            # (R · P_phys)_col_i = sum_j P_phys_col_j · M_int[j, i].
            # Working in column-vector convention: R · P_phys = P_phys @ M_int.
            M_int[best_j, i] = best_sign
        SSG.append(M_int)
    return np.stack(SSG)


def verify_group_closure(SSG: np.ndarray, tol: int = 0) -> bool:
    """The 60 matrices should be closed under multiplication.

    Verified by checking that for every pair (M_a, M_b), the product
    M_a @ M_b is present in the set, up to the given integer tolerance
    (default 0 — exact for integer matrices).
    """
    matset = {M.tobytes(): k for k, M in enumerate(SSG)}
    for a in range(len(SSG)):
        for b in range(len(SSG)):
            prod = SSG[a] @ SSG[b]
            if prod.tobytes() not in matset:
                return False
    return True


def verify_acts_on_phys_as_rotation(SSG: np.ndarray) -> np.ndarray:
    """For each SSG element M, compute the induced action on physical
    3D and verify it's a proper rotation.

    P_phys @ M maps 6D → 3D-physical, factoring R through
    R @ P_phys = P_phys @ M.

    Returns array of (det, frobenius_residual) per SSG element.
    """
    P_phys, P_perp = _projection_matrices()
    R3d = rotation_matrices()
    out = np.zeros((len(SSG), 2))
    for k in range(len(SSG)):
        M = SSG[k].astype(np.float64)
        # P_phys @ P_phys^T = I_3 (top-left block of the orthogonal 6×6
        # M = [P_phys; P_perp] equals identity), so the induced action
        # in physical space is R = P_phys @ M @ P_phys^T.
        R_recovered = P_phys @ M @ P_phys.T
        # Best match against the 60 3D rotations.
        best = min(range(len(R3d)),
                   key=lambda i: np.max(np.abs(R_recovered - R3d[i])))
        residual = float(np.max(np.abs(R_recovered - R3d[best])))
        out[k, 0] = float(np.linalg.det(R_recovered))
        out[k, 1] = residual
    return out


def char_on_6d_rep(SSG: np.ndarray) -> dict:
    """Compute the character χ_6D(g) = trace of the 6×6 SSG matrix
    for each element, grouped by conjugacy class.

    Returns {class_idx: χ_6D(class) value}. By the character formula,
    this should equal χ_T1 + χ_T2 of I, since the 6D representation
    decomposes as T1 ⊕ T2.
    """
    R3d = rotation_matrices()
    by_class = {i: [] for i in range(5)}
    for k in range(len(SSG)):
        R = R3d[k]
        cls = conjugacy_class_of(R)
        by_class[cls].append(int(np.trace(SSG[k])))
    # Every element of the same class should have the same character
    # (it's a class function).
    out = {}
    for cls, vals in by_class.items():
        v0 = vals[0]
        assert all(v == v0 for v in vals), (
            f"χ_6D not a class function on class {cls}: {vals}"
        )
        out[cls] = v0
    return out


def decompose_6d_rep_via_mpmath(SSG: np.ndarray, prec: int = 30) -> dict:
    """Decompose the 6D representation into I irreps using the
    character formula at `prec` decimal digits via mpmath.

        a_i = (1/|G|) Σ_g χ_i(g) · χ_6D(g)

    where χ_i is the character of irrep i (A, T1, T2, G, H).
    Returns {irrep_name: multiplicity} as integers (rounded; mismatch
    > 1e-25 raises).
    """
    from mpmath import mp, mpf
    mp.dps = prec
    chi_table = character_table_mp(prec)
    chi_6d_by_cls = char_on_6d_rep(SSG)
    multiplicities = {}
    G = mpf(60)
    for i, name in enumerate(IRREP_NAMES):
        s = mpf(0)
        for cls in range(5):
            size = mpf(CLASS_SIZES[cls])
            s += size * chi_table[i][cls] * mpf(chi_6d_by_cls[cls])
        a = s / G
        rounded = int(round(float(a)))
        residual = float(abs(a - mpf(rounded)))
        if residual > 1e-25:
            raise ValueError(
                f"non-integer multiplicity for irrep {name}: "
                f"a = {a}, residual {residual}"
            )
        multiplicities[name] = rounded
    return multiplicities


def project_function_onto_irrep(f: np.ndarray, SSG: np.ndarray,
                                  irrep_name: str, prec: int = 30) -> np.ndarray:
    """Project a function f: Z⁶ → ℝ onto its irrep-`irrep_name` component.

    Uses the projection operator
        P_i = (d_i / |G|) Σ_g χ_i(g)^* · ρ(g)
    where ρ(g) is the 6D rep matrix and d_i is the irrep dimension.

    For irrep `irrep_name` chosen to be the trivial irrep ('A'), this
    is the standard averaging operator, projecting onto SSG-invariants.

    `f` shape (N, 6) where each row is a 6D point on which the SSG
    acts. Returns array of the same shape with f projected onto the
    irrep.
    """
    from mpmath import mp, mpf
    mp.dps = prec
    chi_table = character_table_mp(prec)
    name_to_idx = {n: i for i, n in enumerate(IRREP_NAMES)}
    i = name_to_idx[irrep_name]
    R3d = rotation_matrices()
    G = 60
    d_i = IRREP_DIMS[i]

    proj = np.zeros_like(f, dtype=np.float64)
    for k in range(len(SSG)):
        cls = conjugacy_class_of(R3d[k])
        chi_val = float(chi_table[i][cls])
        proj += chi_val * (f @ SSG[k].astype(np.float64).T)
    proj *= d_i / G
    return proj
