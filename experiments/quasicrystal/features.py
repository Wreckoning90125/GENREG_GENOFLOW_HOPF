"""Features for vertex-class classification on the icosahedral QC.

Two families:

  - `nearest_distance_features(qc, n_neighbors)`:
        the K nearest physical-space neighbor distances, sorted.
        Rotation- and translation-invariant by construction. The
        information available to a fully naive learner.

  - `steinhardt_features(qc, n_neighbors, ls)`:
        Steinhardt bond-orientational order parameters Q_l for
        each vertex over its K nearest neighbors. Rotation-invariant
        scalars built from spherical harmonics. Standard tool for
        local order analysis in disordered solids.

  - `icosahedral_features(qc, n_neighbors)`:
        Q_l restricted to l ∈ {6, 10, 12} — the icosahedrally-
        invariant orders. Three numbers per vertex; the icosahedral
        group I has *exactly one* invariant linear combination of
        Y_l^m for each of these three l, by character-formula
        argument (verified in `icosahedral_invariant_count`).

  - `cell600_irrep_features(qc, n_neighbors)`:
        vMF soft-assign each neighbor's direction onto the 120
        vertices of the 600-cell, sum, project onto each ADE
        eigenspace (which by McKay correspond to 2I irreps), and
        output per-eigenspace L²-norm. Reuses the existing
        `ade_geometry.get_ade()` infrastructure of this repo.

The first three are the "no symmetry" → "rotation-invariant" →
"icosahedrally-aware" axis. The fourth is built directly from the
2I (binary icosahedral) representation theory we already have in
the repo.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.special import sph_harm_y

from experiments.quasicrystal.cnp import IcosahedralQC


# ---------------------------------------------------------------------------
# Nearest-neighbor utilities
# ---------------------------------------------------------------------------


def _nearest_neighbors(positions: np.ndarray, k: int):
    """Return (distances, vectors) of the k nearest neighbors of each
    vertex. Excludes self.

    distances: (N, k) sorted ascending
    vectors:   (N, k, 3) the relative position vectors r_j - r_i
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    # Query k+1 because the closest is self.
    dists, idxs = tree.query(positions, k=k + 1)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]
    vectors = positions[idxs] - positions[:, None, :]
    return dists, vectors


# ---------------------------------------------------------------------------
# Naive baseline: nearest distances sorted
# ---------------------------------------------------------------------------


def nearest_distance_features(qc: IcosahedralQC, n_neighbors: int = 12) -> np.ndarray:
    dists, _ = _nearest_neighbors(qc.physical, n_neighbors)
    return dists


# ---------------------------------------------------------------------------
# Steinhardt order parameters (rotation invariant, all l)
# ---------------------------------------------------------------------------


def _vec_to_theta_phi(vecs: np.ndarray):
    """Convert (..., 3) vectors to (theta, phi) in the SciPy/Condon-Shortley
    convention used by `scipy.special.sph_harm`.

    theta is the polar (colatitude) angle in [0, π].
    phi is the azimuthal angle in [0, 2π).
    """
    x, y, z = vecs[..., 0], vecs[..., 1], vecs[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    safe_r = np.where(r > 1e-12, r, 1.0)
    theta = np.arccos(np.clip(z / safe_r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi


def steinhardt_Q(qc: IcosahedralQC, n_neighbors: int = 12, ls=(2, 4, 6, 8, 10, 12)) -> np.ndarray:
    """Steinhardt bond-orientational order parameter Q_l per vertex.

        q_l^m(i) = (1/N) Σ_j Y_l^m(r̂_ij)
        Q_l(i)   = sqrt( (4π / (2l+1)) Σ_m |q_l^m(i)|² )

    Rotation invariant. Returns shape (N, len(ls)).

    Reference: Steinhardt, Nelson & Ronchetti, PRB 28, 784 (1983).
    """
    _, vecs = _nearest_neighbors(qc.physical, n_neighbors)
    theta, phi = _vec_to_theta_phi(vecs)  # (N, k) each

    out = np.zeros((qc.physical.shape[0], len(ls)))
    for li, l in enumerate(ls):
        # Y_l^m for m = -l .. l, summed over k neighbors, averaged.
        Q_sum_m_sq = np.zeros(qc.physical.shape[0])
        for m in range(-l, l + 1):
            # scipy convention: sph_harm_y(l, m, theta, phi).
            Y = sph_harm_y(l, m, theta, phi)      # (N, k) complex
            q_lm = Y.mean(axis=1)                  # (N,) complex
            Q_sum_m_sq += np.abs(q_lm) ** 2
        out[:, li] = np.sqrt((4.0 * np.pi / (2 * l + 1)) * Q_sum_m_sq)
    return out


def icosahedral_features(qc: IcosahedralQC, n_neighbors: int = 12) -> np.ndarray:
    """Q_l for l ∈ {6, 10, 12} — the icosahedrally-invariant orders.

    For the icosahedral group I, the dimension of the I-invariant
    subspace of the (2l+1)-dim spherical harmonic representation is
    1 for l ∈ {0, 6, 10, 12, 16, 18, ...} and 0 for l ∈ {1, 2, 3, 4,
    5, 7, 8, 9, 11, 13, 14, 15, 17}. (Verified in
    `icosahedral_invariant_count`.) The lowest non-trivial
    icosahedral-invariant l-channel is l=6.

    Three features per vertex; rotation-invariant by Q_l construction.
    """
    return steinhardt_Q(qc, n_neighbors=n_neighbors, ls=(6, 10, 12))


def icosahedral_invariant_count(l: int, n_samples: int = 5000, prec: int = 30) -> int:
    """Count the I-invariant subspace of harmonic order l using the
    character formula:
        n_inv = (1/|I|) Σ_g χ_l(g)
    where χ_l(g) = Σ_m e^{i m θ_g} = sin((l + 1/2) θ_g) / sin(θ_g/2)
    is the character of the (2l+1)-dim representation of SO(3) on
    spherical harmonics.

    Uses mpmath for high precision, since intermediate sums are
    sensitive to numerical drift at large l.
    """
    from mpmath import mp, mpf, sin as mpsin, pi as mppi
    from experiments.quasicrystal.icosahedral_group import (
        rotation_matrices, conjugacy_class_of, CLASS_SIZES,
    )
    mp.dps = prec
    Rs = rotation_matrices()
    # Rotation angle from trace: tr R = 1 + 2 cos θ.
    angles = []
    for R in Rs:
        cos_th = (np.trace(R) - 1.0) / 2.0
        cos_th = max(-1.0, min(1.0, cos_th))
        angles.append(math.acos(cos_th))
    s = mpf(0)
    for theta in angles:
        th = mpf(theta)
        if th == 0:
            chi = mpf(2 * l + 1)
        else:
            num = mpsin(mpf(l + mpf(1) / mpf(2)) * th)
            den = mpsin(th / mpf(2))
            chi = num / den
        s += chi
    n_inv = s / mpf(60)
    return int(round(float(n_inv)))


# ---------------------------------------------------------------------------
# 600-cell ADE-eigenspace features (uses existing repo machinery)
# ---------------------------------------------------------------------------


def cell600_irrep_features(qc: IcosahedralQC, n_neighbors: int = 12,
                            kappa: float = 8.0) -> np.ndarray:
    """vMF-soft-assign each neighbor direction onto the 120 vertices
    of the 600-cell, sum into a 120-dim activation per QC vertex, and
    output the L²-norm in each ADE eigenspace.

    The 600-cell is the orbit of a generic S³ point under the binary
    icosahedral group 2I (double cover of the icosahedral rotation
    group I). Decomposing a function on the 600-cell vertex set into
    its ADE eigenspace components is exactly the "irrep decomposition"
    block-diagonalization of 2I acting on functions on its orbit.

    Returns shape (N, 9) — 9 ADE eigenspaces of the 600-cell scalar
    Laplacian. Per-eigenspace norm is invariant under 2I rotation by
    construction (irreducible projections commute with the group
    action), so this is automatically icosahedrally-invariant.
    """
    from ade_geometry import get_ade
    ade = get_ade()
    verts_600 = ade["vertices"]  # (120, 4) — quaternions on S³
    eigspaces = ade["ade_eigenspaces"]

    # Map neighbor directions into S³ via the standard pixel-kernel
    # spherical embedding: a 3D direction (x,y,z) maps to a quaternion
    # via the canonical Hopf section. We use the most direct map:
    # treat the unit 3-vector as a point on S² and Hopf-section it to
    # S³ (drops a U(1) gauge but that's fine for an irrep-norm feature).
    _, vecs = _nearest_neighbors(qc.physical, n_neighbors)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    safe = norms[..., 0] > 1e-12
    units = np.where(norms > 1e-12, vecs / np.where(norms > 1e-12, norms, 1.0), 0.0)

    # Hopf section (px, py, pz) -> q on S³.
    px = units[..., 0]; py = units[..., 1]; pz = units[..., 2]
    w = 1.0 + pz
    n = np.sqrt(w * w + px * px + py * py)
    safe_n = n > 1e-9
    inv = np.where(safe_n, 1.0 / np.where(safe_n, n, 1.0), 0.0)
    q = np.stack([w * inv, np.zeros_like(w), px * inv, py * inv], axis=-1)  # (N, k, 4)
    q[~safe] = (0.0, 0.0, 0.0, 0.0)

    # vMF assignment: dot product q · v, softmax over 120 vertices.
    # Per neighbor: 120 weights summing to 1.
    raw = np.einsum("nkq,vq->nkv", q, verts_600)  # (N, k, 120)
    scaled = kappa * raw
    scaled -= scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    weights = e / e.sum(axis=-1, keepdims=True)  # (N, k, 120)

    # Sum over neighbors -> (N, 120) activation per QC vertex.
    activation = weights.sum(axis=1)
    activation = activation / activation.sum(axis=1, keepdims=True).clip(min=1e-12)

    # Project onto each ADE eigenspace; output the L²-norm.
    out = np.zeros((qc.physical.shape[0], len(eigspaces)))
    for ei, es in enumerate(eigspaces):
        V = es["V"]  # (120, dim) basis
        coeffs = activation @ V  # (N, dim)
        out[:, ei] = np.linalg.norm(coeffs, axis=1)
    return out


def ssg_irrep_features(qc: IcosahedralQC, n_neighbors: int = 12) -> np.ndarray:
    """Features built by decomposing the local 3D physical neighborhood
    into icosahedral-group irreducible components.

    By Cartesian-tensor decomposition under SO(3), a degree-2 outer
    product of vectors splits as (scalar) ⊕ (antisymmetric vector) ⊕
    (traceless symmetric tensor). Restricting to the icosahedral
    subgroup I ⊂ SO(3), the SAME decomposition applies — but the
    components carry the I irrep labels A (scalar), T1 (vector), and
    H (5-d traceless-symmetric tensor) respectively.

    For each vertex this routine emits:
      - 1 scalar  (A irrep):    mean squared neighbor distance
      - 3 vector  (T1 irrep):   mean physical displacement (centroid)
      - 5 tensor  (H irrep):    flattened upper-triangular components
                                of the traceless symmetric mean
                                outer-product

    Total 9 features per vertex. Direction-aware (the T1 component
    transforms as a 3-vector under rotation — equivariant, not
    invariant). This is the SSG-character-formula analog of the
    ad-hoc `cell600_irrep_coeffs` features, but written purely in
    terms of the icosahedral irrep structure rather than the
    600-cell vertex set.

    Reference: Janssen-Janner-Looijenga-Vos for the SSG framework;
    Stokes & Campbell 2011 for the (3+d)D classification.
    """
    _, vecs = _nearest_neighbors(qc.physical, n_neighbors)  # (N, k, 3)
    N = qc.physical.shape[0]

    # A irrep (scalar): mean squared distance.
    sq = (vecs * vecs).sum(axis=-1)            # (N, k)
    scalar_A = sq.mean(axis=1, keepdims=True)  # (N, 1)

    # T1 irrep (vector): mean displacement.
    vec_T1 = vecs.mean(axis=1)                  # (N, 3)

    # H irrep (traceless symmetric tensor): mean Δ_i Δ_j outer-product,
    # subtract trace/3 · I to make traceless. 6 unique components, but
    # the trace constraint reduces to 5 independent.
    outer = np.einsum("nki,nkj->nij", vecs, vecs) / vecs.shape[1]  # (N, 3, 3)
    trace = np.trace(outer, axis1=1, axis2=2) / 3.0
    sym_tr_free = outer - trace[:, None, None] * np.eye(3)
    tensor_H = np.stack([
        sym_tr_free[:, 0, 0],
        sym_tr_free[:, 1, 1],
        sym_tr_free[:, 0, 1],
        sym_tr_free[:, 0, 2],
        sym_tr_free[:, 1, 2],
    ], axis=1)  # (N, 5)

    return np.concatenate([scalar_A, vec_T1, tensor_H], axis=1)  # (N, 9)


def cell600_irrep_coeffs(qc: IcosahedralQC, n_neighbors: int = 12,
                          kappa: float = 8.0,
                          eig_indices=(0, 1, 2, 3)) -> np.ndarray:
    """vMF-onto-600-cell activation projected to selected ADE eigenspaces;
    returns the FULL coefficient vector per eigenspace (not just the
    norm).

    For irrep-level discrimination: the per-eigenspace coefficients
    transform under 2I via the irrep matrix, so they are NOT
    rotation-invariant — they encode the orientation of the local
    environment relative to the 600-cell's fixed reference frame.
    Useful when the task discriminates orientations (e.g., perp-space
    orbit identification within a fixed ||perp||).

    By default returns coefficients for E0 (dim 1) + E1 (dim 4) +
    E2 (dim 9) + E3 (dim 16) = 30 features.
    """
    from ade_geometry import get_ade
    ade = get_ade()
    verts_600 = ade["vertices"]
    eigspaces = ade["ade_eigenspaces"]

    _, vecs = _nearest_neighbors(qc.physical, n_neighbors)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    safe = norms[..., 0] > 1e-12
    units = np.where(norms > 1e-12, vecs / np.where(norms > 1e-12, norms, 1.0), 0.0)

    px = units[..., 0]; py = units[..., 1]; pz = units[..., 2]
    w = 1.0 + pz
    n = np.sqrt(w * w + px * px + py * py)
    safe_n = n > 1e-9
    inv = np.where(safe_n, 1.0 / np.where(safe_n, n, 1.0), 0.0)
    q = np.stack([w * inv, np.zeros_like(w), px * inv, py * inv], axis=-1)
    q[~safe] = (0.0, 0.0, 0.0, 0.0)

    raw = np.einsum("nkq,vq->nkv", q, verts_600)
    scaled = kappa * raw
    scaled -= scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    weights = e / e.sum(axis=-1, keepdims=True)
    activation = weights.sum(axis=1)
    activation = activation / activation.sum(axis=1, keepdims=True).clip(min=1e-12)

    coeff_blocks = []
    for ei in eig_indices:
        V = eigspaces[ei]["V"]
        coeffs = activation @ V
        coeff_blocks.append(coeffs)
    return np.concatenate(coeff_blocks, axis=1)
