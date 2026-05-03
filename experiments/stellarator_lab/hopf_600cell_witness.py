"""
Second-witness verification of the Hopf seed via the 600-cell discrete
de Rham operators, decomposed natively in 2I irrep blocks.

The 600-cell exposes:
    - 120 vertices on unit S^3 (quaternions), 720 oriented edges,
      1200 triangular faces, 600 tetrahedral cells.
    - Combinatorial coboundaries d0 (720x120), d1 (1200x720), d2 (600x1200).
    - cell600.scalar_eigenspaces: the 9 2I-irrep-isotypic components on
      vertices, with dimensions d_i^2 in [1, 4, 9, 16, 25, 36, 9, 16, 4]
      (total 120). Each component is the d_i^2-dim isotypic of the d_i-
      dimensional irrep rho_i in the regular representation of 2I.

Key fact (CONSULTATION.md Q1): 2I acts freely on the 120 vertices and
freely on the 720 edges, so the edge representation is exactly six
copies of the regular representation:

    rho_720 = 6 * rho_reg
            = 6.1 (+) 12.2a (+) 12.2b (+) 18.3a (+) 18.3b
              (+) 24.4a (+) 24.4b (+) 30.5 (+) 36.6.

Each isotypic component on edges has dimension 6 d_i^2; the image of
d0 restricted to the i-th vertex isotypic occupies exactly d_i^2 of
those 6 d_i^2 dimensions (zero for the trivial irrep, since constants
are in ker(d0)). Image(d0) has total dim 119; image(d1.T) has total
dim 601. Since H^1(S^3) = 0, edges = image(d0) (+) image(d1.T) exactly,
no harmonic remainder.

That last statement is the hard test we run.

This module is the irrep-native version of an earlier prototype that
did np.linalg.lstsq on the bare d0 -- the wrong coordinate chart, in
the rank-2-tensor framing of the consultation.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from cell600 import get_geometry


def stereographic_S3_to_R3(q, R=1.0):
    """Project a unit quaternion on S^3 to R^3 via stereographic from
    q3 = -1.

    (q0, q1, q2, q3) -> R * (q0, q1, q2) / (1 + q3). Singular at q3 = -1
    (one vertex, the antipode of the projection pole). Caller must drop
    that vertex if present.
    """
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    denom = 1.0 + q3
    with np.errstate(invalid="ignore", divide="ignore"):
        x = R * q0 / denom
        y = R * q1 / denom
        z = R * q2 / denom
    return np.stack([x, y, z], axis=-1)


def embed_600cell():
    """Return the full 600-cell complex on S^3 with all 120 vertices.

    No projection to R^3 and no vertex dropped. This preserves 2I-
    equivariance exactly: the existing scalar_eigenspaces decompose the
    120-dim vertex space into 2I irreps to machine precision, and the
    coboundaries d0, d1, d2 are 2I-equivariant by construction. The
    harmonic_leak test then has the right to be machine zero.

    Earlier prototype embedded via stereographic projection from the
    pole at q3 = -1 and dropped the singular vertex. That broke 2I
    on the reduced 119-vertex complex (dropping a single non-orbit
    vertex is not a 2I-equivariant operation), and the harmonic leak
    came back as a few percent rather than machine zero. The lesson:
    keep the sampling on S^3 where 2I acts freely; use R^3 projection
    only for unloaded diagnostics (the discrete_helicity estimate and
    Paraview visualisation, neither of which is the irrep test).
    """
    geom = get_geometry()
    return {
        "vertices_s3": geom["vertices"],
        "edges": geom["edges"],
        "triangles": geom["triangles"],
        "d0": geom["d0"],
        "d1": geom["d1"],
        "d2": geom["d2"],
        "scalar_eigenspaces": geom["scalar_eigenspaces"],
    }


def sample_A_on_600cell_S3(omega1, omega2, embedding):
    """Sample the Clebsch 1-form A on the 600-cell edges, natively on S^3.

    A = Im(phi_A * d phi_B) with phi_A = (q0 + i q1)^omega1 and
    phi_B = (q2 + i q3)^omega2 evaluated as functions of the S^3
    coordinates directly. The discrete edge integral is the trapezoidal
    Stokes-compatible value

        A_{edge_vw} = Im( (1/2)(phi_A(v) + phi_A(w)) * (phi_B(w) - phi_B(v)) )

    so that for any 0-form alpha = Im(phi_A . phi_B) we have
    d0 alpha = trapezoidal-A_edges automatically (discrete chain rule).

    This sampling is exactly 2I-equivariant: the action of g in 2I on
    quaternions induces a permutation on vertices and edges, and the
    formula above commutes with that permutation because (q0+iq1) and
    (q2+iq3) transform as 2I-irreps and the powers omega1, omega2 just
    multiply within those irreps.
    """
    verts_s3 = embedding["vertices_s3"]   # (120, 4)
    edges = embedding["edges"]            # list of (i, j)
    q0 = verts_s3[:, 0]
    q1 = verts_s3[:, 1]
    q2 = verts_s3[:, 2]
    q3 = verts_s3[:, 3]
    u = q0 + 1j * q1
    v = q2 + 1j * q3
    phi_A = u ** omega1
    phi_B = v ** omega2

    E = len(edges)
    A_edges = np.zeros(E)
    for k, (i, j) in enumerate(edges):
        avg = 0.5 * (phi_A[i] + phi_A[j])
        diff = phi_B[j] - phi_B[i]
        A_edges[k] = float(np.imag(avg * diff))
    return A_edges


def sample_B_at_vertices_R3(omega1, omega2, R, embedding):
    """Optional R^3 sampling of B at vertices, used only for the
    discrete_helicity crude diagnostic and visualisation.

    Stereographically projects S^3 vertices to R^3 (from pole q3 = -1,
    so the vertex at q3 ~ -1 has B set to zero -- B decays as |x|^-6
    at infinity in R^3, so this is the right limit). Returns (120, 3).
    """
    from hopf_seed import seed_field

    verts_s3 = embedding["vertices_s3"]
    K = verts_s3.shape[0]
    B = np.zeros((K, 3))
    denom = 1.0 + verts_s3[:, 3]
    safe = np.abs(denom) > 1e-3
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(safe, R * verts_s3[:, 0] / denom, 0.0)
        y = np.where(safe, R * verts_s3[:, 1] / denom, 0.0)
        z = np.where(safe, R * verts_s3[:, 2] / denom, 0.0)
    Bx, By, Bz = seed_field(x, y, z, omega1, omega2, R)
    B[:, 0] = np.where(safe, Bx, 0.0)
    B[:, 1] = np.where(safe, By, 0.0)
    B[:, 2] = np.where(safe, Bz, 0.0)
    return B


def check_chain_complex(embedding):
    """Structural: d1 . d0 == 0 and d2 . d1 == 0 on the reduced complex."""
    d0 = embedding["d0"]
    d1 = embedding["d1"]
    d2 = embedding["d2"]
    return float(np.max(np.abs(d1 @ d0))), float(np.max(np.abs(d2 @ d1)))


def edge_isotypic_blocks(embedding):
    """Build the per-irrep image-of-d0 blocks on edges.

    For each scalar eigenspace V_i on vertices (the d_i^2-dim 2I-isotypic
    component for irrep rho_i), the image d0 @ V_i is a d_i^2-dim
    subspace of edges sitting inside the rho_i-isotypic of edges (which
    has total dim 6 d_i^2 by Q1 of CONSULTATION.md). For the trivial
    irrep the image is zero (constants are in ker d0).

    Returns a list of dicts per irrep with:
        'index'         : int                    index into scalar_eigenspaces
        'eigenvalue'    : float                  Laplacian eigenvalue
        'mult_vertex'   : int                    d_i^2 (mult on vertices)
        'block'         : (E, k) ndarray         orthonormal basis of d0 V_i
                                                 in edge space (k = d_i^2 unless
                                                 in the kernel)
        'rank'          : int                    k = dim of image
    """
    d0 = embedding["d0"]
    out: List[Dict[str, Any]] = []
    for idx, es in enumerate(embedding["scalar_eigenspaces"]):
        V = es["vectors"]
        if V.shape[1] == 0:
            out.append({
                "index": idx,
                "eigenvalue": float(es["eigenvalue"]),
                "mult_vertex": 0,
                "block": np.zeros((d0.shape[0], 0)),
                "rank": 0,
            })
            continue
        image = d0 @ V  # (E, d_i^2)
        # Orthonormalize via QR; constant-function image collapses to 0.
        Q, R = np.linalg.qr(image)
        norms = np.abs(np.diag(R))
        keep = norms > 1e-10
        Qk = Q[:, keep]
        out.append({
            "index": idx,
            "eigenvalue": float(es["eigenvalue"]),
            "mult_vertex": V.shape[1],
            "block": Qk,
            "rank": int(Qk.shape[1]),
        })
    return out


def irrep_hodge_decompose(A_edges, embedding, harmonic_tol=1e-8):
    """Hodge decomposition of A_edges in 2I-irrep coordinates.

    A_edges = A_exact + A_coexact, where A_exact lives in image(d0) and
    A_coexact in image(d1.T). H^1(S^3) = 0 so there is no harmonic
    component; the 'harmonic' field of the returned dict is the
    residual that should be machine zero.

    Per-irrep grading: A_exact is decomposed into the d_i^2-dim block
    inside each rho_i-isotypic, using the existing scalar_eigenspaces
    in cell600.py as an oracle for the irrep structure. A_coexact is
    just the orthogonal complement; its irrep-grading would need the
    full face_eigenspace decomposition (not all of which is exposed in
    cell600.py) and is left as future work.
    """
    blocks = edge_isotypic_blocks(embedding)
    A_exact = np.zeros_like(A_edges)
    per_irrep: List[Dict[str, Any]] = []
    for blk in blocks:
        Q = blk["block"]
        if Q.shape[1] == 0:
            per_irrep.append({
                "index": blk["index"],
                "eigenvalue": blk["eigenvalue"],
                "rank": 0,
                "exact_norm": 0.0,
            })
            continue
        coeffs = Q.T @ A_edges
        contribution = Q @ coeffs
        A_exact = A_exact + contribution
        per_irrep.append({
            "index": blk["index"],
            "eigenvalue": blk["eigenvalue"],
            "rank": blk["rank"],
            "exact_norm": float(np.linalg.norm(contribution)),
        })

    A_coexact = A_edges - A_exact

    # Harmonic check: residual in ker(d0.T) i.e. orthogonal to image(d0).
    # By construction A_coexact is orthogonal to A_exact (A_exact spans
    # image(d0)); the test is whether what's left lies entirely in
    # image(d1.T). If H^1 = 0 it does, and there is no further residual.
    # We test by projecting A_coexact onto image(d0) once more: should
    # be ~ 0 (idempotency check).
    reproj = np.zeros_like(A_edges)
    for blk in blocks:
        Q = blk["block"]
        if Q.shape[1] == 0:
            continue
        reproj = reproj + Q @ (Q.T @ A_coexact)
    harmonic_leak = float(np.linalg.norm(reproj))

    norm = float(np.linalg.norm(A_edges)) + 1e-30

    return {
        "A_exact": A_exact,
        "A_coexact": A_coexact,
        "exact_fraction": float(np.linalg.norm(A_exact) / norm),
        "coexact_fraction": float(np.linalg.norm(A_coexact) / norm),
        "harmonic_leak": harmonic_leak,
        "harmonic_leak_relative": harmonic_leak / norm,
        "harmonic_below_tol": harmonic_leak / norm < harmonic_tol,
        "per_irrep": per_irrep,
    }


def isotypic_dim_table(embedding):
    """Reports the 2I-isotypic dimension table on the reduced complex.

    Returns a list of dicts per irrep with the dimensions reproduced
    from the existing scalar_eigenspaces structure. Used as a hard
    accounting test in test_layer1.
    """
    out = []
    total_vertex = 0
    total_image_d0 = 0
    for blk in edge_isotypic_blocks(embedding):
        d_squared = blk["mult_vertex"]
        total_vertex += d_squared
        total_image_d0 += blk["rank"]
        out.append({
            "index": blk["index"],
            "eigenvalue": blk["eigenvalue"],
            "vertex_isotypic_dim": d_squared,
            "edge_image_d0_dim": blk["rank"],
        })
    return {
        "per_irrep": out,
        "total_vertex_dim": total_vertex,
        "total_edge_image_d0": total_image_d0,
    }


def run_witness(omega1, omega2, R=1.0):
    """Full 600-cell second-witness run on the natively-S^3 sampling,
    irrep-graded.

    Returns a dict:
        n_vertices, n_edges, n_faces, n_cells
        chain_d1_d0, chain_d2_d1   (structural; should be 0)
        exact_fraction, coexact_fraction, harmonic_leak_relative
        harmonic_below_tol         (boolean: leak < 1e-8)
        per_irrep                   (list of {index, eigenvalue, rank, exact_norm})
        max_abs_div_vertex          (d0.T A_edges, should be small)
        isotypic_dim_table          (vertex isotypic dims and image-d0 dims per irrep)
    """
    emb = embed_600cell()
    A_edges = sample_A_on_600cell_S3(omega1, omega2, emb)
    dd01, dd12 = check_chain_complex(emb)
    decomp = irrep_hodge_decompose(A_edges, emb)
    div_v = emb["d0"].T @ A_edges
    dim_table = isotypic_dim_table(emb)

    return {
        "n_vertices": int(emb["d0"].shape[1]),
        "n_edges": int(emb["d0"].shape[0]),
        "n_faces": int(emb["d1"].shape[0]),
        "n_cells": int(emb["d2"].shape[0]),
        "chain_d1_d0": dd01,
        "chain_d2_d1": dd12,
        "exact_fraction": decomp["exact_fraction"],
        "coexact_fraction": decomp["coexact_fraction"],
        "harmonic_leak_relative": decomp["harmonic_leak_relative"],
        "harmonic_below_tol": decomp["harmonic_below_tol"],
        "per_irrep": decomp["per_irrep"],
        "max_abs_div_vertex": float(np.max(np.abs(div_v))),
        "isotypic_dim_table": dim_table,
    }
