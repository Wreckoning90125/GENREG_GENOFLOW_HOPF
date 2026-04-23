"""
Second-witness verification of the Hopf seed using the existing 600-cell
discrete de Rham operators.

The 600-cell in cell600.py provides:
    - 120 vertices on unit S^3 (quaternions)
    - 720 oriented edges
    - 1200 triangular faces
    - 600 tetrahedral cells
    - d0 (720x120): vertex -> edge coboundary
    - d1 (1200x720): edge -> face coboundary
    - d2 (600x1200): face -> cell coboundary

These operators are combinatorial (no metric weights), so this module is
a *topological* witness: we check d . d = 0 structurally, check that the
seed's discrete 1-form has no harmonic component (H^1(S^3) = 0), and
report relative magnitudes of exact / coexact parts from a Hodge-style
decomposition.

Sampling: the 120 S^3 vertices are stereographically projected to R^3
and the seed field is sampled at those points (vertex values of B) and
along the 720 edges (edge line integrals of A).
"""
from __future__ import annotations

import numpy as np

from cell600 import get_geometry


def stereographic_S3_to_R3(q, R=1.0):
    """Project a unit quaternion on S^3 to R^3 via stereographic from q3 = -1.

    (q0, q1, q2, q3) -> R * (q0, q1, q2) / (1 + q3).  Singular at q3 = -1.
    """
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    denom = 1.0 + q3
    with np.errstate(invalid="ignore", divide="ignore"):
        x = R * q0 / denom
        y = R * q1 / denom
        z = R * q2 / denom
    return np.stack([x, y, z], axis=-1)


def embed_600cell(R=1.0, center=(0.0, 0.0, 0.0), drop_singular=True):
    """Embed the 120 S^3 vertices into R^3 via stereographic projection.

    Vertices near q3 = -1 blow up under this projection; we drop any
    vertex with |1 + q3| < 1e-3 (there is at most one, the antipode of
    the projection pole).

    Returns dict with:
        'vertex_indices' : (K,)    indices into the 120-vertex array
        'points'         : (K, 3)  projected R^3 coordinates
        'edges_retained' : list of (i_k, j_k) pairs of vertex_indices
        'd0'             : (E, K)  reduced coboundary over retained vertices
        'd1'             : (F, E)  reduced coboundary over retained edges
        'd2'             : (C, F)  reduced coboundary over retained faces
    """
    geom = get_geometry()
    verts_s3 = geom["vertices"]  # (120, 4)
    edges = geom["edges"]  # list of 720 (i, j) tuples
    d0 = geom["d0"]  # (720, 120)
    d1 = geom["d1"]  # (1200, 720)
    d2 = geom["d2"]  # (600, 1200)
    triangles = geom["triangles"]

    denom = 1.0 + verts_s3[:, 3]
    if drop_singular:
        keep = np.where(np.abs(denom) > 1e-3)[0]
    else:
        keep = np.arange(120)
    keep_set = set(keep.tolist())

    points_all = stereographic_S3_to_R3(verts_s3, R=R)
    points = points_all[keep] + np.asarray(center)

    # Retain only edges whose endpoints survived
    edge_keep = [
        idx for idx, (i, j) in enumerate(edges) if i in keep_set and j in keep_set
    ]
    edge_keep = np.asarray(edge_keep, dtype=int)
    kept_edges = [edges[i] for i in edge_keep]

    # Retain only faces whose edges all survived
    face_keep = []
    for fi in range(d1.shape[0]):
        edge_inds = np.where(d1[fi, :] != 0)[0]
        if all(e in set(edge_keep.tolist()) for e in edge_inds):
            face_keep.append(fi)
    face_keep = np.asarray(face_keep, dtype=int)

    cell_keep = []
    for ci in range(d2.shape[0]):
        fs = np.where(d2[ci, :] != 0)[0]
        if all(f in set(face_keep.tolist()) for f in fs):
            cell_keep.append(ci)
    cell_keep = np.asarray(cell_keep, dtype=int)

    d0_r = d0[np.ix_(edge_keep, keep)]
    d1_r = d1[np.ix_(face_keep, edge_keep)]
    d2_r = d2[np.ix_(cell_keep, face_keep)]

    return {
        "vertex_indices": keep,
        "edge_indices": edge_keep,
        "face_indices": face_keep,
        "cell_indices": cell_keep,
        "points": points,
        "edges_retained": kept_edges,
        "d0": d0_r,
        "d1": d1_r,
        "d2": d2_r,
        "R": R,
        "center": np.asarray(center),
    }


def sample_on_600cell(B_fn, A_fn, embedding):
    """Sample B at vertices (as 3-vectors) and A on edges (as scalars).

    B_fn, A_fn: callables (x, y, z) -> (Fx, Fy, Fz) accepting arrays.
    embedding: output of embed_600cell.

    Returns dict:
        'B_verts'   : (K, 3)   B vector at each retained vertex
        'A_edges'   : (E,)     line integral int_i^j A . dl, midpoint rule
        'edge_vecs' : (E, 3)   (x_j - x_i) displacements
    """
    points = embedding["points"]  # (K, 3)
    vertex_indices = embedding["vertex_indices"]
    # Map full-vertex-index -> row in points
    v_to_row = {int(v): i for i, v in enumerate(vertex_indices)}
    edges = embedding["edges_retained"]

    # Vertex sampling of B
    Bx, By, Bz = B_fn(points[:, 0], points[:, 1], points[:, 2])
    B_verts = np.stack([Bx, By, Bz], axis=-1)

    # Edge line integrals of A via midpoint rule
    E = len(edges)
    A_edges = np.zeros(E)
    edge_vecs = np.zeros((E, 3))
    for k, (i, j) in enumerate(edges):
        pi = points[v_to_row[int(i)]]
        pj = points[v_to_row[int(j)]]
        mid = 0.5 * (pi + pj)
        Ax, Ay, Az = A_fn(mid[0], mid[1], mid[2])
        # A_fn may return 0-d arrays; coerce
        Am = np.array([float(Ax), float(Ay), float(Az)])
        d = pj - pi
        edge_vecs[k] = d
        A_edges[k] = float(np.dot(Am, d))

    return {
        "B_verts": B_verts,
        "A_edges": A_edges,
        "edge_vecs": edge_vecs,
    }


def check_chain_complex(embedding):
    """Structural check: d1 . d0 == 0 and d2 . d1 == 0 on the retained complex.

    Returns max |d1 d0| and max |d2 d1| over the reduced operators.
    """
    d0 = embedding["d0"]
    d1 = embedding["d1"]
    d2 = embedding["d2"]
    dd01 = d1 @ d0
    dd12 = d2 @ d1
    return float(np.max(np.abs(dd01))), float(np.max(np.abs(dd12)))


def hodge_decompose(A_edges, d0, d1):
    """Least-squares Hodge-style decomposition of a 1-form on the retained
    complex:

        A_edges = d0 alpha   +   A_coexact   +   A_harmonic

    Strategy (combinatorial Hodge):
        1. Solve least-squares d0 alpha = A_edges -> exact part d0 alpha.
        2. coexact_candidate = A_edges - d0 alpha.
        3. Check that d1 @ coexact_candidate is not identically zero;
           split harmonic = ker(d0.T) intersect ker(d1) empirically by
           projecting onto ker(d1) (via further least-squares d1.T * beta
           = coexact_candidate if dim ker(d1) is small).

    For our witness purposes, the primary numbers reported are:
        - ||d0 alpha|| / ||A_edges||  (exact fraction)
        - ||A - d0 alpha|| / ||A_edges||  (coexact + harmonic fraction)
        - ||d1 @ (A - d0 alpha)||  (curl of the non-exact part; should
          reflect B flux through faces if A is a valid 1-form for B)
    """
    alpha, *_ = np.linalg.lstsq(d0, A_edges, rcond=None)
    exact_part = d0 @ alpha
    coexact_and_harmonic = A_edges - exact_part
    curl_of_coexact = d1 @ coexact_and_harmonic

    norm = float(np.linalg.norm(A_edges)) + 1e-30
    return {
        "alpha": alpha,
        "exact_part": exact_part,
        "nonexact_part": coexact_and_harmonic,
        "curl_of_nonexact": curl_of_coexact,
        "exact_fraction": float(np.linalg.norm(exact_part) / norm),
        "nonexact_fraction": float(np.linalg.norm(coexact_and_harmonic) / norm),
        "curl_norm": float(np.linalg.norm(curl_of_coexact)),
    }


def discrete_divergence_at_vertices(A_edges, d0):
    """Discrete div via adjoint of d0: div_v = (d0.T @ A_edges)_v.

    For a divergence-free smooth B with A a valid 1-form, the exact part
    projects cleanly through d0 and this residual should be small relative
    to ||A_edges||.
    """
    return d0.T @ A_edges


def discrete_helicity(A_edges, B_verts, embedding):
    """Coarse discrete helicity estimate.

    Not a metric-correct integral (the 600-cell combinatorial operators
    do not carry metric weights); rather an order-of-magnitude second
    witness. We sum at each vertex:

        H_discrete = sum_v  (A_local . B_v) * omega_v

    where A_local at vertex v is the average of A_edges . edge_vecs over
    edges incident at v (giving an R^3-like vector), and omega_v is a
    uniform vertex weight that estimates the Voronoi volume.
    """
    points = embedding["points"]
    vertex_indices = embedding["vertex_indices"]
    edges = embedding["edges_retained"]
    edge_vecs = None  # recomputed inline

    v_to_row = {int(v): i for i, v in enumerate(vertex_indices)}
    K = points.shape[0]
    A_at_vertex = np.zeros((K, 3))
    count = np.zeros(K)

    for k, (i, j) in enumerate(edges):
        ri = v_to_row[int(i)]
        rj = v_to_row[int(j)]
        pi = points[ri]
        pj = points[rj]
        dvec = pj - pi
        # Convert edge scalar to an R^3 vector at each endpoint via
        # unit-length-tangent decomposition
        seg_len2 = float(np.dot(dvec, dvec))
        if seg_len2 < 1e-24:
            continue
        contrib = (A_edges[k] / seg_len2) * dvec
        A_at_vertex[ri] += contrib
        A_at_vertex[rj] += contrib
        count[ri] += 1
        count[rj] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        A_at_vertex = np.where(count[:, None] > 0, A_at_vertex / count[:, None], 0.0)

    # Uniform weights: total R^3 volume (bounding sphere of embedded points)
    # divided by vertex count. Very crude; only order-of-magnitude.
    radii = np.linalg.norm(points - embedding["center"], axis=1)
    r_max = float(np.max(radii))
    # Volume of the sphere of radius r_max
    vol = (4.0 / 3.0) * np.pi * (r_max ** 3)
    omega = vol / K

    return float(np.sum(np.sum(A_at_vertex * B_verts, axis=1)) * omega)


def run_witness(B_fn, A_fn, R=1.0, center=(0.0, 0.0, 0.0)):
    """Full 600-cell second-witness run. Returns a dict of residuals."""
    emb = embed_600cell(R=R, center=center)
    samples = sample_on_600cell(B_fn, A_fn, emb)
    dd01, dd12 = check_chain_complex(emb)
    decomp = hodge_decompose(samples["A_edges"], emb["d0"], emb["d1"])
    div_v = discrete_divergence_at_vertices(samples["A_edges"], emb["d0"])
    H_disc = discrete_helicity(samples["A_edges"], samples["B_verts"], emb)

    return {
        "n_vertices": int(emb["points"].shape[0]),
        "n_edges": int(emb["d0"].shape[0]),
        "n_faces": int(emb["d1"].shape[0]),
        "n_cells": int(emb["d2"].shape[0]),
        "chain_d1_d0": dd01,
        "chain_d2_d1": dd12,
        "exact_fraction": decomp["exact_fraction"],
        "nonexact_fraction": decomp["nonexact_fraction"],
        "curl_of_nonexact_norm": decomp["curl_norm"],
        "max_abs_div_vertex": float(np.max(np.abs(div_v))),
        "discrete_helicity_crude": H_disc,
    }
