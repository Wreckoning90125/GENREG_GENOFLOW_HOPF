"""
Discrete Hopf decagon decomposition of the 600-cell.

Stage 2 of the omnigenity-relaxation lab. The classical Hopf fibration
S^3 -> S^2 with S^1 fibres has a discrete analogue on the 600-cell:
the 120 vertices partition into 12 disjoint great-circle DECAGONS, each
holding 10 vertices at edge-arc pi/5. Each decagon is a left coset of
<g> in 2I, where g is any order-10 element of the binary icosahedral
group.

Concretely, the icosian g = (phi/2, 1/2, 1/(2 phi), 0) is order-10 by
direct verification (its scalar part equals cos(pi/5) and its spatial
magnitude equals sin(pi/5), so g = (cos(pi/5), n . sin(pi/5)) for some
unit vector n, hence g^10 = identity quaternion). Left-multiplication
by g takes any 600-cell vertex to another 600-cell vertex (closure of
2I under multiplication), and adjacent vertices in any orbit
(v, gv, g^2 v, ...) are at angular distance arccos(<v, gv>) =
arccos(g_w) = arccos(phi/2) = pi/5. So adjacent vertices in an orbit
ARE 600-cell edges (the 600-cell's edges are exactly the pairs at
edge-arc pi/5).

This gives:
    * 12 orbits of 10 vertices each, partitioning the 120 vertices.
    * 120 oriented edges-along-fibres (the "C_10 action edges").
    * The remaining 720 - 120 = 600 600-cell edges go BETWEEN fibres.
    * Inter-fibre angular distances form the edge structure of an
      icosahedron on S^2 (the discrete S^2 = S^3 / S^1 quotient).

This is what big-Claude's Q1 / Q3 / Q5 framing was building toward:
the Hopf 1-cochain on edges, which assigns +-1 along fibre edges and
0 between fibres, is the canonical "discrete connection" for the
omnigenity-relaxation lab.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from cell600 import get_geometry
from hopf_controller import qmul


PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _find_vertex(q, verts, tol=1e-10):
    """Index of a unit quaternion in verts (or -1 if absent)."""
    diffs = np.max(np.abs(verts - q), axis=1)
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] < tol else -1


def quaternion_order(q, max_order=120, tol=1e-10):
    """Smallest n >= 1 such that q^n is the identity quaternion. Returns
    -1 if no such n <= max_order exists.
    """
    p = np.array([1.0, 0.0, 0.0, 0.0])
    for n in range(1, max_order + 1):
        p = qmul(p, q)
        if np.max(np.abs(p - np.array([1.0, 0.0, 0.0, 0.0]))) < tol:
            return n
    return -1


def find_order_10_generator():
    """Return (idx, q) for a 600-cell vertex whose quaternion order is
    exactly 10. The Wilson icosian (phi/2, 1/2, 1/(2 phi), 0) and its
    image under sign / coordinate-permutation symmetries are all order
    10; this function returns the first such vertex by index, which in
    cell600.py's ordering is index 48.
    """
    verts = get_geometry()["vertices"]
    for idx in range(120):
        q = verts[idx]
        if abs(q[0] - PHI / 2.0) < 1e-10 and quaternion_order(q) == 10:
            return idx, q.copy()
    raise RuntimeError("no order-10 icosian found among 600-cell vertices")


def hopf_decagon_partition(generator=None):
    """Partition the 120 vertices into 12 left cosets of <g>, where g is
    an order-10 icosian.

    Returns:
        orbits: ndarray (12, 10) of vertex indices, row k = (v, g v,
                g^2 v, ..., g^9 v) for some coset representative v.
        generator_idx: index of g in the cell600 vertex list.
        generator: the order-10 icosian quaternion (4,) ndarray.
    """
    verts = get_geometry()["vertices"]
    if generator is None:
        gen_idx, gen = find_order_10_generator()
    else:
        gen = np.asarray(generator, dtype=float)
        gen_idx = _find_vertex(gen, verts)
        if gen_idx < 0:
            raise ValueError("provided generator is not a 600-cell vertex")
        if quaternion_order(gen) != 10:
            raise ValueError("provided generator does not have quaternion order 10")

    seen = [False] * 120
    orbits = []
    for v0_idx in range(120):
        if seen[v0_idx]:
            continue
        orbit = [v0_idx]
        p = verts[v0_idx].copy()
        for _ in range(9):
            p = qmul(gen, p)
            idx = _find_vertex(p, verts)
            if idx < 0:
                raise RuntimeError("orbit closure broken (numerical)")
            orbit.append(idx)
        if len(set(orbit)) != 10:
            raise RuntimeError(
                f"orbit collapsed (generator order is not 10 on this vertex)"
            )
        for k in orbit:
            seen[k] = True
        orbits.append(orbit)
    if len(orbits) != 12:
        raise RuntimeError(f"expected 12 orbits, got {len(orbits)}")
    return np.asarray(orbits, dtype=int), gen_idx, gen


def fiber_label(orbits):
    """Map each vertex index 0..119 to its fibre index 0..11."""
    labels = np.full(120, -1, dtype=int)
    for fid, orbit in enumerate(orbits):
        for v in orbit:
            labels[v] = fid
    if (labels < 0).any():
        raise RuntimeError("partition is incomplete")
    return labels


def fiber_edges(orbits):
    """Return the 120 oriented "fibre edges" (v_k, v_{k+1}) for k=0..9
    in each orbit, as a list of (i, j) pairs (i = source, j = sink).
    The orientation is along the C_10 action of the generator g.
    """
    out = []
    for orbit in orbits:
        for k in range(10):
            out.append((int(orbit[k]), int(orbit[(k + 1) % 10])))
    return out


def hopf_1_cochain(orbits=None):
    """Build the discrete Hopf 1-cochain A_hopf in R^720.

    For each 600-cell edge {v, w}:
        A_hopf[edge_idx] = +1  if (v, w) appears as a forward fibre edge
                                 (i.e., g . v = w for the chosen
                                 generator) under the cell600 edge
                                 orientation,
                          -1   if the cell600 orientation is the
                                 reverse of the forward fibre direction,
                           0   if the edge is BETWEEN fibres (not within
                                 any decagon).

    By construction this 1-cochain has 120 nonzero entries (12 orbits *
    10 edges each), and integrates to +10 around any forward fibre
    walk and 0 around any closed loop that visits each fibre orbit-
    averaged (the trivial-irrep characteristic).

    Returns: ndarray of shape (720,) with entries in {-1, 0, +1}.
    """
    geom = get_geometry()
    edges = geom["edges"]
    if orbits is None:
        orbits, _, _ = hopf_decagon_partition()

    # Map (i, j) -> oriented edge index k. cell600 stores edges as
    # ordered tuples (i, j) with i < j; our forward fibre might go
    # either way under that ordering.
    edge_index = {}
    for k, (i, j) in enumerate(edges):
        edge_index[(int(i), int(j))] = k
        edge_index[(int(j), int(i))] = k  # tolerate either lookup

    A = np.zeros(720, dtype=float)
    forward_pairs = fiber_edges(orbits)
    for (i, j) in forward_pairs:
        if (i, j) not in edge_index:
            raise RuntimeError(f"fibre pair ({i}, {j}) not in cell600 edge list")
        k = edge_index[(i, j)]
        # Determine sign relative to cell600's stored orientation
        stored = edges[k]
        if int(stored[0]) == i and int(stored[1]) == j:
            A[k] = +1.0
        else:
            A[k] = -1.0
    return A


def inter_fiber_distances(orbits):
    """For each pair (i, j) of fibres, compute the minimum angular
    distance between any vertex in fibre i and any vertex in fibre j.
    Returns a (12, 12) ndarray of arccos values; diagonal is 0.
    """
    verts = get_geometry()["vertices"]
    n = len(orbits)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            best = math.pi
            for v in orbits[i]:
                for w in orbits[j]:
                    dot = float(np.clip(np.dot(verts[v], verts[w]), -1.0, 1.0))
                    arc = math.acos(dot)
                    if arc < best:
                        best = arc
            D[i, j] = best
            D[j, i] = best
    return D


def vertex_action_permutation(g_quaternion):
    """Build the 120-element permutation induced by left-multiplication
    by the quaternion g on the 600-cell vertices. perm[i] = j such that
    g . v_i = v_j.
    """
    verts = get_geometry()["vertices"]
    perm = np.zeros(120, dtype=int)
    for i in range(120):
        p = qmul(g_quaternion, verts[i])
        diffs = np.max(np.abs(verts - p), axis=1)
        idx = int(np.argmin(diffs))
        if diffs[idx] > 1e-10:
            raise RuntimeError("g . v_i is not a 600-cell vertex (numerical)")
        perm[i] = idx
    return perm


def edge_signed_action(g_quaternion):
    """Build the 720x720 signed permutation matrix induced by left-mult
    by g on oriented edges. cell600 stores each edge as (i, j) with
    i < j (canonical orientation); under g the edge maps to (g(i),
    g(j)), which may be in canonical or reversed order. This routine
    returns a signed permutation (+/-1 entries) that maps the cell600
    1-cochain on edges to itself under the g action.
    """
    geom = get_geometry()
    edges = geom["edges"]
    edge_index = {(int(i), int(j)): k for k, (i, j) in enumerate(edges)}
    vertex_perm = vertex_action_permutation(g_quaternion)
    P = np.zeros((720, 720))
    for k, (i, j) in enumerate(edges):
        gi = int(vertex_perm[int(i)])
        gj = int(vertex_perm[int(j)])
        if (gi, gj) in edge_index:
            P[edge_index[(gi, gj)], k] = +1.0
        elif (gj, gi) in edge_index:
            P[edge_index[(gj, gi)], k] = -1.0
        else:
            raise RuntimeError(f"edge image ({gi},{gj}) not in edge list")
    return P


def integrate_along_fiber(A_edges, orbit, edges=None):
    """Sum a 1-cochain along the forward fibre walk
    (v_0 -> v_1 -> ... -> v_9 -> v_0), respecting cell600's
    canonical edge orientation.
    """
    if edges is None:
        edges = get_geometry()["edges"]
    edge_index = {(int(i), int(j)): k for k, (i, j) in enumerate(edges)}
    total = 0.0
    for k in range(10):
        a, b = int(orbit[k]), int(orbit[(k + 1) % 10])
        if (a, b) in edge_index:
            total += A_edges[edge_index[(a, b)]]
        elif (b, a) in edge_index:
            total -= A_edges[edge_index[(b, a)]]
        else:
            raise RuntimeError(f"fibre edge ({a},{b}) missing from edge list")
    return float(total)
