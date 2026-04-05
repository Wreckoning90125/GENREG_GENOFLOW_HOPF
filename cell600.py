# ================================================================
# 600-Cell Geometric Infrastructure
#
# Constructs the 120 vertices of the binary icosahedral group 2I,
# computes the graph Laplacian eigenbasis (Theorem 3) and the
# co-exact 1-form eigenbasis (Theorem 5). All results cached.
#
# Nothing in this file is learned or evolved.
# Everything is determined by the geometry of S³.
# ================================================================

import math
import numpy as np
from itertools import permutations

# Golden ratio
PHI = (1.0 + math.sqrt(5.0)) / 2.0
INV_PHI = 1.0 / PHI  # = PHI - 1


# ================================================================
# Step 0: Construct the 600-cell vertices
# ================================================================

def _even_permutations_4():
    """Return all 12 even permutations of (0,1,2,3)."""
    all_perms = list(permutations(range(4)))
    even = []
    for p in all_perms:
        # Count inversions
        inv = 0
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]:
                    inv += 1
        if inv % 2 == 0:
            even.append(p)
    return even


def build_600_cell():
    """
    Construct the 120 unit quaternions of the binary icosahedral group 2I.

    These are the vertices of the 600-cell in S³:
    - 8 vertices: all permutations of (±1, 0, 0, 0)
    - 16 vertices: (±½, ±½, ±½, ±½)
    - 96 vertices: all even permutations of ½(±1, ±φ, ±1/φ, 0)

    Returns:
        np.ndarray of shape (120, 4) — unit quaternions [w, x, y, z]
    """
    vertices = []

    # --- 8 vertices: permutations of (±1, 0, 0, 0) ---
    for i in range(4):
        for sign in [1.0, -1.0]:
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = sign
            vertices.append(v)

    # --- 16 vertices: (±½, ±½, ±½, ±½) ---
    for s0 in [0.5, -0.5]:
        for s1 in [0.5, -0.5]:
            for s2 in [0.5, -0.5]:
                for s3 in [0.5, -0.5]:
                    vertices.append([s0, s1, s2, s3])

    # --- 96 vertices: even permutations of ½(±1, ±φ, ±1/φ, 0) ---
    base_values = [1.0, PHI, INV_PHI, 0.0]
    even_perms = _even_permutations_4()

    for perm in even_perms:
        # perm assigns each position an index into base_values
        vals = [base_values[perm[i]] for i in range(4)]
        # Apply all sign combinations to nonzero entries
        nonzero_indices = [i for i in range(4) if vals[i] != 0.0]
        n_nonzero = len(nonzero_indices)
        for sign_bits in range(1 << n_nonzero):
            v = vals[:]
            for k, idx in enumerate(nonzero_indices):
                if sign_bits & (1 << k):
                    v[idx] = -v[idx]
            # Multiply by ½
            v = [x * 0.5 for x in v]
            vertices.append(v)

    vertices = np.array(vertices, dtype=np.float64)

    # Remove duplicates (within tolerance)
    unique = []
    for v in vertices:
        is_dup = False
        for u in unique:
            if np.linalg.norm(v - u) < 1e-10:
                is_dup = True
                break
        if not is_dup:
            unique.append(v)
    vertices = np.array(unique, dtype=np.float64)

    # Verify
    assert vertices.shape[0] == 120, f"Expected 120 vertices, got {vertices.shape[0]}"

    # Verify all unit quaternions
    norms = np.linalg.norm(vertices, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12), f"Not all unit quaternions: {norms}"

    return vertices


def verify_group_closure(vertices):
    """
    Verify that the 120 vertices form a group under quaternion multiplication.
    Every product q_i * q_j must equal some q_k in the set.
    """
    n = len(vertices)

    def qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    for i in range(n):
        for j in range(n):
            prod = qmul(vertices[i], vertices[j])
            prod = prod / np.linalg.norm(prod)  # numerical safety
            # Find closest vertex
            dots = np.abs(vertices @ prod)  # |inner product|
            best = np.max(dots)
            if best < 1.0 - 1e-8:
                return False, f"Product of vertex {i} * {j} not in group (best match: {best})"

    return True, "Group closure verified"


def verify_neighbor_count(vertices, adjacency):
    """Verify each vertex has exactly 12 neighbors."""
    degrees = adjacency.sum(axis=1)
    if not np.all(degrees == 12):
        unique, counts = np.unique(degrees, return_counts=True)
        return False, f"Degree distribution: {dict(zip(unique.astype(int), counts))}"
    return True, "All vertices have degree 12"


# ================================================================
# Step 1: Graph Laplacian and Eigenbasis (Theorem 3)
# ================================================================

def build_adjacency(vertices):
    """
    Build adjacency matrix for the 600-cell graph.

    Two vertices are connected if their Euclidean dot product (NOT absolute
    value) exceeds the threshold. The 600-cell is a convex polytope in R⁴;
    edges connect Euclidean nearest neighbors.

    Nearest neighbor dot product: cos(π/5) = φ/2 ≈ 0.809
    Next nearest dot product: 0.5
    Threshold between them: 0.75

    Each vertex connects to exactly 12 neighbors.
    """
    n = len(vertices)
    # Raw dot products — no absolute value
    dots = vertices @ vertices.T

    threshold = 0.75
    adj = (dots > threshold).astype(np.float64)
    np.fill_diagonal(adj, 0.0)

    return adj


def compute_graph_laplacian(adjacency):
    """Compute L = D - A."""
    D = np.diag(adjacency.sum(axis=1))
    return D - adjacency


def compute_eigenbasis(laplacian):
    """
    Compute eigendecomposition of the graph Laplacian.
    Returns eigenvalues and eigenvectors, sorted by eigenvalue.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    # Sort by eigenvalue (should already be sorted by eigh, but be safe)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def group_eigenspaces(eigenvalues, eigenvectors, tol=0.1):
    """
    Group eigenvectors by eigenvalue (within tolerance).
    Returns list of (eigenvalue, eigenvector_matrix) tuples.
    """
    groups = []
    i = 0
    while i < len(eigenvalues):
        lam = eigenvalues[i]
        j = i
        while j < len(eigenvalues) and abs(eigenvalues[j] - lam) < tol:
            j += 1
        group_eigenvalue = np.mean(eigenvalues[i:j])
        group_vectors = eigenvectors[:, i:j]  # shape (120, multiplicity)
        groups.append((group_eigenvalue, group_vectors))
        i = j
    return groups


def verify_theorem3(eigenspace_groups):
    """
    Verify Theorem 3: first 6 eigenspaces have multiplicities [1, 4, 9, 16, 25, 36].
    """
    expected = [1, 4, 9, 16, 25, 36]
    actual = [g[1].shape[1] for g in eigenspace_groups[:6]]

    if actual == expected:
        return True, f"Theorem 3 verified: multiplicities = {actual}"
    else:
        return False, f"Theorem 3 FAILED: expected {expected}, got {actual}"


# ================================================================
# Step 1b: Co-exact 1-form eigenbasis (Theorem 5)
# ================================================================

def build_oriented_edges(adjacency):
    """
    Build oriented edge list from adjacency matrix.
    Only include edge (i,j) where i < j to avoid double-counting.
    Returns list of (i, j) tuples and incidence matrix d0.
    """
    n = adjacency.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                edges.append((i, j))

    n_edges = len(edges)

    # Incidence matrix d0: (n_edges, n_vertices)
    # d0[e, i] = -1, d0[e, j] = +1 for edge e = (i, j)
    d0 = np.zeros((n_edges, n), dtype=np.float64)
    for e_idx, (i, j) in enumerate(edges):
        d0[e_idx, i] = -1.0
        d0[e_idx, j] = 1.0

    return edges, d0


def find_triangles(adjacency):
    """
    Find all triangular faces in the adjacency graph.
    Returns list of (i, j, k) with i < j < k.
    """
    n = adjacency.shape[0]
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] == 0:
                continue
            for k in range(j + 1, n):
                if adjacency[j, k] > 0 and adjacency[i, k] > 0:
                    triangles.append((i, j, k))
    return triangles


def build_face_boundary(edges, triangles):
    """
    Build the face-boundary operator d1: (n_triangles, n_edges).
    For triangle (i, j, k) with oriented boundary (i→j) + (j→k) - (i→k):
    d1[f, e(i,j)] = +1, d1[f, e(j,k)] = +1, d1[f, e(i,k)] = -1
    """
    edge_index = {}
    for e_idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = e_idx

    n_faces = len(triangles)
    n_edges = len(edges)
    d1 = np.zeros((n_faces, n_edges), dtype=np.float64)

    for f_idx, (i, j, k) in enumerate(triangles):
        # Oriented boundary: (i→j) + (j→k) - (i→k)
        # Edge (i,j): i<j, orientation i→j → +1
        if (i, j) in edge_index:
            d1[f_idx, edge_index[(i, j)]] = 1.0
        # Edge (j,k): j<k, orientation j→k → +1
        if (j, k) in edge_index:
            d1[f_idx, edge_index[(j, k)]] = 1.0
        # Edge (i,k): i<k, orientation i→k → -1 (boundary goes k→i, opposite of i→k)
        if (i, k) in edge_index:
            d1[f_idx, edge_index[(i, k)]] = -1.0

    return d1


def compute_coexact_laplacian(d1):
    """
    Compute the co-exact part of the 1-form Hodge Laplacian.
    L_coexact = d1^T d1
    This captures curl modes.
    """
    return d1.T @ d1


def verify_theorem5(eigenspace_groups):
    """
    Verify Theorem 5: co-exact 1-form multiplicities start with [6, 16, 30, 48].
    """
    expected = [6, 16, 30, 48]
    # Skip the zero eigenvalue (harmonic forms)
    nonzero_groups = [(lam, vecs) for lam, vecs in eigenspace_groups if lam > 0.5]
    actual = [g[1].shape[1] for g in nonzero_groups[:4]]

    if actual == expected:
        return True, f"Theorem 5 verified: curl multiplicities = {actual}"
    else:
        return False, f"Theorem 5 check: expected {expected}, got {actual}"


# ================================================================
# Master construction function (cached)
# ================================================================

_CACHE = {}


def get_geometry():
    """
    Compute and cache all geometric infrastructure.
    Returns dict with everything needed for the HopfController.
    """
    if _CACHE:
        return _CACHE

    print("[cell600] Building 600-cell geometry (one-time computation)...")

    # Build vertices
    vertices = build_600_cell()
    print(f"  Vertices: {vertices.shape[0]}")

    # Build adjacency
    adj = build_adjacency(vertices)
    ok, msg = verify_neighbor_count(vertices, adj)
    print(f"  Neighbor check: {msg}")
    assert ok, msg

    # Graph Laplacian eigenbasis (Theorem 3)
    L = compute_graph_laplacian(adj)
    eigenvalues, eigenvectors = compute_eigenbasis(L)
    scalar_groups = group_eigenspaces(eigenvalues, eigenvectors)
    ok, msg = verify_theorem3(scalar_groups)
    print(f"  Theorem 3: {msg}")
    assert ok, msg

    # Report all eigenspace multiplicities
    mults = [g[1].shape[1] for g in scalar_groups]
    print(f"  All scalar multiplicities: {mults}")

    # Co-exact 1-form eigenbasis (Theorem 5)
    edges, d0 = build_oriented_edges(adj)
    print(f"  Edges: {len(edges)}")

    triangles = find_triangles(adj)
    print(f"  Triangles: {len(triangles)}")

    d1 = build_face_boundary(edges, triangles)
    L_coexact = compute_coexact_laplacian(d1)
    curl_evals, curl_evecs = compute_eigenbasis(L_coexact)
    curl_groups = group_eigenspaces(curl_evals, curl_evecs)
    ok, msg = verify_theorem5(curl_groups)
    print(f"  Theorem 5: {msg}")
    if not ok:
        # Report what we got for debugging
        nonzero = [(round(g[0], 4), g[1].shape[1]) for g in curl_groups if g[0] > 0.5]
        print(f"  Curl eigenspaces (nonzero): {nonzero[:10]}")

    # Package results
    _CACHE["vertices"] = vertices                    # (120, 4)
    _CACHE["adjacency"] = adj                        # (120, 120)
    _CACHE["edges"] = edges                          # list of (i, j)
    _CACHE["triangles"] = triangles                  # list of (i, j, k)

    # Scalar eigenspaces: E0..E5 (clean, Theorem 3) + E6..E8 (aliased)
    _CACHE["scalar_eigenspaces"] = []
    for i, (lam, vecs) in enumerate(scalar_groups):
        _CACHE["scalar_eigenspaces"].append({
            "eigenvalue": lam,
            "vectors": vecs,          # (120, multiplicity)
            "multiplicity": vecs.shape[1],
        })

    # Curl eigenspaces (Theorem 5): C1..C4
    nonzero_curl = [(lam, vecs) for lam, vecs in curl_groups if lam > 0.5]
    _CACHE["curl_eigenspaces"] = []
    for i, (lam, vecs) in enumerate(nonzero_curl[:4]):
        _CACHE["curl_eigenspaces"].append({
            "eigenvalue": lam,
            "vectors": vecs,          # (720, multiplicity)
            "multiplicity": vecs.shape[1],
        })

    # Incidence matrix for computing edge signals
    _CACHE["d0"] = d0                                # (720, 120)

    print("[cell600] Geometry construction complete.")
    return _CACHE


# ================================================================
# Standalone verification
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("600-Cell Geometric Infrastructure — Verification")
    print("=" * 60)

    geo = get_geometry()

    print("\n--- Detailed Verification ---")

    # Group closure (slow but definitive)
    print("\nVerifying group closure (120² = 14400 products)...")
    ok, msg = verify_group_closure(geo["vertices"])
    print(f"  {msg}")

    # Scalar eigenspaces
    print("\nScalar eigenspaces (Theorem 3):")
    for i, es in enumerate(geo["scalar_eigenspaces"]):
        print(f"  E{i}: eigenvalue={es['eigenvalue']:.6f}, "
              f"multiplicity={es['multiplicity']}, "
              f"expected={(i+1)**2}")

    # Curl eigenspaces
    print("\nCurl eigenspaces (Theorem 5):")
    expected_curl = [6, 16, 30, 48]
    for i, es in enumerate(geo["curl_eigenspaces"]):
        exp = expected_curl[i] if i < len(expected_curl) else "?"
        print(f"  C{i+1}: eigenvalue={es['eigenvalue']:.6f}, "
              f"multiplicity={es['multiplicity']}, "
              f"expected={exp}")

    # Summary
    total_scalar = sum(es['multiplicity'] for es in geo["scalar_eigenspaces"])
    total_curl = sum(es['multiplicity'] for es in geo["curl_eigenspaces"])
    print(f"\nTotal scalar features (E0-E5): {total_scalar}")
    print(f"Total curl features (C1-C4):   {total_curl}")
    print(f"Combined feature dimension:    {total_scalar + total_curl}")
