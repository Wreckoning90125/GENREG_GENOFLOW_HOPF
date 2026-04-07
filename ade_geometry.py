# ================================================================
# ADE Geometric Infrastructure for v8 Hopf Controller
#
# Builds on cell600.py to add:
# 1. Group action permutations for all 120 elements of 2I
# 2. Irrep copy decomposition for each eigenspace (orbit method)
# 3. CG cross-product matrices (character projector to V_1)
# 4. All results cached for reuse in forward pass
#
# Nothing in this file is learned. Everything is determined by
# the representation theory of 2I and the McKay correspondence.
# ================================================================

import numpy as np
from cell600 import get_geometry


def _qmul(a, b):
    """Hamilton quaternion product."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


_CACHE = {}


def get_ade():
    """Build and cache all ADE infrastructure."""
    if _CACHE:
        return _CACHE

    print("[ade] Building ADE infrastructure (one-time)...")
    geo = get_geometry()
    verts = geo["vertices"]
    es_list = geo["scalar_eigenspaces"]
    dims = geo["irrep_dims"]
    N = len(verts)

    # ---- 1. Group action permutations ----
    # perms[g][i] = index of g * v_i in vertex set
    print("  Group action permutations...")
    perms = np.zeros((N, N), dtype=np.int32)
    for g in range(N):
        for i in range(N):
            p = _qmul(verts[g], verts[i])
            p /= np.linalg.norm(p)
            perms[g, i] = np.argmax(np.abs(verts @ p))

    # ---- 2. Process each eigenspace ----
    print("  Eigenspace decomposition and CG matrices...")
    np.random.seed(42)
    ade_es = []

    for idx, es in enumerate(es_list):
        V = es["vectors"]            # (120, d^2) eigenvector matrix
        d = dims[idx]
        d2 = es["multiplicity"]      # = d^2

        # Build restricted representation matrices: M_g = V^T @ V[perm_g]
        # (L_g f)(h) = f(g^{-1}h), but perm stores g*v_i, so we need
        # the inverse permutation for each g.
        # Actually: V^T @ (permuted V) directly gives the rep matrix.
        # For left action L_g: (L_g V)[i,:] = V[sigma_g^{-1}(i), :]
        # where sigma_g(i) = index of g*v_i. So sigma_g^{-1}(i) = index of g^{-1}*v_i.
        # We need inv_perms.
        inv_perms = np.zeros_like(perms)
        for g in range(N):
            inv_perms[g, perms[g]] = np.arange(N)

        M_all = np.zeros((N, d2, d2))
        for g in range(N):
            M_all[g] = V.T @ V[inv_perms[g]]

        # Decompose into irrep copies via orbit method
        copies = _orbit_decompose(M_all, d, d2, N)

        # Per-copy restricted representation matrices (d_copy x d_copy)
        copy_reps = []
        copy_dims = []
        for cb in copies:
            dc = cb.shape[1]
            R = np.zeros((N, dc, dc))
            for g in range(N):
                R[g] = cb.T @ M_all[g] @ cb
            copy_reps.append(R)
            copy_dims.append(dc)

        # CG projection to V_1 (spin-1, character chi_1(g) = 4w^2 - 1)
        # Only compute if we have at least 2 copies
        cg_v1 = None
        if len(copies) >= 2:
            cg_v1 = _cg_projector_v1(copy_reps[0], verts, N)

        ade_es.append({
            "V": V,                    # (120, d^2) eigenspace basis
            "copies": copies,          # list of (d^2, d_copy) basis matrices
            "copy_dims": copy_dims,    # dimension of each copy
            "copy_reps": copy_reps,    # list of (N, d_copy, d_copy) arrays
            "cg_v1": cg_v1,            # (k, d_copy^2) projection or None
            "d": d,
            "d2": d2,
        })

        cg_dim = cg_v1.shape[0] if cg_v1 is not None else 0
        n_copies = len(copies)
        cdims = [c.shape[1] for c in copies]
        print(f"    E{idx}: d={d}, copies={n_copies} (dims {cdims}), "
              f"V_1 proj dim={cg_dim}")

    # ---- 3. Cache everything ----
    _CACHE.update(geo)
    _CACHE["ade_eigenspaces"] = ade_es
    _CACHE["inv_perms"] = inv_perms

    # Precompute eigenspace-to-E8-node reverse mapping
    e2e8 = geo["eigenspace_to_e8"]
    e8_to_es = {}
    for es_idx, node in enumerate(e2e8):
        e8_to_es[node] = es_idx
    _CACHE["e8_to_eigenspace"] = e8_to_es

    print("[ade] ADE infrastructure complete.")
    return _CACHE


def _orbit_decompose(M_all, d, d2, N):
    """
    Decompose the d^2-dimensional representation into d copies of
    d-dimensional irrep subspaces using the group orbit method.

    The orbit of a generic vector typically spans MORE than d dims
    (because the vector has components in multiple copies). We truncate
    to the top d singular vectors, which isolate one copy. The remaining
    dimensions are left for subsequent copies.
    """
    if d2 == 1:
        return [np.eye(1)]

    copies = []
    proj_found = np.zeros((d2, d2))

    for copy_idx in range(d):
        remaining = d2 - d * copy_idx
        if remaining < d:
            break

        comp = np.eye(d2) - proj_found

        # Try a few random starts
        best_sv_sum = -1
        best_Vt = None
        for _ in range(5):
            v0 = comp @ np.random.randn(d2)
            n = np.linalg.norm(v0)
            if n < 1e-10:
                continue
            v0 /= n

            # Group orbit projected to complement
            orbit = np.stack([M_all[g] @ v0 for g in range(N)]) @ comp
            _, S, Vt = np.linalg.svd(orbit, full_matrices=False)

            # Take top d singular vectors as one copy
            sv_sum = np.sum(S[:d])
            if sv_sum > best_sv_sum:
                best_sv_sum = sv_sum
                best_Vt = Vt

        if best_Vt is None:
            break

        basis = best_Vt[:d].T  # (d2, d)
        Q, _ = np.linalg.qr(basis)
        cb = Q[:, :d]
        copies.append(cb)
        proj_found += cb @ cb.T

    return copies


def _cg_projector_v1(copy_rep, verts, N):
    """
    Compute the CG projection from V_j tensor V_j to V_1 (spin-1)
    using the character projector method.

    P_1 = (3/|G|) * sum_g chi_1(g) * kron(rho(g), rho(g))

    Returns (k, d_copy^2) matrix, or None if no V_1 component.
    """
    dc = copy_rep.shape[1]  # copy dimension
    dc2 = dc * dc
    P = np.zeros((dc2, dc2))

    for g in range(N):
        w = verts[g, 0]
        chi1 = 4 * w * w - 1  # spin-1 character
        Mg = copy_rep[g]       # (dc, dc)
        P += chi1 * np.kron(Mg, Mg)

    P *= 3.0 / N

    # P should be a projector (idempotent). Extract its range.
    U, S, _ = np.linalg.svd(P, full_matrices=True)
    k = int(np.sum(S > 0.1))

    if k == 0:
        return None

    # Return basis for V_1 component: (k, dc^2) projection matrix
    return U[:, :k].T


# ================================================================
# Convenience: precompute feature extraction matrices
# ================================================================

def count_features():
    """Count total features produced by the ADE v8 architecture."""
    ade = get_ade()
    ade_es = ade["ade_eigenspaces"]
    e8_edges = ade["e8_edges"]

    n_feat = 0
    for idx, aes in enumerate(ade_es):
        d = aes["d"]
        copies = aes["copies"]
        cg = aes["cg_v1"]

        if d <= 2:
            # Hopf: 3 S2 + 1 mag
            n_feat += 4
        else:
            # Per-copy magnitudes
            n_feat += len(copies)
            # Pairwise CG cross products
            if cg is not None:
                n_pairs = len(copies) * (len(copies) - 1) // 2
                n_feat += n_pairs * cg.shape[0]

    # E8 edge features
    n_feat += len(e8_edges) * 2

    return n_feat


if __name__ == "__main__":
    ade = get_ade()
    n = count_features()
    print(f"\nTotal ADE v8 features: {n}")
