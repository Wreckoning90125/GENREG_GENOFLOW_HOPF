#!/usr/bin/env python3
"""
Atom-centered molecular kernel: SOAP-on-the-600-cell.

For each atom i in a molecule, build a local environment descriptor by:
1. Find neighbors j within cutoff radius
2. Compute relative direction r_hat_ij and distance |r_ij|
3. Map r_hat_ij to 600-cell vertices via Hopf section + vMF kernel
4. Weight by radial basis functions of |r_ij| (per atom type channel)
5. Run each (channel, rbf) vertex activation through ADE eigenspace
   decomposition -> 293 features
6. Sum over atoms -> molecular descriptor

This replaces the center-of-mass projection in mol_kernel.py with
atom-centered local environments, following the SOAP philosophy
(Bartok et al. 2013) but using the 600-cell / ADE eigenspace basis
instead of spherical harmonics.
"""

import numpy as np
from cell600 import get_geometry
from hopf_controller import hopf_section

_GEO = None

def _get_geo():
    global _GEO
    if _GEO is None:
        _GEO = get_geometry()
    return _GEO


ATOM_CH = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
N_CH = 5


def gaussian_rbf_centers(r_min=0.5, r_max=6.0, n_rbf=20, gamma=10.0):
    """Return (centers, gamma) for Gaussian radial basis functions."""
    centers = np.linspace(r_min, r_max, n_rbf)
    return centers, gamma


def hopf_section_batch(dirs):
    """Batch Hopf section: (N,3) unit vectors on S^2 -> (N,4) quaternions."""
    px, py, pz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    N = len(px)
    q = np.zeros((N, 4))
    w = 1.0 + pz
    n = np.sqrt(w * w + px * px + py * py)
    ok = n > 1e-10
    q[ok, 0] = w[ok] / n[ok]
    q[ok, 2] = px[ok] / n[ok]
    q[ok, 3] = py[ok] / n[ok]
    q[~ok, 2] = 1.0
    return q


def vmf_assign(quats, kappa, use_abs=True):
    """(N,4) quaternions -> (N,120) soft assignment to 600-cell vertices.

    use_abs: if True (default, original chirality-blind v9 behaviour),
        |q . v| collapses the spinor double cover and identifies
        antipodal vertex pairs. If False, signed q . v preserves the
        full 120-vertex 2I structure (resolves chirality).
    """
    verts = _get_geo()["vertices"]
    raw = quats @ verts.T
    dots = np.abs(raw) if use_abs else raw
    s = kappa * dots
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def atom_centered_activations(coords, atoms, kappa, rbf_centers, rbf_gamma,
                               cutoff=5.0, use_abs=True):
    """Compute atom-centered vertex activations for one molecule.

    For each atom i, find neighbors j within cutoff. For each neighbor,
    compute direction r_hat_ij, soft-assign to 120 vertices via vMF,
    weight by Gaussian RBF of |r_ij| and accumulate per atom-type channel.

    Returns:
        (N_CH, n_rbf, 120) array. Sum over atoms already done —
        each [ch, rbf, :] is the molecular-level descriptor for that
        (channel, radial) combination.
    """
    n_atoms = len(atoms)
    n_rbf = len(rbf_centers)
    result = np.zeros((N_CH, n_rbf, 120))

    if n_atoms == 0:
        return result

    coords = np.asarray(coords, dtype=np.float64)

    for i in range(n_atoms):
        # Displacement vectors from atom i to all others
        diffs = coords - coords[i]  # (n_atoms, 3)
        dists = np.linalg.norm(diffs, axis=1)  # (n_atoms,)

        # Neighbors within cutoff (exclude self)
        mask = (dists > 1e-8) & (dists < cutoff)
        if not np.any(mask):
            continue

        nb_diffs = diffs[mask]
        nb_dists = dists[mask]
        nb_atoms = [atoms[j] for j in range(n_atoms) if mask[j]]

        # Unit direction vectors
        nb_dirs = nb_diffs / nb_dists[:, None]

        # Hopf section + vMF soft assignment
        nb_quats = hopf_section_batch(nb_dirs)
        nb_assign = vmf_assign(nb_quats, kappa, use_abs=use_abs)  # (n_nb, 120)

        # Gaussian RBF weights: (n_nb, n_rbf)
        rbf_weights = np.exp(-rbf_gamma * (nb_dists[:, None] - rbf_centers[None, :]) ** 2)

        # Smooth cutoff (cosine)
        fc = 0.5 * (np.cos(np.pi * nb_dists / cutoff) + 1.0)

        # Accumulate per channel and RBF
        for j_local in range(len(nb_atoms)):
            ch = ATOM_CH.get(nb_atoms[j_local], 1)
            weighted_assign = fc[j_local] * nb_assign[j_local]  # (120,)
            for r in range(n_rbf):
                result[ch, r] += rbf_weights[j_local, r] * weighted_assign

    return result


def extract_atom_centered_features(all_coords, all_atoms, ade,
                                    kappas=(3.0, 5.5, 8.0),
                                    n_rbf=20, rbf_min=0.5, rbf_max=6.0,
                                    rbf_gamma=10.0, cutoff=5.0,
                                    chunk_size=5000, use_abs=True):
    """Full pipeline: molecules -> atom-centered ADE features.

    Processes molecules in chunks to keep peak memory under control.
    For each chunk: compute atom-centered vertex activations, extract
    293 ADE features per (kappa, channel, rbf) combo, write into
    pre-allocated output matrix, free intermediates.

    Returns:
        (N_mol, n_features) float32 array
    """
    from train_ade_hopf import extract_features_from_F

    rbf_centers, gamma = gaussian_rbf_centers(rbf_min, rbf_max, n_rbf, rbf_gamma)
    n_mol = len(all_coords)
    n_combos = len(kappas) * N_CH * n_rbf

    # Probe the feature width once with a small dummy F matrix. Used to
    # be hardcoded at 293 (the v8/v9 design with ~50 cg_v1 projections);
    # after the ade_geometry perms fix it's 535. Safer to query.
    probe_F = np.zeros((1, 120))
    n_feat_per_combo = extract_features_from_F(probe_F, ade).shape[1]
    n_feat = n_feat_per_combo * n_combos
    print(f"  Atom-centered features: {len(kappas)} kappas x {N_CH} channels "
          f"x {n_rbf} RBFs = {n_combos} combos, "
          f"{n_feat_per_combo} feats/combo, expect {n_feat} features/mol")
    print(f"  Chunk size: {chunk_size}, estimated memory: "
          f"{n_mol * n_feat * 4 / 1e9:.1f} GB (float32)")

    X_all = np.zeros((n_mol, n_feat), dtype=np.float32)

    for ki, kappa in enumerate(kappas):
        for start in range(0, n_mol, chunk_size):
            end = min(start + chunk_size, n_mol)
            n_chunk = end - start

            # Compute vertex activations for this chunk
            chunk_acts = []
            for mi in range(start, end):
                act = atom_centered_activations(
                    all_coords[mi], all_atoms[mi], kappa,
                    rbf_centers, gamma, cutoff, use_abs=use_abs)
                chunk_acts.append(act)  # (N_CH, n_rbf, 120)

            # Extract features for each (channel, rbf) combo
            for ch in range(N_CH):
                for r in range(n_rbf):
                    combo_idx = ki * (N_CH * n_rbf) + ch * n_rbf + r
                    col_start = combo_idx * n_feat_per_combo
                    col_end = col_start + n_feat_per_combo

                    F = np.array([chunk_acts[i][ch, r]
                                  for i in range(n_chunk)])  # (n_chunk, 120)
                    feats = extract_features_from_F(F, ade)  # (n_chunk, F)
                    X_all[start:end, col_start:col_end] = feats.astype(np.float32)

            del chunk_acts

            if (start // chunk_size) % 5 == 0 or end == n_mol:
                print(f"    kappa={kappa}: {end}/{n_mol} molecules, "
                      f"{ki * (n_mol // chunk_size + 1) + start // chunk_size + 1} chunks done")

    print(f"  Feature matrix: {X_all.shape}, "
          f"{np.sum(X_all.std(axis=0) < 1e-12)} dead features")
    return X_all


def test_atom_centered_rotation_invariance(n_molecules=10, seed=42):
    """Test rotation invariance of atom-centered features."""
    from ade_geometry import get_ade
    from train_ade_hopf import extract_features_from_F

    rng = np.random.default_rng(seed)
    ade = get_ade()
    rbf_centers, gamma = gaussian_rbf_centers(n_rbf=10)
    kappa = 5.5

    changes = []
    for _ in range(n_molecules):
        n = rng.integers(5, 12)
        coords = rng.standard_normal((n, 3)) * 1.5
        atoms = list(rng.choice(["C", "H", "N", "O"], size=n))

        act_orig = atom_centered_activations(coords, atoms, kappa,
                                              rbf_centers, gamma)
        # Extract features from one channel/rbf for speed
        F_orig = act_orig[1, 0].reshape(1, 120)  # C channel, first RBF
        feat_orig = extract_features_from_F(F_orig, ade).ravel()

        # Random rotation
        M = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(M)
        Q *= np.sign(np.linalg.det(Q))
        rot_coords = coords @ Q.T

        act_rot = atom_centered_activations(rot_coords, atoms, kappa,
                                             rbf_centers, gamma)
        F_rot = act_rot[1, 0].reshape(1, 120)
        feat_rot = extract_features_from_F(F_rot, ade).ravel()

        norm = np.linalg.norm(feat_orig)
        if norm > 1e-10:
            changes.append(np.linalg.norm(feat_orig - feat_rot) / norm)

    mean_rc = np.mean(changes) if changes else 0.0
    max_rc = np.max(changes) if changes else 0.0
    print(f"Atom-centered rotation invariance ({n_molecules} molecules):")
    print(f"  Mean relative change: {mean_rc:.4f}")
    print(f"  Max relative change:  {max_rc:.4f}")
    return mean_rc, max_rc
