#!/usr/bin/env python3
"""
Molecular kernel: map 3D molecular structure to 600-cell vertex activations.

This replaces the pixel-to-vertex kernel used for MNIST. Instead of mapping
2D pixel positions to S^3 via stereographic projection, it maps 3D atom
directions (from center of mass) to S^3 via the Hopf section, then soft-
assigns to 120 vertices using the same von-Mises-Fisher kernel.

The downstream pipeline (ADE eigenspace decomposition, Hopf projections,
CG cross products, kernel ridge readout) is unchanged from v10.
"""

import math
import numpy as np

from cell600 import get_geometry
from hopf_controller import hopf_section, poincare_warp_scalar
from qm9_data import ATOM_TYPES, N_ATOM_CHANNELS


# ================================================================
# Geometry cache
# ================================================================

_GEO = None


def _get_geo():
    global _GEO
    if _GEO is None:
        _GEO = get_geometry()
    return _GEO


# ================================================================
# Vectorized Hopf section: S^2 -> S^3
# ================================================================

def hopf_section_batch(points_s2):
    """Lift an array of S^2 points to S^3 via the standard Hopf section.

    Args:
        points_s2: (N, 3) array of unit vectors on S^2

    Returns:
        (N, 4) array of unit quaternions on S^3
    """
    px, py, pz = points_s2[:, 0], points_s2[:, 1], points_s2[:, 2]
    N = len(px)

    quats = np.zeros((N, 4), dtype=np.float64)

    # Standard section: q = [w/n, 0, px/n, py/n] where w = 1 + pz
    w = 1.0 + pz
    n = np.sqrt(w * w + px * px + py * py)

    # Normal case: pz > -0.999 (point not near south pole)
    normal = n > 1e-10
    quats[normal, 0] = w[normal] / n[normal]
    quats[normal, 1] = 0.0
    quats[normal, 2] = px[normal] / n[normal]
    quats[normal, 3] = py[normal] / n[normal]

    # South pole case: pz ~ -1
    south = ~normal
    quats[south, 0] = 0.0
    quats[south, 1] = 0.0
    quats[south, 2] = 1.0
    quats[south, 3] = 0.0

    return quats


# ================================================================
# Von-Mises-Fisher soft assignment to 600-cell vertices
# ================================================================

def vmf_soft_assign(quats, kappa, use_abs=True):
    """Compute soft assignment of quaternions to 600-cell vertices.

    Args:
        quats: (N, 4) array of unit quaternions
        kappa: concentration parameter (higher = sharper assignment)
        use_abs: if True (default, backward-compatible), uses |q . v| so
            antipodal vertices q and -q get identical weight (collapses
            120 vertices to 60 antipodal pairs; throws away the spinor
            double cover, hence chirality information). If False, uses
            signed q . v so the assignment treats q and -q as distinct
            (preserves the full 120-vertex 2I structure and resolves
            chirality).

    Returns:
        (N, 120) soft assignment matrix (rows sum to 1)
    """
    geo = _get_geo()
    vertices = geo["vertices"]  # (120, 4)

    # Quaternion inner product. Absolute value collapses the spinor
    # double cover (q ~ -q under SO(3)) -- this was the original
    # behaviour but it also collapses chirality. Signed version below
    # keeps q and -q distinct, resolving chirality.
    raw = quats @ vertices.T  # (N, 120)
    dots = np.abs(raw) if use_abs else raw

    # Softmax with numerical stability
    scaled = kappa * dots - (kappa * dots).max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


# ================================================================
# Single molecule -> vertex activations
# ================================================================

def molecule_to_vertex_activations(coords, atom_symbols, kappa=5.5, sigma=1.0,
                                    use_abs=True):
    """Convert one molecule to multi-channel 600-cell vertex activations.

    Args:
        coords: (N_atoms, 3) array of atom positions in Angstroms,
                already centered on center of mass.
        atom_symbols: list of element symbols (e.g., ['C', 'H', 'H', ...])
        kappa: angular concentration parameter for von-Mises-Fisher kernel
        sigma: radial scale parameter in Angstroms
        use_abs: if True (default), use |q . v| in the vMF kernel
            (chirality-blind, original behaviour). If False, use signed
            q . v (chirality-resolving; preserves the full 120-vertex
            structure of 2I rather than collapsing to 60 antipodal pairs).

    Returns:
        (N_ATOM_CHANNELS, 120) array of vertex activations, one channel
        per atom type (H, C, N, O, F).
    """
    n_atoms = len(atom_symbols)
    activations = np.zeros((N_ATOM_CHANNELS, 120), dtype=np.float64)

    if n_atoms == 0:
        return activations

    # Compute distances from origin (center of mass)
    distances = np.linalg.norm(coords, axis=1)  # (n_atoms,)

    # Radial weights: Gaussian decay
    radial_weights = np.exp(-distances ** 2 / (2.0 * sigma ** 2))  # (n_atoms,)

    # Separate atoms into directional (|r| > eps) and central (|r| ~ 0)
    eps = 1e-6
    has_direction = distances > eps

    # For atoms with direction: compute S^2 directions and soft-assign
    if np.any(has_direction):
        dir_coords = coords[has_direction]
        dir_dists = distances[has_direction]
        dir_weights = radial_weights[has_direction]

        # Normalize to unit vectors on S^2
        directions = dir_coords / dir_dists[:, np.newaxis]

        # Lift to S^3 and soft-assign to 600-cell vertices
        quats = hopf_section_batch(directions)
        soft_assign = vmf_soft_assign(quats, kappa, use_abs=use_abs)  # (n_dir, 120)

        # Accumulate per channel
        dir_indices = np.where(has_direction)[0]
        for local_i, global_i in enumerate(dir_indices):
            sym = atom_symbols[global_i]
            ch = ATOM_TYPES.get(sym, 1)  # default to C channel
            activations[ch] += dir_weights[local_i] * soft_assign[local_i]

    # For atoms at center: distribute uniformly across all vertices
    central_indices = np.where(~has_direction)[0]
    for global_i in central_indices:
        sym = atom_symbols[global_i]
        ch = ATOM_TYPES.get(sym, 1)
        # Uniform angular contribution, weighted by radial
        activations[ch] += radial_weights[global_i] / 120.0

    return activations


# ================================================================
# Batch: all molecules -> feature matrix
# ================================================================

def batch_vertex_activations(all_coords, all_atoms, kappa=5.5, sigma=1.0,
                              use_abs=True):
    """Compute vertex activations for a batch of molecules.

    Args:
        all_coords: list of (N_atoms, 3) arrays
        all_atoms: list of lists of element symbols
        kappa, sigma: kernel parameters
        use_abs: passed through to molecule_to_vertex_activations.

    Returns:
        List of N_ATOM_CHANNELS (N_molecules, 120) arrays, one per channel.
        Each channel's (i, j) entry is the activation of vertex j for
        molecule i in that atom-type channel.
    """
    n_mol = len(all_coords)
    channel_arrays = [np.zeros((n_mol, 120), dtype=np.float64)
                      for _ in range(N_ATOM_CHANNELS)]

    for mol_i in range(n_mol):
        # Center on center of mass
        coords = all_coords[mol_i].copy()
        coords -= coords.mean(axis=0)

        act = molecule_to_vertex_activations(
            coords, all_atoms[mol_i], kappa, sigma,
            use_abs=use_abs)  # (N_ATOM_CHANNELS, 120)

        for ch in range(N_ATOM_CHANNELS):
            channel_arrays[ch][mol_i] = act[ch]

    return channel_arrays


def extract_molecular_features(all_coords, all_atoms, ade,
                                kappas=(3.0, 5.5, 8.0),
                                sigmas=(0.5, 1.0, 2.0),
                                use_abs=True):
    """Full feature extraction: molecules -> concatenated ADE features.

    For each (kappa, sigma, channel) combination:
      1. Compute vertex activations for all molecules
      2. Run through ADE eigenspace decomposition (293 features)
    Concatenate all blocks.

    Args:
        all_coords: list of (N_atoms, 3) arrays
        all_atoms: list of lists of element symbols
        ade: ADE geometry dict from get_ade()
        kappas: angular concentration parameters
        sigmas: radial scale parameters

    Returns:
        (N_molecules, n_features) array where
        n_features = 293 * len(kappas) * len(sigmas) * N_ATOM_CHANNELS
    """
    # Import the refactored feature extractor
    from train_ade_hopf import extract_features_from_F

    n_mol = len(all_coords)
    all_feature_blocks = []
    n_combos = len(kappas) * len(sigmas) * N_ATOM_CHANNELS
    combo_i = 0

    for kappa in kappas:
        for sigma in sigmas:
            # Get per-channel vertex activations for this (kappa, sigma)
            channel_arrays = batch_vertex_activations(
                all_coords, all_atoms, kappa, sigma, use_abs=use_abs)

            for ch in range(N_ATOM_CHANNELS):
                combo_i += 1
                F = channel_arrays[ch]  # (N_mol, 120)

                # Extract 293 ADE features from vertex activations
                features = extract_features_from_F(F, ade)  # (N_mol, 293)
                all_feature_blocks.append(features)

                if combo_i % 5 == 0 or combo_i == n_combos:
                    print(f"  Feature extraction: {combo_i}/{n_combos} "
                          f"(kappa={kappa}, sigma={sigma}, ch={ch})")

    return np.hstack(all_feature_blocks)


# ================================================================
# Rotation invariance test
# ================================================================

def test_rotation_invariance(n_molecules=10, seed=42):
    """Test that molecular features are approximately invariant under SO(3).

    Generates random molecules, applies random rotations, and measures
    how much the features change. Because the 600-cell has discrete
    icosahedral symmetry (not continuous SO(3)), features are approximately
    but not exactly invariant. The error quantifies the symmetry breaking.
    """
    from qm9_data import generate_synthetic_molecules
    from ade_geometry import get_ade
    from train_ade_hopf import extract_features_from_F

    rng = np.random.default_rng(seed)
    ade = get_ade()
    kappa, sigma = 5.5, 1.0

    max_relative_change = 0.0
    mean_relative_changes = []

    for trial in range(n_molecules):
        # Random molecule
        n_atoms = rng.integers(5, 15)
        coords = rng.standard_normal((n_atoms, 3)) * 1.5
        coords -= coords.mean(axis=0)
        atoms = list(rng.choice(["C", "H", "N", "O"], size=n_atoms))

        # Features without rotation
        act_orig = molecule_to_vertex_activations(coords, atoms, kappa, sigma)
        F_orig = act_orig.reshape(1, -1)[:, :120]  # just C channel for test
        # Actually: use full channels
        features_blocks = []
        for ch in range(N_ATOM_CHANNELS):
            F_ch = act_orig[ch].reshape(1, 120)
            features_blocks.append(extract_features_from_F(F_ch, ade))
        feat_orig = np.concatenate(features_blocks, axis=1).ravel()

        # Random SO(3) rotation via QR decomposition
        M = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(M)
        # Ensure proper rotation (det = +1)
        Q *= np.sign(np.linalg.det(Q))
        rotated_coords = coords @ Q.T

        # Features after rotation
        act_rot = molecule_to_vertex_activations(rotated_coords, atoms,
                                                  kappa, sigma)
        features_blocks_rot = []
        for ch in range(N_ATOM_CHANNELS):
            F_ch = act_rot[ch].reshape(1, 120)
            features_blocks_rot.append(extract_features_from_F(F_ch, ade))
        feat_rot = np.concatenate(features_blocks_rot, axis=1).ravel()

        # Relative change
        norm_orig = np.linalg.norm(feat_orig)
        if norm_orig > 1e-10:
            rel_change = np.linalg.norm(feat_orig - feat_rot) / norm_orig
            max_relative_change = max(max_relative_change, rel_change)
            mean_relative_changes.append(rel_change)

    mean_rc = np.mean(mean_relative_changes) if mean_relative_changes else 0.0
    print(f"Rotation invariance test ({n_molecules} molecules):")
    print(f"  Mean relative feature change under SO(3): {mean_rc:.4f}")
    print(f"  Max relative feature change:              {max_relative_change:.4f}")
    print(f"  (0 = perfect invariance, >0.5 = poor invariance)")

    return mean_rc, max_relative_change


# ================================================================
# Coulomb matrix baseline
# ================================================================

def coulomb_matrix_features(all_coords, all_atoms, max_atoms=29):
    """Compute sorted Coulomb matrix eigenvalues as a classical baseline.

    The Coulomb matrix is a standard molecular fingerprint:
    C_ij = Z_i * Z_j / |r_i - r_j|  for i != j
    C_ii = 0.5 * Z_i^2.4

    Sorted eigenvalues are rotation-invariant by construction.

    Returns:
        (N_molecules, max_atoms) array of sorted eigenvalues (zero-padded)
    """
    from qm9_data import ATOM_NUMBERS

    n_mol = len(all_coords)
    features = np.zeros((n_mol, max_atoms), dtype=np.float64)

    for mol_i in range(n_mol):
        coords = all_coords[mol_i]
        atoms = all_atoms[mol_i]
        n = len(atoms)

        Z = np.array([ATOM_NUMBERS.get(a, 6) for a in atoms], dtype=np.float64)

        C = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            C[i, i] = 0.5 * Z[i] ** 2.4
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist > 1e-10:
                    C[i, j] = Z[i] * Z[j] / dist
                    C[j, i] = C[i, j]

        eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
        features[mol_i, :len(eigvals)] = eigvals

    return features
