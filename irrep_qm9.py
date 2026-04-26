"""
Irrep-native vs flat eigendecomposition of molecular graph Laplacians.

The experiment (CONSULTATION pivot to Option A): does projecting the graph
Laplacian into Aut(graph)-isotypic blocks before eigendecomposition produce
features that beat flat eigendecomposition on QM9 property prediction?

A (flat baseline):    standard eigendecomposition of the molecular graph
                      Laplacian; top-k eigenvalues + per-atom-type vertex
                      amplitudes.
B (irrep treatment):  for each molecule, compute Aut(G), decompose into
                      isotypic blocks (trivial + non-trivial), eigendecompose
                      each block independently, concatenate features.

For trivial Aut (= identity only), B collapses to A by construction. This
is the "sanity check" -- |Delta MAE_trivialaut| ~ 0 confirms the irrep
machinery is correct. For non-trivial Aut, B carries the symmetry
structure explicitly.

Predictions (stated up front so they're falsifiable):
    - Most QM9 molecules have trivial Aut, so the bulk of the test set
      gives identical features. Net ΔMAE_full likely small.
    - Win concentrates on the high-symmetry tail (|Aut| > 1).
    - Possible NULL outcome: framework provides no empirical advantage
      on this task. That's fine; that's a finding, not a failure.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from qm9_data import ATOM_TYPES, ATOM_NUMBERS, N_ATOM_CHANNELS


N_MAX_ATOMS = 30  # QM9 max is 29; pad to 30 for fixed feature shape


def mol_to_graph(coords, atoms, bond_threshold=1.7):
    """Build a labeled graph for the molecule.

    Edge between atoms if Euclidean distance < bond_threshold (Angstroms),
    weighted by exp(-d^2 / 2). Node label = atom symbol so isomorphism
    respects atom types.
    """
    n = len(atoms)
    G = nx.Graph()
    for i, a in enumerate(atoms):
        G.add_node(i, atom=a)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < bond_threshold:
                G.add_edge(i, j, weight=float(np.exp(-d * d / 2.0)))
    return G


def aut_graph(G, max_aut_size=240):
    """Compute graph automorphisms preserving atom labels.

    Returns: list of permutation arrays (each shape (n,)) where perm[i] = j
    means atom i maps to atom j under the automorphism. The first entry is
    always the identity permutation.

    Bounded by max_aut_size to prevent runaway on rare hyper-symmetric cases
    (cap is well above any realistic QM9 molecule's Aut order).
    """
    n = G.number_of_nodes()
    nm = lambda a, b: a.get("atom") == b.get("atom")
    GM = GraphMatcher(G, G, node_match=nm)
    perms = []
    for iso in GM.isomorphisms_iter():
        perm = np.array([iso[i] for i in range(n)], dtype=int)
        perms.append(perm)
        if len(perms) >= max_aut_size:
            break
    if len(perms) == 0:
        # Fallback: identity (shouldn't happen for non-empty graph)
        perms = [np.arange(n, dtype=int)]
    return perms


def graph_laplacian(G):
    """Weighted graph Laplacian L = D - W. Returns (n, n) ndarray."""
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros((0, 0))
    W = nx.to_numpy_array(G, nodelist=range(n), weight="weight")
    D = np.diag(W.sum(axis=1))
    return D - W


def trivial_isotypic_projector(perms, n):
    """Projector onto Aut-invariant vectors:
        P = (1 / |G|) sum_g rho(g)
    where rho(g) is the permutation matrix of g."""
    G = len(perms)
    if G == 1:
        return np.eye(n)
    P = np.zeros((n, n))
    for perm in perms:
        # rho(g): row i, column perm[i] = 1
        # i.e., (rho(g) v)[i] = v[perm^{-1}(i)] -- but for the projector
        # we just need to average the perm matrices, both directions
        # give the same final projector since we're symmetrizing.
        for i in range(n):
            P[perm[i], i] += 1.0
    return P / G


def per_atom_type_amplitudes(eigvecs, atoms):
    """For each eigvec, sum |v_i|^2 over atoms of each type.

    Returns (n_eigvecs, N_ATOM_CHANNELS) array.
    """
    n_atoms = len(atoms)
    n_eigvecs = eigvecs.shape[1] if eigvecs.size > 0 else 0
    amps = np.zeros((n_eigvecs, N_ATOM_CHANNELS), dtype=np.float64)
    if n_atoms == 0 or n_eigvecs == 0:
        return amps
    sq = eigvecs[:n_atoms, :] ** 2  # (n_atoms, n_eigvecs)
    for ch_name, ch_idx in ATOM_TYPES.items():
        mask = np.array([a == ch_name for a in atoms], dtype=bool)
        if mask.any():
            amps[:, ch_idx] = sq[mask, :].sum(axis=0)
    return amps


def _pad_eigvals(eigvals, n_target):
    """Pad/truncate to length n_target with zeros (small-eigenvalue padding)."""
    out = np.zeros(n_target, dtype=np.float64)
    n = min(len(eigvals), n_target)
    out[:n] = eigvals[:n]
    return out


def _pad_amps(amps, n_target):
    """Pad/truncate per-channel amps to (n_target, N_ATOM_CHANNELS)."""
    out = np.zeros((n_target, N_ATOM_CHANNELS), dtype=np.float64)
    n = min(amps.shape[0], n_target)
    out[:n, :] = amps[:n, :]
    return out


def flat_features(L, atoms, n_eig=15):
    """A: standard flat eigendecomposition of the graph Laplacian.

    Returns a (n_eig * (1 + N_ATOM_CHANNELS),) feature vector:
        [eigvals (n_eig)] +
        [per-atom-type vertex amplitudes (n_eig, N_ATOM_CHANNELS) flattened]
    For QM9 with 5 atom types and n_eig = 15, that's 15 * 6 = 90 features.
    """
    n = L.shape[0]
    if n == 0:
        return np.zeros(n_eig * (1 + N_ATOM_CHANNELS))
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals_p = _pad_eigvals(eigvals, n_eig)
    amps = per_atom_type_amplitudes(eigvecs, atoms)
    amps_p = _pad_amps(amps, n_eig)
    return np.concatenate([eigvals_p, amps_p.ravel()])


def isotypic_decomp_features(L, atoms, perms, n_eig=15):
    """Symmetry-resolved channels alone (NOT a strict superset of flat).

    Layout: [eigvals_triv (n_eig)] + [amps_triv (n_eig, N_ATOM_CHANNELS)]
            from the eigendecomposition of P_triv L P_triv, restricted to
            the rank_triv = trace(P_triv) genuine modes. For trivial Aut
            this returns identical content to flat_features (P_triv = I,
            rank_triv = n).

    Used as the "symmetry-aware" channel inside irrep_features (the strict-
    superset version of flat). Standalone, this is informationally weaker
    than flat for non-trivial Aut because it discards the non-trivial
    isotypic block.
    """
    n = L.shape[0]
    if n == 0:
        return np.zeros(n_eig * (1 + N_ATOM_CHANNELS))

    P_triv = trivial_isotypic_projector(perms, n)
    rank_triv = int(np.round(np.trace(P_triv)))
    if rank_triv == 0:
        # Pathological; return zeros
        return np.zeros(n_eig * (1 + N_ATOM_CHANNELS))

    L_triv = P_triv @ L @ P_triv
    L_triv = 0.5 * (L_triv + L_triv.T)
    eigvals_t, eigvecs_t = np.linalg.eigh(L_triv)
    # Keep the top-rank_triv eigenvalues (largest); rest are projector
    # zeros that do not represent real modes. Then sort ascending for
    # consistency with flat_features.
    keep = np.argsort(eigvals_t)[-rank_triv:]
    eigvals_keep = np.sort(eigvals_t[keep])
    eigvecs_keep = eigvecs_t[:, keep[np.argsort(eigvals_t[keep])]]

    eigvals_p = _pad_eigvals(eigvals_keep, n_eig)
    amps = per_atom_type_amplitudes(eigvecs_keep, atoms)
    amps_p = _pad_amps(amps, n_eig)
    return np.concatenate([eigvals_p, amps_p.ravel()])


def irrep_features(L, atoms, perms, n_eig=15):
    """B: STRICT REFINEMENT of flat_features.

    Layout: [flat_features] (+) [non-trivial isotypic channels]

    The flat block is identical to flat_features(L, atoms, n_eig); on top of
    that, an extra n_eig * (1 + N_ATOM_CHANNELS) columns carry the
    eigendecomposition of P_ntr L P_ntr (the non-trivial-isotypic component).

    For trivial Aut (|G| = 1):
        - P_ntr = 0, the non-trivial block is identically zero
        - irrep_features[:flat_dim] == flat_features  (bitwise)
        - irrep_features[flat_dim:] == 0

    For non-trivial Aut:
        - The non-trivial block carries genuine symmetry-resolved
          information beyond what flat captures
    """
    n = L.shape[0]
    flat_dim = n_eig * (1 + N_ATOM_CHANNELS)
    if n == 0:
        return np.zeros(2 * flat_dim)

    flat_block = flat_features(L, atoms, n_eig=n_eig)

    if len(perms) == 1:
        return np.concatenate([flat_block, np.zeros(flat_dim)])

    P_triv = trivial_isotypic_projector(perms, n)
    P_ntr = np.eye(n) - P_triv
    rank_ntr = n - int(np.round(np.trace(P_triv)))

    if rank_ntr == 0:
        return np.concatenate([flat_block, np.zeros(flat_dim)])

    L_ntr = P_ntr @ L @ P_ntr
    L_ntr = 0.5 * (L_ntr + L_ntr.T)
    eigvals_n, eigvecs_n = np.linalg.eigh(L_ntr)
    keep = np.argsort(eigvals_n)[-rank_ntr:]
    eigvals_n_real = np.sort(eigvals_n[keep])
    eigvecs_n_real = eigvecs_n[:, keep[np.argsort(eigvals_n[keep])]]

    eigvals_n_p = _pad_eigvals(eigvals_n_real, n_eig)
    amps_n = per_atom_type_amplitudes(eigvecs_n_real, atoms)
    amps_n_p = _pad_amps(amps_n, n_eig)

    irrep_block = np.concatenate([eigvals_n_p, amps_n_p.ravel()])
    return np.concatenate([flat_block, irrep_block])


def aut_order(G):
    """Compute |Aut(G)| (preserving atom labels). Bounded by max_aut_size."""
    return len(aut_graph(G))


def extract_dataset_features(all_coords, all_atoms, mode="flat", n_eig=15,
                              bond_threshold=1.7, verbose=True):
    """Extract per-molecule feature vectors for the entire dataset.

    Returns:
        X: (N_molecules, n_features) feature matrix.
           For mode='flat': n_features = n_eig * (1 + N_ATOM_CHANNELS).
           For mode='irrep': n_features = 2 * n_eig * (1 + N_ATOM_CHANNELS),
           since irrep is a strict refinement (flat block + symmetry block).
        aut_orders: (N_molecules,) order of each molecule's Aut(graph)
    """
    n_mol = len(all_coords)
    flat_dim = n_eig * (1 + N_ATOM_CHANNELS)
    n_features = 2 * flat_dim if mode == "irrep" else flat_dim
    X = np.zeros((n_mol, n_features))
    aut_orders = np.zeros(n_mol, dtype=int)

    for mol_i in range(n_mol):
        coords = np.asarray(all_coords[mol_i], dtype=np.float64)
        atoms = list(all_atoms[mol_i])
        G = mol_to_graph(coords, atoms, bond_threshold=bond_threshold)
        L = graph_laplacian(G)

        if mode == "flat":
            X[mol_i] = flat_features(L, atoms, n_eig=n_eig)
            # Still compute aut_orders so we can subset the test set
            aut_orders[mol_i] = aut_order(G)
        elif mode == "irrep":
            perms = aut_graph(G)
            aut_orders[mol_i] = len(perms)
            X[mol_i] = irrep_features(L, atoms, perms, n_eig=n_eig)
        else:
            raise ValueError(f"unknown mode {mode}")

        if verbose and (mol_i + 1) % max(1, n_mol // 20) == 0:
            print(f"  [{mode}] {mol_i + 1}/{n_mol} extracted")

    return X, aut_orders
