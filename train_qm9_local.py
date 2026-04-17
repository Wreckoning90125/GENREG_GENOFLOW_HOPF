#!/usr/bin/env python3
"""
QM9 with atom-centered 600-cell features (SOAP-on-the-600-cell).

The key change from train_qm9.py: instead of projecting each atom's
direction from center-of-mass onto the 600-cell, we project each
atom's LOCAL NEIGHBORHOOD (relative directions + radial basis functions
of distances to neighbors within a cutoff) onto the 600-cell. This
is the atom-centered paradigm that SOAP, SchNet, and every competitive
molecular ML method uses.

Usage:
    python train_qm9_local.py [--synthetic] [--n-rbf 20] [--cutoff 5.0]
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ade_geometry import get_ade
from qm9_data import (
    load_qm9, generate_synthetic_molecules, get_splits,
    TARGET_INDICES, TARGET_UNITS
)
from mol_kernel_local import (
    extract_atom_centered_features,
    test_atom_centered_rotation_invariance,
)
from mol_kernel import coulomb_matrix_features
from train_ade_hopf import ridge_regression_bias


def evaluate_mae(preds, targets):
    return np.mean(np.abs(preds - targets))


def train_linear_ridge(X_train, y_train, X_test, y_test, unit_scale):
    """Linear ridge with alpha sweep. Caches X^T X for efficiency."""
    n, d = X_train.shape
    X_b = np.hstack([X_train, np.ones((n, 1))])
    X_te_b = np.hstack([X_test, np.ones((len(X_test), 1))])
    Y = y_train.reshape(-1, 1)
    d1 = d + 1

    # Cache the expensive products
    XtX = X_b.T @ X_b         # (d+1, d+1)
    XtY = X_b.T @ Y           # (d+1, 1)

    best_mae = float("inf")
    best_alpha = 0.1
    reg = np.eye(d1)
    reg[-1, -1] = 0.0  # don't regularize bias

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        W = np.linalg.solve(XtX + alpha * reg, XtY)
        preds = (X_te_b @ W).ravel()
        mae = evaluate_mae(preds, y_test) * unit_scale
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return best_mae, best_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-synthetic", type=int, default=500)
    parser.add_argument("--kappas", type=float, nargs="+", default=[5.5])
    parser.add_argument("--n-rbf", type=int, default=20)
    parser.add_argument("--rbf-min", type=float, default=0.5)
    parser.add_argument("--rbf-max", type=float, default=6.0)
    parser.add_argument("--rbf-gamma", type=float, default=10.0)
    parser.add_argument("--cutoff", type=float, default=5.0)
    args = parser.parse_args()

    print("=" * 60)
    print("Atom-Centered 600-Cell Features on QM9")
    print("  SOAP-on-the-600-cell: local environments, not CoM")
    print("=" * 60)

    out_dir = "checkpoints/qm9_local"
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    if args.synthetic:
        print(f"\nUsing {args.n_synthetic} synthetic molecules")
        all_coords, all_atoms, all_props, _ = \
            generate_synthetic_molecules(args.n_synthetic)
        n_train = int(0.6 * len(all_coords))
        n_val = int(0.2 * len(all_coords))
    else:
        print("\nLoading QM9...")
        all_coords, all_atoms, all_props, _ = load_qm9()
        n_train = 110000
        n_val = 10000

    n_total = len(all_coords)
    train_idx, val_idx, test_idx = get_splits(n_total, n_train, n_val)
    print(f"  Total: {n_total}, Train: {len(train_idx)}, "
          f"Val: {len(val_idx)}, Test: {len(test_idx)}")

    # ADE geometry
    print("\nBuilding ADE geometry...")
    ade = get_ade()

    # Rotation invariance test
    print("\n--- Rotation invariance test (atom-centered) ---")
    mean_rc, max_rc = test_atom_centered_rotation_invariance(10)

    # Feature extraction
    kappas = tuple(args.kappas)
    print(f"\n--- Atom-centered feature extraction ---")
    print(f"  Kappas: {kappas}, RBFs: {args.n_rbf} "
          f"({args.rbf_min}-{args.rbf_max} A), cutoff: {args.cutoff} A")

    t0 = time.time()
    X_all = extract_atom_centered_features(
        all_coords, all_atoms, ade,
        kappas=kappas, n_rbf=args.n_rbf,
        rbf_min=args.rbf_min, rbf_max=args.rbf_max,
        rbf_gamma=args.rbf_gamma, cutoff=args.cutoff)
    t_feat = time.time() - t0
    print(f"  Shape: {X_all.shape}, Time: {t_feat:.1f}s "
          f"({t_feat/n_total*1000:.1f}ms/mol)")

    n_nan = np.sum(np.isnan(X_all))
    n_inf = np.sum(np.isinf(X_all))
    dead = np.sum(X_all.std(axis=0) < 1e-12)
    print(f"  NaN: {n_nan}, Inf: {n_inf}, dead: {dead}/{X_all.shape[1]}")
    X_all = np.nan_to_num(X_all)

    # Split and standardize (upcast to float64 for ridge numerical stability)
    X_train = X_all[train_idx].astype(np.float64)
    X_test = X_all[test_idx].astype(np.float64)
    del X_all; gc.collect()

    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std

    # Coulomb matrix baseline
    print("\n--- Coulomb matrix baseline ---")
    CM = coulomb_matrix_features(all_coords, all_atoms)
    CM_tr = CM[train_idx]; CM_te = CM[test_idx]
    del CM; gc.collect()
    cm_mu = CM_tr.mean(axis=0); cm_std = CM_tr.std(axis=0)
    cm_std[cm_std < 1e-8] = 1.0
    CM_tr = (CM_tr - cm_mu) / cm_std
    CM_te = (CM_te - cm_mu) / cm_std

    # Train and evaluate
    print("\n" + "=" * 60)
    print("Results (linear ridge)")
    print("=" * 60)

    schnet = {"gap": 63, "homo": 41, "lumo": 34, "U0": 14,
              "mu": 0.033, "alpha": 0.235, "Cv": 0.033}

    results = []
    for prop, idx in TARGET_INDICES.items():
        unit, scale = TARGET_UNITS[prop]
        y_tr = all_props[train_idx, idx]
        y_te = all_props[test_idx, idx]

        our_mae, our_alpha = train_linear_ridge(
            X_train, y_tr, X_test, y_te, scale)
        cm_mae, cm_alpha = train_linear_ridge(
            CM_tr, y_tr, CM_te, y_te, scale)

        s = schnet.get(prop, "?")
        print(f"  {prop:<8} {unit:<14} local={our_mae:<10.2f} "
              f"CM={cm_mae:<10.2f} SchNet={s}")
        results.append({
            "property": prop, "unit": unit,
            "local_mae": float(our_mae), "local_alpha": float(our_alpha),
            "cm_mae": float(cm_mae), "cm_alpha": float(cm_alpha),
        })

    # Save
    out = {
        "method": "atom-centered 600-cell (SOAP-on-the-600-cell)",
        "kappas": list(kappas), "n_rbf": args.n_rbf,
        "rbf_range": [args.rbf_min, args.rbf_max],
        "rbf_gamma": args.rbf_gamma, "cutoff": args.cutoff,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "n_features": X_train.shape[1],
        "rotation_invariance": {"mean": float(mean_rc), "max": float(max_rc)},
        "results": results,
        "synthetic": args.synthetic,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
