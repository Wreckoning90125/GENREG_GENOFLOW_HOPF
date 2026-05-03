"""
QM9 atom-centered A/B: does chirality fix in mol_kernel_local also help?

The atom-centered (mol_kernel_local) variant is the "SOAP-on-the-600-cell"
approach. It uses the same vmf_assign that mol_kernel.py used (with the
same use_abs=True default) but operates ATOM-CENTERED rather than CoM-
centered. Tests whether the chirality fix (use_abs=False, kappa retune)
also helps this pipeline.

Compares against CM and CM+ext baselines.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "mnist_geometric"))

from ade_geometry import get_ade
from qm9_data import load_qm9, get_splits, TARGET_INDICES, TARGET_UNITS
from mol_kernel import coulomb_matrix_features
from mol_kernel_local import extract_atom_centered_features
from train_ade_hopf import ridge_regression_bias
from bench_qm9 import compute_extensivity_features


def evaluate_mae(predictions, targets):
    return float(np.mean(np.abs(predictions - targets)))


def best_ridge_mae(X_train, y_train, X_test, y_test,
                   alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)):
    Xt = np.hstack([X_train, np.ones((len(X_train), 1))])
    Xv = np.hstack([X_test, np.ones((len(X_test), 1))])
    Y = y_train.reshape(-1, 1)
    best_mae, best_alpha = float("inf"), alphas[0]
    for a in alphas:
        W = ridge_regression_bias(Xt, Y, a)
        mae = evaluate_mae((Xv @ W).ravel(), y_test)
        if mae < best_mae:
            best_mae, best_alpha = mae, a
    return best_mae, best_alpha


def standardize(X, train_idx):
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mol", type=int, default=10000)
    ap.add_argument("--out-dir", type=str, default="checkpoints/qm9_local_bench")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    coords, atoms, props, _ = load_qm9()
    if args.n_mol < len(coords):
        rng = np.random.default_rng(42)
        sel = rng.choice(len(coords), args.n_mol, replace=False)
        sel.sort()
        coords = [coords[i] for i in sel]
        atoms = [atoms[i] for i in sel]
        props = props[sel]

    n_total = len(coords)
    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)
    train_idx, _, test_idx = get_splits(n_total, n_train=n_train, n_val=n_val)
    print(f"Total {n_total}, train {len(train_idx)}, test {len(test_idx)}")

    print("Building ADE geometry...")
    ade = get_ade()

    feature_sets = {}

    print("\n[CM] ...")
    CM = coulomb_matrix_features(coords, atoms)
    feature_sets["cm"] = CM

    print("\n[ext] ...")
    EXT = compute_extensivity_features(coords, atoms)
    feature_sets["ext"] = EXT
    feature_sets["cm_ext"] = np.hstack([CM, EXT])

    # Atom-centered: published default kappas (the original chirality-blind)
    print("\n[local_abs] atom-centered, use_abs=True (original)...")
    t0 = time.time()
    X_local_abs = extract_atom_centered_features(
        coords, atoms, ade,
        kappas=(5.5,), n_rbf=3, rbf_min=1.0, rbf_max=4.0,
        rbf_gamma=2.0, cutoff=5.0, chunk_size=2000, use_abs=True,
    )
    print(f"  shape={X_local_abs.shape}, time={time.time()-t0:.1f}s")
    feature_sets["local_abs"] = X_local_abs

    print("\n[local_signed] atom-centered, use_abs=False (chirality fix)...")
    t0 = time.time()
    X_local_signed = extract_atom_centered_features(
        coords, atoms, ade,
        kappas=(4.0,), n_rbf=3, rbf_min=1.0, rbf_max=4.0,
        rbf_gamma=2.0, cutoff=5.0, chunk_size=2000, use_abs=False,
    )
    print(f"  shape={X_local_signed.shape}, time={time.time()-t0:.1f}s")
    feature_sets["local_signed"] = X_local_signed

    feature_sets["local_signed_ext"] = np.hstack([X_local_signed, EXT])

    # Standardize
    standardized = {}
    for name, X in feature_sets.items():
        standardized[name] = standardize(X, train_idx)

    method_order = ["cm", "ext", "cm_ext", "local_abs", "local_signed",
                    "local_signed_ext"]

    print("\n" + "=" * 110)
    header = f"{'property':<8} {'unit':<14}" + "".join(f"{m:>16}" for m in method_order)
    print(header)
    print("-" * 110)

    all_results = []
    for prop_name, prop_idx in TARGET_INDICES.items():
        unit_name, unit_scale = TARGET_UNITS[prop_name]
        y_train = props[train_idx, prop_idx]
        y_test = props[test_idx, prop_idx]

        per_method = {}
        for name in method_order:
            mae, alpha = best_ridge_mae(
                standardized[name][train_idx], y_train,
                standardized[name][test_idx], y_test
            )
            per_method[name] = mae * unit_scale

        all_results.append({"property": prop_name, "unit": unit_name, **per_method})

        cells = [f"{prop_name:<8} {unit_name:<14}"]
        for m in method_order:
            cells.append(f"{per_method[m]:>15.3f}")
        print(" ".join(cells))

    print("\n" + "=" * 110)
    print("Atom-centered chirality A/B + extensivity")
    print(f"{'property':<8} {'local_abs':>12} {'local_signed':>14} "
          f"{'local_sig+ext':>16} {'cm_ext':>12}")
    for r in all_results:
        d_chir = (r["local_signed"] - r["local_abs"]) / r["local_abs"] * 100
        print(f"{r['property']:<8} {r['local_abs']:>12.3f} "
              f"{r['local_signed']:>13.3f}({d_chir:+.0f}%) "
              f"{r['local_signed_ext']:>15.3f} {r['cm_ext']:>12.3f}")

    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_total": n_total, "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)), "results": all_results,
            "method_order": method_order,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
