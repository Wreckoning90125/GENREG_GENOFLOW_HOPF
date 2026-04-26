"""
Chirality-resolving 600-cell kernel A/B vs the existing chirality-blind one.

The existing mol_kernel.py uses |q . v| in the von-Mises-Fisher soft
assignment, which collapses the spinor double cover (q ~ -q under SO(3))
and therefore identifies antipodal vertices of the 600-cell. This treats
the 120-vertex 2I structure as if it were the 60-vertex I. The collapse
discards chirality information.

The fix is one parameter (use_abs=False). This script runs both passes
on the same QM9 subset, same train/test split, same hyperparameters, and
reports per-property MAE comparison.

Predictions (stated falsifiable up front):
    mu        : 5-15% drop. mu depends on 3D orientation; chirality
                ceiling has been observed in every other variant we
                tested. If mu doesn't move, the chirality story is wrong.
    alpha     : 2-5% drop. polarizability is also orientation-dependent.
    others    : <= 1% movement (electronic properties dominated by
                connectivity).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ade_geometry import get_ade
from qm9_data import (
    load_qm9, get_splits, TARGET_INDICES, TARGET_UNITS,
)
from mol_kernel import extract_molecular_features
from train_ade_hopf import ridge_regression_bias


def evaluate_mae(predictions, targets):
    return float(np.mean(np.abs(predictions - targets)))


def best_ridge_mae(X_train, y_train, X_test, y_test,
                   alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)):
    n_train = X_train.shape[0]
    Xt = np.hstack([X_train, np.ones((n_train, 1))])
    Xv = np.hstack([X_test, np.ones((len(X_test), 1))])
    Y = y_train.reshape(-1, 1)
    best_mae = float("inf")
    best_alpha = alphas[0]
    for alpha in alphas:
        W = ridge_regression_bias(Xt, Y, alpha)
        mae = evaluate_mae((Xv @ W).ravel(), y_test)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return best_mae, best_alpha


def run_pass(coords, atoms, ade, use_abs, kappas, sigmas):
    label = "abs (chirality-blind)" if use_abs else "signed (chirality-resolving)"
    print(f"\n=== Extracting features: {label} ===")
    t0 = time.time()
    X = extract_molecular_features(
        coords, atoms, ade, kappas=kappas, sigmas=sigmas, use_abs=use_abs
    )
    print(f"  shape={X.shape}, time={time.time() - t0:.1f}s")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mol", type=int, default=15000,
                    help="Subsample to N molecules (default 15000)")
    ap.add_argument("--kappas", type=float, nargs="+", default=[5.5])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--out-dir", type=str, default="checkpoints/qm9_chirality")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading QM9...")
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
    train_idx, val_idx, test_idx = get_splits(n_total, n_train=n_train, n_val=n_val)
    print(f"Total {n_total}, train {len(train_idx)}, test {len(test_idx)}")

    print("Building ADE geometry...")
    ade = get_ade()

    kappas = tuple(args.kappas)
    sigmas = tuple(args.sigmas)

    # === Pass A: existing (use_abs=True, chirality-blind) ===
    X_abs = run_pass(coords, atoms, ade, use_abs=True,
                     kappas=kappas, sigmas=sigmas)

    # === Pass B: new (use_abs=False, chirality-resolving) ===
    X_signed = run_pass(coords, atoms, ade, use_abs=False,
                        kappas=kappas, sigmas=sigmas)

    # Sanity: features differ! (if they don't, the use_abs flag isn't wired)
    diff = float(np.max(np.abs(X_abs - X_signed)))
    print(f"\nSanity: max|X_abs - X_signed| = {diff:.4e} (must be > 0; if 0, "
          f"use_abs flag not wired)")
    assert diff > 1e-6, "BUG: X_abs and X_signed are identical -- flag not wired"

    # Standardize each feature matrix on its own train mean/std
    def standardize(X, train_idx):
        mean = X[train_idx].mean(axis=0)
        std = X[train_idx].std(axis=0)
        std[std < 1e-8] = 1.0
        return (X - mean) / std

    X_abs = standardize(X_abs, train_idx)
    X_signed = standardize(X_signed, train_idx)

    # === Per-property A/B ===
    print("\n" + "=" * 70)
    print(f"{'property':<8} {'unit':<14} {'MAE_abs':>12} {'MAE_signed':>12} "
          f"{'Δ':>10} {'%Δ':>8}")
    print("-" * 70)

    all_results = []
    for prop_name, prop_idx in TARGET_INDICES.items():
        unit_name, unit_scale = TARGET_UNITS[prop_name]
        y_train = props[train_idx, prop_idx]
        y_test = props[test_idx, prop_idx]

        mae_abs, alpha_abs = best_ridge_mae(
            X_abs[train_idx], y_train, X_abs[test_idx], y_test
        )
        mae_signed, alpha_signed = best_ridge_mae(
            X_signed[train_idx], y_train, X_signed[test_idx], y_test
        )

        mae_abs_s = mae_abs * unit_scale
        mae_signed_s = mae_signed * unit_scale
        delta = mae_signed_s - mae_abs_s
        pct = 100.0 * delta / mae_abs_s

        result = {
            "property": prop_name,
            "unit": unit_name,
            "mae_abs": mae_abs_s,
            "mae_signed": mae_signed_s,
            "delta": delta,
            "pct_delta": pct,
            "alpha_abs": alpha_abs,
            "alpha_signed": alpha_signed,
        }
        all_results.append(result)

        marker = ""
        if delta < -0.001 * mae_abs_s:
            marker = "  <-- signed better"
        elif delta > 0.001 * mae_abs_s:
            marker = "  <-- abs better"

        print(f"{prop_name:<8} {unit_name:<14} "
              f"{mae_abs_s:>12.3f} {mae_signed_s:>12.3f} "
              f"{delta:>+10.3f} {pct:>+7.2f}% {marker}")

    print("\n" + "=" * 70)
    n_wins = sum(1 for r in all_results if r["delta"] < -0.001 * r["mae_abs"])
    n_losses = sum(1 for r in all_results if r["delta"] > 0.001 * r["mae_abs"])
    print(f"signed better: {n_wins} / {len(all_results)}")
    print(f"abs better:    {n_losses} / {len(all_results)}")

    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_total": n_total,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "kappas": list(kappas),
            "sigmas": list(sigmas),
            "n_features": int(X_abs.shape[1]),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
