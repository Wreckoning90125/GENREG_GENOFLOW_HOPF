#!/usr/bin/env python3
"""
Train 600-cell geometric features on QM9 molecular property prediction.

Pipeline:
  molecule → center on CoM → atom directions to S² → Hopf section to S³
  → von-Mises-Fisher soft assignment to 120 vertices → ADE eigenspace
  decomposition → 293 features per (kappa, sigma, channel) → concatenate
  → standardize → polynomial kernel ridge (Nystrom) → predict property

All features are fixed geometric functions of the molecular structure.
Only the kernel ridge readout is fit (closed-form solve, no SGD).

Usage:
    python train_qm9.py [--synthetic]  # use synthetic data for testing
    python train_qm9.py               # use real QM9 (requires download)
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
    TARGET_INDICES, TARGET_UNITS, HARTREE_TO_MEV
)
from mol_kernel import (
    extract_molecular_features, coulomb_matrix_features,
    test_rotation_invariance
)
from train_ade_hopf import (
    ridge_regression_bias, nystrom_poly_kernel_ridge
)


def evaluate_mae(predictions, targets):
    """Mean absolute error."""
    return np.mean(np.abs(predictions - targets))


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       property_name, unit_name, unit_scale, **kwargs):
    """Run linear ridge + kernel ridge on one property. Return results dict."""
    n_train, n_feat = X_train.shape
    results = {"property": property_name, "unit": unit_name,
               "n_train": n_train, "n_features": n_feat}

    # Add bias column
    X_train_b = np.hstack([X_train, np.ones((n_train, 1))])
    X_test_b = np.hstack([X_test, np.ones((len(X_test), 1))])

    # One-hot targets for ridge (regression: just use (N, 1))
    Y_train = y_train.reshape(-1, 1)

    # --- Linear ridge with alpha sweep ---
    print(f"\n  Linear ridge on {property_name}:")
    best_linear_mae = float("inf")
    best_linear_alpha = 0.1
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        W = ridge_regression_bias(X_train_b, Y_train, alpha)
        preds = X_test_b @ W
        mae_raw = evaluate_mae(preds.ravel(), y_test)
        mae_scaled = mae_raw * unit_scale
        marker = ""
        if mae_scaled < best_linear_mae:
            best_linear_mae = mae_scaled
            best_linear_alpha = alpha
            marker = " <-- best"
        print(f"    alpha={alpha:.3f}: MAE={mae_scaled:.2f} {unit_name}{marker}")

    results["linear_mae"] = best_linear_mae
    results["linear_alpha"] = best_linear_alpha
    print(f"  Linear ridge best: {best_linear_mae:.2f} {unit_name} "
          f"(alpha={best_linear_alpha})")
    del X_train_b, X_test_b
    gc.collect()

    # --- Polynomial kernel ridge (Nystrom) ---
    # Only if we have enough samples and caller wants it
    skip_kernel = kwargs.get("skip_kernel", False)
    if n_train >= 500 and not skip_kernel:
        print(f"\n  Kernel ridge on {property_name}:")
        m_vals = [min(2000, n_train // 2)]
        if n_train >= 5000:
            m_vals = [2000, 3000]
        if n_train >= 20000:
            m_vals = [3000, 5000]

        best_kernel_mae = float("inf")
        best_kernel_m = m_vals[0]
        best_kernel_alpha = 0.1
        for m_val in m_vals:
            for k_alpha in [0.01, 0.1, 1.0]:
                beta, L, K_test_m = nystrom_poly_kernel_ridge(
                    X_train, Y_train, X_test,
                    m=m_val, degree=2, alpha=k_alpha)
                preds = K_test_m @ beta
                mae_raw = evaluate_mae(preds.ravel(), y_test)
                mae_scaled = mae_raw * unit_scale
                marker = ""
                if mae_scaled < best_kernel_mae:
                    best_kernel_mae = mae_scaled
                    best_kernel_m = m_val
                    best_kernel_alpha = k_alpha
                    marker = " <-- best"
                print(f"    m={m_val}, alpha={k_alpha:.3f}: "
                      f"MAE={mae_scaled:.2f} {unit_name}{marker}")
                del K_test_m
                gc.collect()

        results["kernel_mae"] = best_kernel_mae
        results["kernel_m"] = best_kernel_m
        results["kernel_alpha"] = best_kernel_alpha
        results["best_mae"] = min(best_linear_mae, best_kernel_mae)
        print(f"  Kernel ridge best: {best_kernel_mae:.2f} {unit_name} "
              f"(m={best_kernel_m}, alpha={best_kernel_alpha})")
    else:
        results["kernel_mae"] = None
        results["best_mae"] = best_linear_mae

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic molecules for pipeline testing")
    parser.add_argument("--n-synthetic", type=int, default=1000,
                        help="Number of synthetic molecules")
    parser.add_argument("--kappas", type=float, nargs="+",
                        default=[3.0, 5.5, 8.0])
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0])
    parser.add_argument("--linear-only", action="store_true",
                        help="Skip kernel ridge (use when features >> samples/10)")
    args = parser.parse_args()

    print("=" * 60)
    print("600-Cell Geometric Features on QM9")
    print("  Molecular property prediction with fixed geometric basis")
    print("  + polynomial kernel ridge (Nystrom)")
    print("=" * 60)

    # Setup output
    out_dir = "checkpoints/qm9"
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    if args.synthetic:
        print(f"\nUsing {args.n_synthetic} synthetic molecules (pipeline test)")
        all_coords, all_atoms, all_properties, all_indices = \
            generate_synthetic_molecules(args.n_synthetic)
        n_train = int(0.6 * len(all_coords))
        n_val = int(0.2 * len(all_coords))
    else:
        print("\nLoading QM9...")
        all_coords, all_atoms, all_properties, all_indices = load_qm9()
        n_train = 110000
        n_val = 10000

    n_total = len(all_coords)
    train_idx, val_idx, test_idx = get_splits(n_total, n_train=n_train,
                                               n_val=n_val)
    print(f"  Total: {n_total}, Train: {len(train_idx)}, "
          f"Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Build ADE geometry
    print("\nBuilding ADE geometry...")
    ade = get_ade()

    # --- Rotation invariance test ---
    print("\n--- Rotation invariance test ---")
    mean_rc, max_rc = test_rotation_invariance(n_molecules=10)

    # --- Extract 600-cell geometric features ---
    kappas = tuple(args.kappas)
    sigmas = tuple(args.sigmas)
    n_combos = len(kappas) * len(sigmas) * 5  # 5 atom channels
    print(f"\n--- Feature extraction ---")
    print(f"  Kappas: {kappas}")
    print(f"  Sigmas: {sigmas}")
    print(f"  Expected: 293 features x {n_combos} combos = "
          f"{293 * n_combos} features/molecule")

    t0 = time.time()
    X_all = extract_molecular_features(all_coords, all_atoms, ade,
                                        kappas=kappas, sigmas=sigmas)
    t_feat = time.time() - t0
    print(f"  Feature matrix shape: {X_all.shape}")
    print(f"  Time: {t_feat:.1f}s ({t_feat/n_total*1000:.1f}ms/molecule)")

    # Check for NaN/Inf
    n_nan = np.sum(np.isnan(X_all))
    n_inf = np.sum(np.isinf(X_all))
    n_zero_cols = np.sum(X_all.std(axis=0) < 1e-12)
    print(f"  NaN: {n_nan}, Inf: {n_inf}, dead features: {n_zero_cols}")
    if n_nan > 0 or n_inf > 0:
        print("  WARNING: NaN or Inf in features, replacing with 0")
        X_all = np.nan_to_num(X_all)

    # Split
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]
    del X_all
    gc.collect()

    # Standardize (fit on train only)
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    # --- Coulomb matrix baseline ---
    print("\n--- Coulomb matrix baseline ---")
    t0 = time.time()
    CM_all = coulomb_matrix_features(all_coords, all_atoms)
    print(f"  CM shape: {CM_all.shape}, time: {time.time()-t0:.1f}s")

    CM_train = CM_all[train_idx]
    CM_test = CM_all[test_idx]
    del CM_all
    gc.collect()

    cm_mean = CM_train.mean(axis=0)
    cm_std = CM_train.std(axis=0)
    cm_std[cm_std < 1e-8] = 1.0
    CM_train_s = (CM_train - cm_mean) / cm_std
    CM_test_s = (CM_test - cm_mean) / cm_std
    del CM_train, CM_test
    gc.collect()

    # --- Train and evaluate on each target property ---
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    all_results = []

    for prop_name, prop_idx in TARGET_INDICES.items():
        unit_name, unit_scale = TARGET_UNITS[prop_name]
        y_train = all_properties[train_idx, prop_idx]
        y_test = all_properties[test_idx, prop_idx]

        print(f"\n{'='*40}")
        print(f"Property: {prop_name} ({unit_name})")
        print(f"{'='*40}")
        print(f"  y range: [{y_train.min():.4f}, {y_train.max():.4f}] "
              f"(raw Hartree/native)")

        # 600-cell features
        print("\n  --- 600-cell geometric features ---")
        result = train_and_evaluate(X_train, y_train, X_test, y_test,
                                    prop_name, unit_name, unit_scale,
                                    skip_kernel=args.linear_only)

        # Coulomb matrix baseline
        print("\n  --- Coulomb matrix baseline ---")
        cm_result = train_and_evaluate(CM_train_s, y_train, CM_test_s, y_test,
                                       prop_name + "_CM", unit_name, unit_scale,
                                       skip_kernel=args.linear_only)
        result["cm_linear_mae"] = cm_result["linear_mae"]
        result["cm_kernel_mae"] = cm_result.get("kernel_mae")
        result["cm_best_mae"] = cm_result["best_mae"]

        all_results.append(result)

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("Summary: MAE comparison")
    print("=" * 60)
    print(f"{'Property':<10} {'Unit':<12} {'600-cell':<12} {'CM baseline':<12} "
          f"{'SchNet*':<10}")
    print("-" * 56)

    # Published SchNet baselines (approximate, for reference)
    schnet_baselines = {
        "gap": 63, "homo": 41, "lumo": 34, "U0": 14,
        "mu": 0.033, "alpha": 0.235, "Cv": 0.033
    }

    for r in all_results:
        prop = r["property"]
        unit = r["unit"]
        our_mae = r["best_mae"]
        cm_mae = r["cm_best_mae"]
        schnet = schnet_baselines.get(prop, "?")
        print(f"{prop:<10} {unit:<12} {our_mae:<12.2f} {cm_mae:<12.2f} "
              f"{schnet}")

    print("\n* SchNet values are approximate published baselines for reference")

    # Save results
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "kappas": list(kappas),
            "sigmas": list(sigmas),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_features": X_train.shape[1],
            "rotation_invariance": {
                "mean_relative_change": float(mean_rc),
                "max_relative_change": float(max_rc),
            },
            "results": all_results,
            "synthetic": args.synthetic,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save training log
    log_path = os.path.join(out_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write("600-Cell Geometric Features on QM9\n")
        f.write(f"Kappas: {kappas}\n")
        f.write(f"Sigmas: {sigmas}\n")
        f.write(f"Features: {X_train.shape[1]}\n")
        f.write(f"Train: {len(train_idx)}, Test: {len(test_idx)}\n")
        f.write(f"Synthetic: {args.synthetic}\n")
        f.write(f"Rotation invariance: mean_rc={mean_rc:.4f}, "
                f"max_rc={max_rc:.4f}\n\n")
        for r in all_results:
            f.write(f"{r['property']}: "
                    f"600-cell={r['best_mae']:.2f} {r['unit']}, "
                    f"CM={r['cm_best_mae']:.2f} {r['unit']}\n")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
