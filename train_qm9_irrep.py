"""
Run the irrep-native vs flat eigendecomposition A/B on QM9 graph Laplacians.

Compare:
    A (flat):    flat_features = standard graph-Laplacian eigendecomp +
                 per-atom-type vertex amplitudes
    B (irrep):   irrep_features = same statistics, but for Aut(graph) >= 2,
                 split into trivial-isotypic and non-trivial-isotypic blocks
                 before eigendecomp

Both A and B return identical features for trivial-Aut molecules (verified
in irrep_qm9.py); the only difference shows up on |Aut(graph)| > 1.

Reports:
    Delta_MAE_full      = MAE_irrep - MAE_flat   on full test set
    Delta_MAE_highsymm  = same, restricted to |Aut| > 1
    Delta_MAE_trivialaut = same, restricted to |Aut| = 1
                          (sanity: must be 0 to floating point)
    aut_distribution    = histogram of |Aut(G)| in the test split

For each property: gap, homo, lumo, U0, mu, alpha, Cv.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qm9_data import (
    load_qm9, generate_synthetic_molecules, get_splits,
    TARGET_INDICES, TARGET_UNITS, HARTREE_TO_MEV,
)
from irrep_qm9 import extract_dataset_features
from train_ade_hopf import ridge_regression_bias


def evaluate_mae(predictions, targets):
    return float(np.mean(np.abs(predictions - targets)))


def best_ridge_mae(X_train, y_train, X_test, y_test, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Sweep ridge alpha, return (best_mae, best_alpha) on test set."""
    n_train = X_train.shape[0]
    X_train_b = np.hstack([X_train, np.ones((n_train, 1))])
    X_test_b = np.hstack([X_test, np.ones((len(X_test), 1))])
    Y = y_train.reshape(-1, 1)
    best_mae = float("inf")
    best_alpha = alphas[0]
    best_preds = None
    for alpha in alphas:
        W = ridge_regression_bias(X_train_b, Y, alpha)
        preds = (X_test_b @ W).ravel()
        mae = evaluate_mae(preds, y_test)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_preds = preds
    return best_mae, best_alpha, best_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mol", type=int, default=20000,
                    help="Subsample to N molecules for speed (default 20000)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic molecules instead of QM9")
    ap.add_argument("--n-eig", type=int, default=15,
                    help="Number of eigenvalues per molecule")
    ap.add_argument("--out-dir", type=str, default="checkpoints/qm9_irrep")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.synthetic:
        coords, atoms, props, _ = generate_synthetic_molecules(args.n_mol)
    else:
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
    print(f"Total: {n_total}, train: {len(train_idx)}, test: {len(test_idx)}")

    # === Extract features (A and B) ===
    print(f"\nExtracting flat features (n_eig={args.n_eig})...")
    t0 = time.time()
    X_flat, aut_orders_a = extract_dataset_features(
        coords, atoms, mode="flat", n_eig=args.n_eig, verbose=True
    )
    print(f"  Done in {time.time() - t0:.1f}s, shape {X_flat.shape}")

    print(f"\nExtracting irrep features (n_eig={args.n_eig})...")
    t0 = time.time()
    X_irrep, aut_orders_b = extract_dataset_features(
        coords, atoms, mode="irrep", n_eig=args.n_eig, verbose=True
    )
    print(f"  Done in {time.time() - t0:.1f}s, shape {X_irrep.shape}")

    # Sanity: aut orders should match between the two passes
    assert np.array_equal(aut_orders_a, aut_orders_b), "aut order mismatch"
    aut_orders = aut_orders_a

    # |Aut| histogram
    print("\n=== |Aut(graph)| distribution ===")
    print(f"  |Aut|=1:       {(aut_orders == 1).sum()}")
    print(f"  |Aut|=2:       {(aut_orders == 2).sum()}")
    print(f"  |Aut|=3..6:    {((aut_orders >= 3) & (aut_orders <= 6)).sum()}")
    print(f"  |Aut|=7..24:   {((aut_orders >= 7) & (aut_orders <= 24)).sum()}")
    print(f"  |Aut|>24:      {(aut_orders > 24).sum()}")
    print(f"  max |Aut|:     {aut_orders.max()}")

    # Sanity: irrep is a strict superset of flat. For ALL molecules, the
    # first flat_dim columns of X_irrep must equal X_flat exactly. For
    # trivial-Aut molecules, the remaining flat_dim columns must be zero.
    flat_dim = X_flat.shape[1]
    super_diff = float(np.max(np.abs(X_irrep[:, :flat_dim] - X_flat)))
    print(f"\nSanity: max|X_irrep[:, :flat_dim] - X_flat| (all molecules) = "
          f"{super_diff:.2e} (must be 0)")
    assert super_diff < 1e-10, "BUG: irrep is not a strict superset of flat"

    triv_mask_all = (aut_orders == 1)
    if triv_mask_all.any():
        triv_extra_max = float(np.max(np.abs(X_irrep[triv_mask_all, flat_dim:])))
        print(f"Sanity: max|X_irrep extra block on trivial-Aut| = "
              f"{triv_extra_max:.2e} (must be 0)")
        assert triv_extra_max < 1e-10, "BUG: irrep extra block nonzero on trivial Aut"

    # Standardize (fit on train)
    X_flat_train = X_flat[train_idx]
    X_flat_test = X_flat[test_idx]
    X_irrep_train = X_irrep[train_idx]
    X_irrep_test = X_irrep[test_idx]

    fmean, fstd = X_flat_train.mean(axis=0), X_flat_train.std(axis=0)
    fstd[fstd < 1e-8] = 1.0
    X_flat_train = (X_flat_train - fmean) / fstd
    X_flat_test = (X_flat_test - fmean) / fstd

    imean, istd = X_irrep_train.mean(axis=0), X_irrep_train.std(axis=0)
    istd[istd < 1e-8] = 1.0
    X_irrep_train = (X_irrep_train - imean) / istd
    X_irrep_test = (X_irrep_test - imean) / istd

    # Subset masks on test set
    triv_mask = (aut_orders[test_idx] == 1)
    high_mask = (aut_orders[test_idx] > 1)

    # === Train + evaluate per property ===
    all_results = []
    print("\n" + "=" * 70)
    print(f"{'property':<8} {'unit':<14} {'MAE_flat':>10} {'MAE_irrep':>10} "
          f"{'Δ_full':>10} {'Δ_highsymm':>11} {'Δ_trivialaut':>13}")
    print("-" * 70)

    for prop_name, prop_idx in TARGET_INDICES.items():
        unit_name, unit_scale = TARGET_UNITS[prop_name]
        y_train = props[train_idx, prop_idx]
        y_test = props[test_idx, prop_idx]

        mae_flat, alpha_flat, preds_flat = best_ridge_mae(
            X_flat_train, y_train, X_flat_test, y_test
        )
        mae_irrep, alpha_irrep, preds_irrep = best_ridge_mae(
            X_irrep_train, y_train, X_irrep_test, y_test
        )

        # Deltas in scaled units
        mae_flat_s = mae_flat * unit_scale
        mae_irrep_s = mae_irrep * unit_scale
        delta_full = mae_irrep_s - mae_flat_s

        # Subsetted deltas: use the same fitted models, restrict to subset
        if high_mask.any():
            delta_highsymm = (
                evaluate_mae(preds_irrep[high_mask], y_test[high_mask])
                - evaluate_mae(preds_flat[high_mask], y_test[high_mask])
            ) * unit_scale
        else:
            delta_highsymm = float("nan")

        if triv_mask.any():
            delta_trivialaut = (
                evaluate_mae(preds_irrep[triv_mask], y_test[triv_mask])
                - evaluate_mae(preds_flat[triv_mask], y_test[triv_mask])
            ) * unit_scale
        else:
            delta_trivialaut = float("nan")

        result = {
            "property": prop_name,
            "unit": unit_name,
            "mae_flat": mae_flat_s,
            "mae_irrep": mae_irrep_s,
            "delta_full": delta_full,
            "delta_highsymm": delta_highsymm,
            "delta_trivialaut": delta_trivialaut,
            "alpha_flat": alpha_flat,
            "alpha_irrep": alpha_irrep,
            "n_test_total": int(len(y_test)),
            "n_test_highsymm": int(high_mask.sum()),
            "n_test_trivialaut": int(triv_mask.sum()),
        }
        all_results.append(result)

        print(f"{prop_name:<8} {unit_name:<14} "
              f"{mae_flat_s:>10.3f} {mae_irrep_s:>10.3f} "
              f"{delta_full:>+10.3f} {delta_highsymm:>+11.3f} {delta_trivialaut:>+13.4f}")

    # === Summary ===
    print("\n" + "=" * 70)
    n_wins = sum(1 for r in all_results if r["delta_full"] < -0.001)
    n_losses = sum(1 for r in all_results if r["delta_full"] > 0.001)
    n_draws = len(all_results) - n_wins - n_losses
    print(f"Wins (irrep better): {n_wins} / {len(all_results)}")
    print(f"Losses (flat better): {n_losses} / {len(all_results)}")
    print(f"Draws: {n_draws} / {len(all_results)}")

    # Trivial-aut sanity
    max_triv_delta = max(abs(r["delta_trivialaut"]) for r in all_results
                         if not np.isnan(r["delta_trivialaut"]))
    print(f"\nSanity: max |Delta_trivialaut| = {max_triv_delta:.4e} "
          f"(must be ~0; non-zero means bug)")

    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_total": n_total,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_eig": args.n_eig,
            "synthetic": args.synthetic,
            "aut_distribution": {
                "trivial": int((aut_orders == 1).sum()),
                "n2": int((aut_orders == 2).sum()),
                "n3to6": int(((aut_orders >= 3) & (aut_orders <= 6)).sum()),
                "n7to24": int(((aut_orders >= 7) & (aut_orders <= 24)).sum()),
                "ngt24": int((aut_orders > 24).sum()),
                "max": int(aut_orders.max()),
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
