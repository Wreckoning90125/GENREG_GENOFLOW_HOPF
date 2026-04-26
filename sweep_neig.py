"""
Parameter-efficiency sweep: smallest irrep n_eig that matches flat n_eig=15
on QM9 MAE.

Reports MAE_flat(n_eig=15) and MAE_irrep(n_eig in {3, 5, 7, 10, 15}) for
each property; the param_efficiency ratio is the smallest irrep n_eig at
which irrep beats flat divided by 15.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qm9_data import load_qm9, get_splits, TARGET_INDICES, TARGET_UNITS
from irrep_qm9 import extract_dataset_features
from train_qm9_irrep import best_ridge_mae, evaluate_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mol", type=int, default=20000)
    ap.add_argument("--n-eig-flat", type=int, default=15)
    ap.add_argument("--n-eig-irrep-list", type=int, nargs="+",
                    default=[3, 5, 7, 10, 15])
    ap.add_argument("--out-dir", type=str, default="checkpoints/qm9_irrep")
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
    train_idx, val_idx, test_idx = get_splits(n_total, n_train=n_train, n_val=n_val)

    # === Flat baseline at n_eig = 15 ===
    print(f"Extracting flat features at n_eig={args.n_eig_flat}...")
    t0 = time.time()
    X_flat, _ = extract_dataset_features(
        coords, atoms, mode="flat", n_eig=args.n_eig_flat, verbose=True
    )
    print(f"  Flat shape {X_flat.shape}, {time.time() - t0:.1f}s")

    fmean, fstd = X_flat[train_idx].mean(axis=0), X_flat[train_idx].std(axis=0)
    fstd[fstd < 1e-8] = 1.0
    X_flat_train_s = (X_flat[train_idx] - fmean) / fstd
    X_flat_test_s = (X_flat[test_idx] - fmean) / fstd

    flat_mae = {}
    for prop_name, prop_idx in TARGET_INDICES.items():
        unit, scale = TARGET_UNITS[prop_name]
        y_tr = props[train_idx, prop_idx]
        y_te = props[test_idx, prop_idx]
        mae, _, _ = best_ridge_mae(X_flat_train_s, y_tr, X_flat_test_s, y_te)
        flat_mae[prop_name] = mae * scale
        print(f"  flat {prop_name}: {flat_mae[prop_name]:.3f} {unit}")

    # === Irrep sweep ===
    irrep_mae_by_neig = {}
    for n_eig in args.n_eig_irrep_list:
        print(f"\nExtracting irrep features at n_eig={n_eig}...")
        t0 = time.time()
        X_irrep, aut_orders = extract_dataset_features(
            coords, atoms, mode="irrep", n_eig=n_eig, verbose=False
        )
        print(f"  Irrep shape {X_irrep.shape}, {time.time() - t0:.1f}s")

        imean, istd = X_irrep[train_idx].mean(axis=0), X_irrep[train_idx].std(axis=0)
        istd[istd < 1e-8] = 1.0
        X_irrep_train_s = (X_irrep[train_idx] - imean) / istd
        X_irrep_test_s = (X_irrep[test_idx] - imean) / istd

        per_prop = {}
        for prop_name, prop_idx in TARGET_INDICES.items():
            unit, scale = TARGET_UNITS[prop_name]
            y_tr = props[train_idx, prop_idx]
            y_te = props[test_idx, prop_idx]
            mae, _, _ = best_ridge_mae(X_irrep_train_s, y_tr, X_irrep_test_s, y_te)
            per_prop[prop_name] = mae * scale
        irrep_mae_by_neig[n_eig] = per_prop
        n_features = X_irrep.shape[1]
        print(f"  irrep n_eig={n_eig} ({n_features} features):")
        for k, v in per_prop.items():
            unit = TARGET_UNITS[k][0]
            ratio = v / flat_mae[k]
            tag = "BEATS" if v < flat_mae[k] else "loses"
            print(f"    {k}: {v:.3f} {unit}  ({ratio:.3f}x flat)  {tag}")

    # === Summary: smallest n_eig that beats flat ===
    print("\n" + "=" * 70)
    print(f"{'Property':<8} {'Flat n=15':>12} " +
          " ".join(f"{f'irrep n={n}':>12}" for n in args.n_eig_irrep_list))
    print("-" * 70)
    for prop_name in TARGET_INDICES:
        unit = TARGET_UNITS[prop_name][0]
        row_strs = [f"{prop_name:<8} {flat_mae[prop_name]:>12.2f}"]
        for n_eig in args.n_eig_irrep_list:
            v = irrep_mae_by_neig[n_eig][prop_name]
            tag = "*" if v < flat_mae[prop_name] else " "
            row_strs.append(f"{v:>11.2f}{tag}")
        print(" ".join(row_strs))

    print()
    print(f"{'Property':<8} {'min_n_eig_beats_flat':<22} {'param_efficiency':<18}")
    for prop_name in TARGET_INDICES:
        beats = [n for n in args.n_eig_irrep_list
                 if irrep_mae_by_neig[n][prop_name] < flat_mae[prop_name]]
        if beats:
            min_n = min(beats)
            # irrep at n_eig has 2 * n_eig * 6 features; flat has 15 * 6 = 90
            irrep_features_count = 2 * min_n * 6
            flat_features_count = args.n_eig_flat * 6
            eff = irrep_features_count / flat_features_count
            print(f"{prop_name:<8} {min_n:<22} {eff:<18.3f}")
        else:
            print(f"{prop_name:<8} {'(none beat flat)':<22} {'n/a':<18}")

    # Save sweep results
    out_path = os.path.join(args.out_dir, "sweep_neig.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_total": n_total,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_eig_flat": args.n_eig_flat,
            "flat_mae_by_property": flat_mae,
            "irrep_mae_by_neig": {str(k): v for k, v in irrep_mae_by_neig.items()},
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
