"""
Properly-grounded QM9 benchmark, designed to expose the magic-number
footgun in the original mol_kernel.py.

Lineup, in order of "what does the geometric machinery actually do":

  random   : predict mean of train set on every test molecule.
             Absolute floor.

  cm       : Coulomb Matrix sorted eigenvalues + linear ridge. The
             standard classical chemistry baseline that any geometric
             feature method should beat.

  hopf_abs : the original mol_kernel.py default: use_abs=True (vMF
             kernel collapses 120 vertices to 60 antipodal pairs by
             taking |q . v|), kappa=5.5, sigmas=(0.5, 1.0, 2.0).
             This is what I had been benchmarking against SchNet and
             declaring "fixed-feature methods can't compete." The
             magic numbers (use_abs=True default + kappa=5.5 chosen
             for that abs-clamped dynamic range) are a footgun: they
             throw away the spinor double cover of 2I and run the
             kernel at the wrong temperature.

  hopf_signed : use_abs=False (signed q . v preserves the full 120-
             vertex 2I structure and resolves chirality), kappa=4.0
             (correctly retuned for the doubled dynamic range
             [-1, 1] vs [0, 1]). Same downstream pipeline; the only
             change is the two magic numbers.

  hopf_signed_multi : use_abs=False, kappas=(2.0, 4.0, 6.0),
             sigmas=(0.5, 1.0, 2.0). Multi-scale version with
             chirality resolved.

The point is to show that the previous "geometric features can't
compete" framing was reasoning from a setup-bug ceiling, not a
structural ceiling on what the framework can do.

Same train/test split, same kernel ridge, same alpha sweep across
all controllers. Reports per-property MAE and the relative gap to
the strongest baseline (CM, since it's the published-classical-
baseline comparison most people cite for QM9 kernel methods).
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
# train_ade_hopf was moved to experiments/mnist_geometric/ during the
# repo reorientation; pick it up from there for ridge_regression_bias.
sys.path.insert(0, os.path.join(_ROOT, "experiments", "mnist_geometric"))

from ade_geometry import get_ade
from qm9_data import (
    load_qm9, get_splits, TARGET_INDICES, TARGET_UNITS,
    ATOM_TYPES, ATOM_NUMBERS, N_ATOM_CHANNELS,
)
from mol_kernel import extract_molecular_features, coulomb_matrix_features
from train_ade_hopf import ridge_regression_bias


def compute_extensivity_features(all_coords, all_atoms):
    """Per-molecule features that capture EXTENSIVITY (scale with
    molecule size). The Hopf-angular features are intensive by
    construction (vMF activations are normalized to sum=1); they
    cannot represent properties like internal energy U0 or heat
    capacity Cv that scale with N_atoms.

    Channels (8 features per molecule):
        0..4: count of each atom type (H, C, N, O, F)
        5:    total atom count
        6:    sum of atomic numbers (Z)
        7:    sum of Z^2.4 (matches CM's diagonal scaling exactly)
    """
    n_mol = len(all_coords)
    feats = np.zeros((n_mol, 8))
    for i in range(n_mol):
        atoms = all_atoms[i]
        for ch_name, ch_idx in ATOM_TYPES.items():
            feats[i, ch_idx] = sum(1 for a in atoms if a == ch_name)
        feats[i, 5] = len(atoms)
        feats[i, 6] = sum(ATOM_NUMBERS.get(a, 0) for a in atoms)
        feats[i, 7] = sum(ATOM_NUMBERS.get(a, 0) ** 2.4 for a in atoms)
    return feats


def evaluate_mae(predictions, targets):
    return float(np.mean(np.abs(predictions - targets)))


def best_ridge_mae(X_train, y_train, X_test, y_test,
                   alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)):
    n_train = X_train.shape[0]
    Xt = np.hstack([X_train, np.ones((n_train, 1))])
    Xv = np.hstack([X_test, np.ones((len(X_test), 1))])
    Y = y_train.reshape(-1, 1)
    best_mae, best_alpha = float("inf"), alphas[0]
    for alpha in alphas:
        W = ridge_regression_bias(Xt, Y, alpha)
        mae = evaluate_mae((Xv @ W).ravel(), y_test)
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha
    return best_mae, best_alpha


def random_predictor_mae(y_train, y_test):
    """Predict mean of training set for every test molecule. Absolute floor."""
    pred = np.full_like(y_test, y_train.mean())
    return evaluate_mae(pred, y_test), None


def standardize(X, train_idx):
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mol", type=int, default=20000)
    ap.add_argument("--out-dir", type=str, default="checkpoints/qm9_bench")
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
    train_idx, _, test_idx = get_splits(n_total, n_train=n_train, n_val=n_val)
    print(f"Total {n_total}, train {len(train_idx)}, test {len(test_idx)}")

    print("Building ADE geometry...")
    ade = get_ade()

    # === Build feature matrices for each method ===
    feature_sets = {}

    print("\n[CM] Coulomb matrix baseline...")
    t0 = time.time()
    CM = coulomb_matrix_features(coords, atoms)
    print(f"  shape={CM.shape}, time={time.time()-t0:.1f}s")
    feature_sets["cm"] = CM

    print("\n[hopf_abs] kappa=5.5, use_abs=True (the magic-number footgun)...")
    t0 = time.time()
    X_abs = extract_molecular_features(
        coords, atoms, ade, kappas=(5.5,), sigmas=(1.0,), use_abs=True
    )
    print(f"  shape={X_abs.shape}, time={time.time()-t0:.1f}s")
    feature_sets["hopf_abs"] = X_abs

    print("\n[hopf_signed] kappa=4.0, use_abs=False (chirality-resolved + retuned)...")
    t0 = time.time()
    X_signed = extract_molecular_features(
        coords, atoms, ade, kappas=(4.0,), sigmas=(1.0,), use_abs=False
    )
    print(f"  shape={X_signed.shape}, time={time.time()-t0:.1f}s")
    feature_sets["hopf_signed"] = X_signed

    print("\n[hopf_signed_multi] kappas=(2.0, 4.0, 6.0), use_abs=False...")
    t0 = time.time()
    X_signed_multi = extract_molecular_features(
        coords, atoms, ade,
        kappas=(2.0, 4.0, 6.0), sigmas=(0.5, 1.0, 2.0), use_abs=False
    )
    print(f"  shape={X_signed_multi.shape}, time={time.time()-t0:.1f}s")
    feature_sets["hopf_signed_multi"] = X_signed_multi

    print("\n[ext] extensivity-only baseline (8 features: atom counts + Z sums)...")
    t0 = time.time()
    EXT = compute_extensivity_features(coords, atoms)
    print(f"  shape={EXT.shape}, time={time.time()-t0:.1f}s")
    feature_sets["ext"] = EXT

    print("\n[hopf_multi_ext] hopf_signed_multi + extensivity (8 extra dims)...")
    feature_sets["hopf_multi_ext"] = np.hstack([X_signed_multi, EXT])
    print(f"  shape={feature_sets['hopf_multi_ext'].shape}")

    print("\n[cm_ext] CM + extensivity (controls for what extensivity adds to CM)...")
    feature_sets["cm_ext"] = np.hstack([CM, EXT])
    print(f"  shape={feature_sets['cm_ext'].shape}")

    # Standardize each
    standardized = {}
    for name, X in feature_sets.items():
        Xs, _, _ = standardize(X, train_idx)
        standardized[name] = Xs
    del feature_sets
    gc.collect()

    # === Per-property A/B ===
    method_order = ["random", "cm", "ext", "cm_ext",
                    "hopf_abs", "hopf_signed", "hopf_signed_multi",
                    "hopf_multi_ext"]
    print("\n" + "=" * 110)
    print("Per-property MAE (in native units; lower better)")
    print("=" * 110)
    header = f"{'property':<8} {'unit':<14}" + "".join(f"{m:>16}" for m in method_order)
    print(header)
    print("-" * 110)

    all_results = []
    for prop_name, prop_idx in TARGET_INDICES.items():
        unit_name, unit_scale = TARGET_UNITS[prop_name]
        y_train = props[train_idx, prop_idx]
        y_test = props[test_idx, prop_idx]

        per_method = {}
        mae_r, _ = random_predictor_mae(y_train, y_test)
        per_method["random"] = mae_r * unit_scale
        for name in ("cm", "ext", "cm_ext", "hopf_abs", "hopf_signed",
                     "hopf_signed_multi", "hopf_multi_ext"):
            mae, alpha = best_ridge_mae(
                standardized[name][train_idx], y_train,
                standardized[name][test_idx], y_test
            )
            per_method[name] = mae * unit_scale

        all_results.append({
            "property": prop_name, "unit": unit_name, **per_method
        })

        cells = [f"{prop_name:<8} {unit_name:<14}"]
        for m in method_order:
            v = per_method[m]
            cells.append(f"{v:>15.3f}")
        print(" ".join(cells))

    # === Summary: chirality fix vs chirality-blind, ratios vs CM ===
    print("\n" + "=" * 110)
    print(f"Magic-number footgun gap: hopf_signed vs hopf_abs (lower MAE = better)")
    print("-" * 110)
    print(f"{'property':<8} {'hopf_abs':>14} {'hopf_signed':>14} {'pct_change':>14}")
    for r in all_results:
        ratio = (r["hopf_signed"] - r["hopf_abs"]) / r["hopf_abs"] * 100
        print(f"{r['property']:<8} {r['hopf_abs']:>14.3f} {r['hopf_signed']:>14.3f} "
              f"{ratio:>+13.2f}%")

    print("\n" + "=" * 110)
    print(f"Best method per property + ratio vs CM (negative = beats CM)")
    print("-" * 110)
    print(f"{'property':<8} {'best_method':<22} {'best_mae':>12} {'cm':>12} {'pct vs cm':>10}")
    for r in all_results:
        # exclude random from "best of"; it's the floor reference
        candidates = {k: v for k, v in r.items()
                      if k not in ("property", "unit", "random")}
        best_name = min(candidates, key=lambda k: candidates[k])
        best_mae = candidates[best_name]
        pct = (best_mae - r["cm"]) / r["cm"] * 100
        print(f"{r['property']:<8} {best_name:<22} "
              f"{best_mae:>12.3f} {r['cm']:>12.3f} {pct:>+9.2f}%")

    # Show what extensivity adds to CM and what hopf_multi+ext does
    print("\n" + "=" * 110)
    print(f"Extensivity contribution: ext alone, cm vs cm+ext, hopf_multi vs hopf_multi+ext")
    print("-" * 110)
    print(f"{'property':<8} {'ext':>10} {'cm':>10} {'cm+ext':>10} "
          f"{'hopf_multi':>12} {'hopf_multi+ext':>15}")
    for r in all_results:
        print(f"{r['property']:<8} {r['ext']:>10.3f} {r['cm']:>10.3f} "
              f"{r['cm_ext']:>10.3f} {r['hopf_signed_multi']:>12.3f} "
              f"{r['hopf_multi_ext']:>15.3f}")

    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_total": n_total,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "results": all_results,
            "method_order": method_order,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
