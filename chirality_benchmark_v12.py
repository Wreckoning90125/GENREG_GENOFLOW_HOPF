#!/usr/bin/env python3
"""
Chirality benchmark: v12 (cell + co-exact 2-form features) vs v11 face vs v10
==============================================================================

The earlier chirality benchmark showed v11 face/Berry features beat v10
on cross-digit hflip transfer by ~3%. v12 adds two more chirality-aware
readings:

- Co-exact 2-form features: triangle signal projected onto d₂ᵀ·cell_eigs
- Cell (Ω³) features: linear signal weighted by signed cell volume
  (the 3-form analog of triangle Berry phase)

Both layers carry orientation-reversing invariants that should strengthen
chirality generalization further.
"""

import os
import sys
import gc
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hopf_controller import _get_pixel_kernel, _berry_verified
from ade_geometry import get_ade
from chirality_benchmark import (
    build_hflip_dataset,
    linear_ridge_binary,
    kernel_ridge_binary,
)
from train_ade_hopf import extract_features_batch
from train_v11 import extract_face_features_batch
from train_v11_ensemble import standardize_in_place
from train_v12 import (
    extract_coexact_face_features_batch,
    extract_cell_features_batch,
)


def extract_full_v12_multiscale(X, ade, kappas):
    """v10 + v11 face + v12 new, all at each kappa scale — the full
    de Rham ladder feature set. Pre-allocates the output array to
    avoid hstack copies (v12 has 1611 features per sample and the
    double-allocation during hstack blows memory on 40k samples)."""
    X_np = np.asarray(X, dtype=np.float64)
    N = X_np.shape[0]

    # Probe feature counts on one image (cheap)
    probe = X_np[:1]
    pk0 = _get_pixel_kernel(784, kappa=kappas[0])
    v10_n = extract_features_batch(probe, ade, pk0).shape[1]
    F0 = probe @ pk0
    v11_n = extract_face_features_batch(F0, ade).shape[1]
    coex_n = extract_coexact_face_features_batch(F0, ade).shape[1]
    cell_n = extract_cell_features_batch(F0, ade).shape[1]
    per_scale = v10_n + v11_n + coex_n + cell_n
    total = per_scale * len(kappas)

    out = np.empty((N, total), dtype=np.float64)
    col = 0
    for k in kappas:
        pk = _get_pixel_kernel(784, kappa=k)
        v10f = extract_features_batch(X_np, ade, pk)
        out[:, col:col + v10_n] = v10f
        col += v10_n
        del v10f
        F = X_np @ pk
        v11f = extract_face_features_batch(F, ade)
        out[:, col:col + v11_n] = v11f
        col += v11_n
        del v11f
        coex = extract_coexact_face_features_batch(F, ade)
        out[:, col:col + coex_n] = coex
        col += coex_n
        del coex
        cell = extract_cell_features_batch(F, ade)
        out[:, col:col + cell_n] = cell
        col += cell_n
        del cell, F
        gc.collect()
    return out


def extract_v12_new_multiscale(X, ade, kappas):
    """
    Extract only the NEW v12 features (coexact 2-form + cell) at
    multiple kappa scales. Does not include v10 or v11 face features —
    those are handled by the existing chirality_benchmark pipeline.
    """
    all_feats = []
    X_np = np.asarray(X, dtype=np.float64)
    for k in kappas:
        pk = _get_pixel_kernel(784, kappa=k)
        F = X_np @ pk
        coex = extract_coexact_face_features_batch(F, ade)
        cell = extract_cell_features_batch(F, ade)
        all_feats.append(np.hstack([coex, cell]))
    return np.hstack(all_feats)


def run_v12_experiment(name, ade, kappa_set, n_train, n_test,
                       train_digits=None, test_digits=None):
    print("\n" + "=" * 60)
    print(f"v12 EXPERIMENT: {name}")
    print("=" * 60)
    if train_digits is not None:
        print(f"  Train digits: {sorted(train_digits)}")
    if test_digits is not None:
        print(f"  Test digits:  {sorted(test_digits)}")

    print("\nBuilding hflip dataset...")
    t0 = time.time()
    X_tr, tr_chiral, X_te, te_chiral = build_hflip_dataset(
        n_train=n_train, n_test=n_test,
        train_digits=train_digits, test_digits=test_digits)
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}, "
          f"time={time.time() - t0:.1f}s")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_tr))
    X_tr = X_tr[idx]
    tr_chiral_sh = tr_chiral[idx]

    # --- v12-new only (coexact 2-form + cell) ---
    print("\nExtracting v12-new (coexact + cell) multi-scale features...")
    t0 = time.time()
    Xv12_tr = extract_v12_new_multiscale(X_tr, ade, kappa_set)
    Xv12_te = extract_v12_new_multiscale(X_te, ade, kappa_set)
    print(f"  shapes: train={Xv12_tr.shape}, test={Xv12_te.shape}, "
          f"time={time.time() - t0:.1f}s")
    standardize_in_place(Xv12_tr, Xv12_te)
    gc.collect()

    print("\nv12-new linear ridge:")
    v12new_lin_acc, _ = linear_ridge_binary(
        Xv12_tr, tr_chiral_sh, Xv12_te, te_chiral,
        alphas=[0.001, 0.01, 0.1, 1.0], tag="v12new-lin")

    print("\nv12-new kernel ridge:")
    v12new_ker_acc, v12new_ker_params = kernel_ridge_binary(
        Xv12_tr, tr_chiral_sh, Xv12_te, te_chiral,
        m_vals=[1500, 2000],
        alphas=[0.1, 1.0],
        degree=2, tag="v12new-ker")
    del Xv12_tr, Xv12_te
    gc.collect()

    # --- Full v12 (v10 + v11 face + v12 new) — the whole Hodge ladder ---
    print("\nExtracting FULL v12 (v10 + v11 face + v12 new) features...")
    t0 = time.time()
    Xfull_tr = extract_full_v12_multiscale(X_tr, ade, kappa_set)
    Xfull_te = extract_full_v12_multiscale(X_te, ade, kappa_set)
    print(f"  shapes: train={Xfull_tr.shape}, test={Xfull_te.shape}, "
          f"time={time.time() - t0:.1f}s")
    standardize_in_place(Xfull_tr, Xfull_te)
    gc.collect()

    print("\nFULL v12 linear ridge:")
    full_lin_acc, _ = linear_ridge_binary(
        Xfull_tr, tr_chiral_sh, Xfull_te, te_chiral,
        alphas=[0.001, 0.01, 0.1, 1.0], tag="full-lin")

    print("\nFULL v12 kernel ridge:")
    full_ker_acc, full_ker_params = kernel_ridge_binary(
        Xfull_tr, tr_chiral_sh, Xfull_te, te_chiral,
        m_vals=[2000, 3000],
        alphas=[0.01, 0.1, 1.0],
        degree=2, tag="full-ker")
    del Xfull_tr, Xfull_te
    gc.collect()

    print()
    print(f"--- {name} RESULTS ---")
    print(f"v12-new linear:   {v12new_lin_acc * 100:.2f}%")
    print(f"v12-new kernel:   {v12new_ker_acc * 100:.2f}%")
    print(f"FULL v12 linear:  {full_lin_acc * 100:.2f}%")
    print(f"FULL v12 kernel:  {full_ker_acc * 100:.2f}%")

    return {
        "experiment": name,
        "train_digits": sorted(train_digits) if train_digits else None,
        "test_digits": sorted(test_digits) if test_digits else None,
        "train_size": int(X_tr.shape[0]),
        "test_size": int(X_te.shape[0]),
        "v12new_linear_acc": float(v12new_lin_acc),
        "v12new_kernel_acc": float(v12new_ker_acc),
        "full_v12_linear_acc": float(full_lin_acc),
        "full_v12_kernel_acc": float(full_ker_acc),
        "full_v12_kernel_params": {"m": full_ker_params[0],
                                   "alpha": full_ker_params[1]},
    }


def main():
    print("=" * 60)
    print("v12 Chirality Benchmark — cell + coexact 2-form features")
    print("=" * 60)
    if _berry_verified:
        print("Cl(3,0) Berry phase verified")

    out_dir = "hopf_chirality_bench_v12"
    os.makedirs(out_dir, exist_ok=True)

    ade = get_ade()
    kappa_set = [3.0, 5.5, 8.0]

    all_results = []
    all_results.append(run_v12_experiment(
        "full-MNIST hflip detection",
        ade, kappa_set, n_train=20000, n_test=5000,
        train_digits=None, test_digits=None))
    all_results.append(run_v12_experiment(
        "cross-digit hflip (train:0-4, test:5-9)",
        ade, kappa_set, n_train=20000, n_test=5000,
        train_digits={0, 1, 2, 3, 4}, test_digits={5, 6, 7, 8, 9}))
    all_results.append(run_v12_experiment(
        "cross-digit hflip (train:5-9, test:0-4)",
        ade, kappa_set, n_train=20000, n_test=5000,
        train_digits={5, 6, 7, 8, 9}, test_digits={0, 1, 2, 3, 4}))

    # Summary table including v11 face-kernel from prior run for comparison
    prior_v11 = {
        "full-MNIST hflip detection": {"v10_k": 0.9908, "face_k": 0.9857},
        "cross-digit hflip (train:0-4, test:5-9)": {"v10_k": 0.9026, "face_k": 0.9287},
        "cross-digit hflip (train:5-9, test:0-4)": {"v10_k": 0.9015, "face_k": 0.9353},
    }
    print()
    print("=" * 90)
    print("CHIRALITY OVERALL — v10, v11 face only, v12-new only, FULL v12 (all)")
    print("=" * 90)
    print(f"{'experiment':48s} {'v10-k':>8s} {'face-k':>8s} "
          f"{'v12new-k':>10s} {'FULL-k':>8s}")
    for r in all_results:
        p = prior_v11.get(r["experiment"], {"v10_k": 0, "face_k": 0})
        print(f"{r['experiment']:48s} "
              f"{p['v10_k']*100:7.2f}% {p['face_k']*100:7.2f}% "
              f"{r['v12new_kernel_acc']*100:9.2f}% "
              f"{r['full_v12_kernel_acc']*100:7.2f}%")
    print("=" * 90)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({
            "kappa_set": kappa_set,
            "berry_phase_verified": _berry_verified,
            "experiments": all_results,
            "prior_v11_for_comparison": prior_v11,
        }, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
