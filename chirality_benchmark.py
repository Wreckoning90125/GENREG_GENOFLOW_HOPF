#!/usr/bin/env python3
"""
Chirality benchmark: hflip-MNIST binary classification
========================================================

Builds a clean chirality-discrimination task by doubling MNIST with
horizontally-flipped copies and asking: "is this image original or
mirror-flipped?"

- Train: 60,000 originals + 60,000 hflips = 120,000 samples
- Test:  10,000 originals + 10,000 hflips =  20,000 samples
- Labels: 0 = original, 1 = flipped

The task is a direct probe of whether a feature extractor can see
chirality. v10's features are built from 2I-equivariant rotation
operations (SO(3) actions on S³, Hopf projection S³ → S², eigenbasis
projections with Laplacians symmetric under 2I). These encode shape
information but their invariants are under proper rotations — they
don't have a first-class channel for the orientation-reversing part
of O(3).

v11 adds the signed Berry phase Ω_ijk/2 to every triangle feature.
Under horizontal flip (which is an orientation-reversing map on the
pixel grid), the Hopf projections of the 600-cell vertices change
in a way that flips the sign of the signed solid angle for every
triangle. So v11's face features carry a dedicated chirality
channel that v10 does not.

Prediction: v11 > v10 significantly on hflip detection. If true,
this validates v11 as the right completion for chirality-sensitive
tasks even though MNIST digit classification is chirality-blind.
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
from train_ade_hopf import extract_features_batch, nystrom_poly_kernel_ridge
from train_v11 import extract_face_features_batch
from train_v11_ensemble import (
    extract_v10_multiscale,
    extract_face_multiscale,
    standardize_in_place,
)


def load_mnist_split(split="train"):
    from nodes.envs.mnist import load_mnist as _load_mnist
    return _load_mnist(split)


def _load_mnist_as_array(split):
    imgs, labels, _, _ = load_mnist_split(split)
    X = np.empty((len(imgs), 784), dtype=np.float32)
    for i, im in enumerate(imgs):
        X[i] = im
    y = np.array(labels, dtype=np.int64)
    return X, y


def build_hflip_dataset(n_train=20000, n_test=5000, seed=42,
                        train_digits=None, test_digits=None):
    """
    Build the hflip chirality dataset (numpy-only, memory efficient).

    Args:
        n_train: number of ORIGINAL MNIST train images to use
                 (doubled to n_train*2 after adding hflipped copies)
        n_test:  number of ORIGINAL MNIST test images to use
        train_digits: if given, only use these digit classes in train
        test_digits:  if given, only use these digit classes in test
                      (cross-digit generalization test: train on some
                       digits, test on disjoint ones)

    Returns:
        X_train: (2*n_train, 784) float32
        y_train: (2*n_train,) int — 0 for original, 1 for flipped
        X_test:  (2*n_test,  784) float32
        y_test:  (2*n_test,)  int
    """
    Xtr_all, ytr_all = _load_mnist_as_array("train")
    Xte_all, yte_all = _load_mnist_as_array("test")

    if train_digits is not None:
        tr_mask = np.isin(ytr_all, list(train_digits))
        Xtr_all = Xtr_all[tr_mask]
        ytr_all = ytr_all[tr_mask]
    if test_digits is not None:
        te_mask = np.isin(yte_all, list(test_digits))
        Xte_all = Xte_all[te_mask]
        yte_all = yte_all[te_mask]

    rng = np.random.default_rng(seed)
    n_tr_avail = len(Xtr_all)
    n_te_avail = len(Xte_all)
    n_train_use = min(n_train, n_tr_avail)
    n_test_use = min(n_test, n_te_avail)
    tr_idx = rng.choice(n_tr_avail, n_train_use, replace=False)
    te_idx = rng.choice(n_te_avail, n_test_use, replace=False)
    X_tr = Xtr_all[tr_idx]
    X_te = Xte_all[te_idx]

    # Horizontal flip via reshape + slice
    X_tr_flip = X_tr.reshape(n_train_use, 28, 28)[:, :, ::-1].reshape(n_train_use, 784).copy()
    X_te_flip = X_te.reshape(n_test_use, 28, 28)[:, :, ::-1].reshape(n_test_use, 784).copy()

    X_train = np.concatenate([X_tr, X_tr_flip], axis=0)
    X_test = np.concatenate([X_te, X_te_flip], axis=0)
    y_train = np.concatenate([np.zeros(n_train_use, dtype=np.int64),
                              np.ones(n_train_use, dtype=np.int64)])
    y_test = np.concatenate([np.zeros(n_test_use, dtype=np.int64),
                             np.ones(n_test_use, dtype=np.int64)])
    return X_train, y_train, X_test, y_test


def kernel_ridge_binary(X_train, y_train, X_test, y_test, m_vals, alphas,
                        degree=2, tag=""):
    """
    Binary classification via kernel ridge with Nystrom approximation.
    Labels are real scalars in {-1, +1}; prediction is sign of score.
    """
    # Map labels to {-1, +1} for regression
    y_tr = (2.0 * np.array(y_train, dtype=np.float64) - 1.0).reshape(-1, 1)
    y_te = np.array(y_test)

    best_acc = 0
    best_params = None
    for m in m_vals:
        for a in alphas:
            beta, L, K_test = nystrom_poly_kernel_ridge(
                X_train, y_tr, X_test, m=m, degree=degree, alpha=a)
            scores = (K_test @ beta).ravel()
            preds = (scores > 0).astype(np.int64)
            acc = float(np.mean(preds == y_te))
            if acc > best_acc:
                best_acc = acc
                best_params = (m, a)
            print(f"  {tag} m={m}, alpha={a:.3f}: test={acc:.4f}")
            del K_test
            gc.collect()
    print(f"  {tag} best: m={best_params[0]}, alpha={best_params[1]}, "
          f"test={best_acc:.4f}")
    return best_acc, best_params


def linear_ridge_binary(X_train, y_train, X_test, y_test, alphas, tag=""):
    y_tr = (2.0 * np.array(y_train, dtype=np.float64) - 1.0).reshape(-1, 1)
    y_te = np.array(y_test)
    XtX = X_train.T @ X_train
    XtY = X_train.T @ y_tr
    n_feat = X_train.shape[1]
    best_acc = 0
    best_alpha = 0.1
    for a in alphas:
        W = np.linalg.solve(XtX + a * np.eye(n_feat), XtY)
        preds = ((X_test @ W).ravel() > 0).astype(np.int64)
        acc = float(np.mean(preds == y_te))
        if acc > best_acc:
            best_acc = acc
            best_alpha = a
        print(f"  {tag} alpha={a:.4f}: test={acc:.4f}")
    print(f"  {tag} best: alpha={best_alpha}, test={best_acc:.4f}")
    return best_acc, best_alpha


def run_experiment(name, ade, kappa_set, n_train, n_test,
                   train_digits=None, test_digits=None):
    """Run one chirality experiment and return a results dict."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {name}")
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
    print(f"  Train: {X_tr.shape} (half original, half hflipped)")
    print(f"  Test:  {X_te.shape}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Shuffle train once so Nystrom landmarks sample both classes evenly
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_tr))
    X_tr = X_tr[idx]
    tr_chiral_sh = tr_chiral[idx]

    # --- v10 feature extraction ---
    print("\n[1/2] Extracting v10 multi-scale features...")
    t0 = time.time()
    Xv10_tr = extract_v10_multiscale(X_tr, ade, kappa_set)
    Xv10_te = extract_v10_multiscale(X_te, ade, kappa_set)
    print(f"  shapes: train={Xv10_tr.shape}, test={Xv10_te.shape}, "
          f"time={time.time() - t0:.1f}s")
    standardize_in_place(Xv10_tr, Xv10_te)
    gc.collect()

    print("\nv10 linear ridge baseline:")
    v10_lin_acc, _ = linear_ridge_binary(
        Xv10_tr, tr_chiral_sh, Xv10_te, te_chiral,
        alphas=[0.001, 0.01, 0.1, 1.0], tag="v10-lin")

    print("\nv10 kernel ridge sweep:")
    v10_ker_acc, v10_ker_params = kernel_ridge_binary(
        Xv10_tr, tr_chiral_sh, Xv10_te, te_chiral,
        m_vals=[2000, 3000],
        alphas=[0.01, 0.1, 1.0],
        degree=2, tag="v10-ker")
    del Xv10_tr, Xv10_te
    gc.collect()

    # --- Face feature extraction ---
    print("\n[2/2] Extracting face multi-scale features...")
    t0 = time.time()
    Xf_tr = extract_face_multiscale(X_tr, ade, kappa_set)
    Xf_te = extract_face_multiscale(X_te, ade, kappa_set)
    print(f"  shapes: train={Xf_tr.shape}, test={Xf_te.shape}, "
          f"time={time.time() - t0:.1f}s")
    standardize_in_place(Xf_tr, Xf_te)
    gc.collect()

    print("\nFace linear ridge baseline:")
    face_lin_acc, _ = linear_ridge_binary(
        Xf_tr, tr_chiral_sh, Xf_te, te_chiral,
        alphas=[0.001, 0.01, 0.1, 1.0], tag="face-lin")

    print("\nFace kernel ridge sweep:")
    face_ker_acc, face_ker_params = kernel_ridge_binary(
        Xf_tr, tr_chiral_sh, Xf_te, te_chiral,
        m_vals=[1500, 2000],
        alphas=[0.1, 1.0],
        degree=2, tag="face-ker")
    del Xf_tr, Xf_te
    gc.collect()

    # --- Summary ---
    print()
    print(f"--- {name} RESULTS ---")
    print(f"Random chance:       50.00%")
    print(f"v10 linear ridge:    {v10_lin_acc * 100:.2f}%")
    print(f"v10 kernel ridge:    {v10_ker_acc * 100:.2f}%  "
          f"(m={v10_ker_params[0]}, alpha={v10_ker_params[1]})")
    print(f"face linear ridge:   {face_lin_acc * 100:.2f}%")
    print(f"face kernel ridge:   {face_ker_acc * 100:.2f}%  "
          f"(m={face_ker_params[0]}, alpha={face_ker_params[1]})")
    print()
    print(f"Face - v10 kernel delta: {(face_ker_acc - v10_ker_acc) * 100:+.2f}%")
    print(f"Face - v10 linear delta: {(face_lin_acc - v10_lin_acc) * 100:+.2f}%")

    return {
        "experiment": name,
        "train_digits": sorted(train_digits) if train_digits else None,
        "test_digits": sorted(test_digits) if test_digits else None,
        "train_size": int(X_tr.shape[0]),
        "test_size": int(X_te.shape[0]),
        "v10_linear_acc": float(v10_lin_acc),
        "v10_kernel_acc": float(v10_ker_acc),
        "v10_kernel_params": {"m": v10_ker_params[0],
                              "alpha": v10_ker_params[1]},
        "face_linear_acc": float(face_lin_acc),
        "face_kernel_acc": float(face_ker_acc),
        "face_kernel_params": {"m": face_ker_params[0],
                               "alpha": face_ker_params[1]},
        "face_minus_v10_kernel": float(face_ker_acc - v10_ker_acc),
        "face_minus_v10_linear": float(face_lin_acc - v10_lin_acc),
    }


def main():
    print("=" * 60)
    print("Chirality Benchmark — hflip-MNIST binary")
    print("=" * 60)
    if _berry_verified:
        print("Cl(3,0) Berry phase verified")

    out_dir = "hopf_chirality_bench"
    os.makedirs(out_dir, exist_ok=True)

    ade = get_ade()
    kappa_set = [3.0, 5.5, 8.0]
    print(f"\nMulti-scale kappa set: {kappa_set}")

    all_results = []

    # Experiment 1: baseline hflip detection on full MNIST
    # Both v10 and face should do well; this is the "easy" chirality test
    # because hflip shifts pixel positions which is trivially detectable.
    all_results.append(run_experiment(
        "full-MNIST hflip detection",
        ade, kappa_set,
        n_train=20000, n_test=5000,
        train_digits=None, test_digits=None))

    # Experiment 2: cross-digit generalization.
    # Train on {0,1,2,3,4} hflip detection, test on {5,6,7,8,9} hflip detection.
    # This is the HARD chirality test: features must encode chirality in
    # a digit-agnostic way. If v10 just memorizes "digit X has its ink
    # curving to the left", it will fail to transfer. If face features
    # encode a real orientation-reversing invariant via the signed Berry
    # phase, they should transfer better.
    all_results.append(run_experiment(
        "cross-digit hflip (train:0-4, test:5-9)",
        ade, kappa_set,
        n_train=20000, n_test=5000,
        train_digits={0, 1, 2, 3, 4}, test_digits={5, 6, 7, 8, 9}))

    # Experiment 3: reverse split for symmetry
    all_results.append(run_experiment(
        "cross-digit hflip (train:5-9, test:0-4)",
        ade, kappa_set,
        n_train=20000, n_test=5000,
        train_digits={5, 6, 7, 8, 9}, test_digits={0, 1, 2, 3, 4}))

    # --- Overall summary ---
    print()
    print("=" * 60)
    print("CHIRALITY BENCHMARK OVERALL SUMMARY")
    print("=" * 60)
    print(f"{'experiment':48s} {'v10-k':>8s} {'face-k':>8s} {'delta':>8s}")
    for r in all_results:
        name = r["experiment"]
        print(f"{name:48s} {r['v10_kernel_acc']*100:7.2f}% "
              f"{r['face_kernel_acc']*100:7.2f}% "
              f"{r['face_minus_v10_kernel']*100:+7.2f}%")
    print("=" * 60)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({
            "kappa_set": kappa_set,
            "berry_phase_verified": _berry_verified,
            "experiments": all_results,
        }, f, indent=2)
    print(f"\nSaved results to {out_dir}/results.json")


if __name__ == "__main__":
    main()
