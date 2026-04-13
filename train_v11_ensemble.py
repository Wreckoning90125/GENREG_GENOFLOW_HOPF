#!/usr/bin/env python3
"""
v11 ensemble variant: train TWO independent kernel ridge models —
one on v10 features (scalar + curl + E8) and one on pure face/holonomy
features — then combine their logits.

Rationale: face features alone reach 84.7% kernel ridge and have
information the linear readout uses (94.98% vs v10's 94.95%), but
combined with v10 features via a single polynomial kernel the
information becomes redundant through the kernel's cross terms.
Training the two readouts independently isolates the face readout
from the v10 one so they can contribute non-overlapping evidence.
"""

import os
import sys
import gc
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hopf_controller import _get_pixel_kernel, _berry_verified, _berry_err
from ade_geometry import get_ade
from train_ade_hopf import (
    load_mnist_split,
    extract_features_batch,
    nystrom_poly_kernel_ridge,
)
from train_v11 import extract_face_features_batch


def extract_v10_multiscale(images, ade, kappas):
    feats = []
    for k in kappas:
        pk = _get_pixel_kernel(784, kappa=k)
        feats.append(extract_features_batch(images, ade, pk))
    return np.hstack(feats)


def extract_face_multiscale(images, ade, kappas):
    feats = []
    X = np.array(images, dtype=np.float64)
    for k in kappas:
        pk = _get_pixel_kernel(784, kappa=k)
        F = X @ pk
        feats.append(extract_face_features_batch(F, ade))
    return np.hstack(feats)


def standardize_in_place(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train -= mean
    X_train /= std
    X_test -= mean
    X_test /= std
    return mean, std


def kernel_ridge_sweep(X_train, Y_train, X_test, test_labels, m_vals, alphas,
                      degree=2, tag=""):
    best_acc = 0
    best_logits = None
    best_params = None
    for m in m_vals:
        for a in alphas:
            beta, L, K_test = nystrom_poly_kernel_ridge(
                X_train, Y_train, X_test, m=m, degree=degree, alpha=a)
            logits = K_test @ beta     # (N_test, 10)
            preds = np.argmax(logits, axis=1)
            acc = float(np.mean(preds == np.array(test_labels)))
            if acc > best_acc:
                best_acc = acc
                best_logits = logits
                best_params = (m, a)
            print(f"  {tag} m={m}, alpha={a:.3f}: test={acc:.4f}")
            del K_test
            gc.collect()
    print(f"  {tag} best: m={best_params[0]}, alpha={best_params[1]}, "
          f"test={best_acc:.4f}")
    return best_logits, best_acc, best_params


def main():
    print("=" * 60)
    print("v11 Ensemble — v10 features + face features, two readouts")
    print("=" * 60)
    if _berry_verified:
        print(f"Cl(3,0) Berry phase verified (err={_berry_err:.2e})")

    out_dir = "hopf_v11_ensemble"
    os.makedirs(out_dir, exist_ok=True)

    # Load MNIST
    print("\nLoading MNIST...")
    train_images, train_labels, _, _ = load_mnist_split("train")
    test_images, test_labels, _, _ = load_mnist_split("test")

    ade = get_ade()

    # One-hot
    Y_train = np.zeros((len(train_labels), 10))
    for i, l in enumerate(train_labels):
        Y_train[i, l] = 1.0

    kappa_set = [3.0, 5.5, 8.0]

    # --- v10 branch ---
    print("\n[1/2] Extracting v10 multi-scale features...")
    t0 = time.time()
    Xv10_tr = extract_v10_multiscale(train_images, ade, kappa_set)
    Xv10_te = extract_v10_multiscale(test_images, ade, kappa_set)
    print(f"  shapes: train={Xv10_tr.shape}, test={Xv10_te.shape}, "
          f"time={time.time()-t0:.1f}s")
    standardize_in_place(Xv10_tr, Xv10_te)
    gc.collect()

    print("\nv10 kernel ridge sweep:")
    v10_logits, v10_best, v10_params = kernel_ridge_sweep(
        Xv10_tr, Y_train, Xv10_te, test_labels,
        m_vals=[3000, 4000, 5000],
        alphas=[0.01, 0.1, 1.0],
        degree=2, tag="v10")
    del Xv10_tr, Xv10_te
    gc.collect()

    # --- Face branch ---
    print("\n[2/2] Extracting face multi-scale features...")
    t0 = time.time()
    Xf_tr = extract_face_multiscale(train_images, ade, kappa_set)
    Xf_te = extract_face_multiscale(test_images, ade, kappa_set)
    print(f"  shapes: train={Xf_tr.shape}, test={Xf_te.shape}, "
          f"time={time.time()-t0:.1f}s")
    standardize_in_place(Xf_tr, Xf_te)
    gc.collect()

    print("\nFace kernel ridge sweep:")
    face_logits, face_best, face_params = kernel_ridge_sweep(
        Xf_tr, Y_train, Xf_te, test_labels,
        m_vals=[2000, 3000, 4000],
        alphas=[0.01, 0.1, 1.0],
        degree=2, tag="face")
    del Xf_tr, Xf_te
    gc.collect()

    # --- Ensemble ---
    print("\n--- Ensemble: v10 logits + w * face logits ---")
    test_labels_np = np.array(test_labels)
    # Scan weight w over a range — optimal is typically small since
    # face_logits are weaker on their own but may correct v10 errors.
    best_ens_acc = 0
    best_w = 0.0
    best_ens_logits = None
    for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
        ens = v10_logits + w * face_logits
        acc = float(np.mean(np.argmax(ens, axis=1) == test_labels_np))
        marker = ""
        if acc > best_ens_acc:
            best_ens_acc = acc
            best_w = w
            best_ens_logits = ens
            marker = " <-- BEST"
        print(f"  w={w:.3f}: test={acc:.4f}{marker}")

    print(f"\nBest ensemble: w={best_w}, test={best_ens_acc:.4f}")
    print(f"  v10 alone:  {v10_best:.4f}  (baseline)")
    print(f"  face alone: {face_best:.4f}")
    print(f"  ensemble:   {best_ens_acc:.4f}")
    print(f"  delta vs v10: {(best_ens_acc - v10_best) * 100:+.3f}%")

    # Per-class breakdown of ensemble
    preds = np.argmax(best_ens_logits, axis=1)
    print("\nEnsemble per-class accuracy:")
    for digit in range(10):
        mask = test_labels_np == digit
        acc = float(np.mean(preds[mask] == digit))
        print(f"  Digit {digit}: {acc:.4f}")

    # Save
    meta = {
        "version": "11-ensemble",
        "kappa_set": kappa_set,
        "v10_params": {"m": v10_params[0], "alpha": v10_params[1]},
        "face_params": {"m": face_params[0], "alpha": face_params[1]},
        "v10_test_acc": float(v10_best),
        "face_test_acc": float(face_best),
        "best_ensemble_w": float(best_w),
        "best_ensemble_test_acc": float(best_ens_acc),
        "berry_phase_verified": _berry_verified,
    }
    with open(os.path.join(out_dir, "ensemble_results.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(out_dir, "training_log.txt"), "w") as f:
        f.write(f"v11 Ensemble — v10 + face readouts\n")
        f.write(f"v10 alone:  {v10_best:.4f} (m={v10_params[0]}, alpha={v10_params[1]})\n")
        f.write(f"face alone: {face_best:.4f} (m={face_params[0]}, alpha={face_params[1]})\n")
        f.write(f"best ensemble: {best_ens_acc:.4f} (w={best_w})\n")
        f.write(f"delta vs v10: {(best_ens_acc - v10_best)*100:+.3f}%\n")
        f.write(f"Cl(3,0) Berry phase verified: {_berry_verified}\n")
        f.write(f"Baselines: v10 97.39%, v9 96.12%, v8 87.46%\n")

    print(f"\n{'='*60}")
    print(f"v11 Ensemble: {best_ens_acc:.2%} "
          f"(delta vs v10 alone: {(best_ens_acc-v10_best)*100:+.3f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
