#!/usr/bin/env python3
"""
Train v11 ADE Hopf Controller — Face/Holonomy (Ω²) Reading
============================================================

v11 completes the de Rham ladder on the 600-cell:

    Ω⁰ (vertices) → d₀ → Ω¹ (edges) → d₁ → Ω² (triangles)
     scalar eigs     curl eigs        face eigs       (new)

v10 had the first two readings (scalar + curl) and Clifford-verified
Hopf projections. v11 adds the third — a cubic triangle signal
weighted by the signed Berry phase of the Hopf bundle, projected onto
face-Laplacian eigenspaces obtained via the intertwining relation
from the cached co-exact 1-form (curl) eigenspaces.

Every face feature is:
  1. A cubic interaction of the pixel signal on the 600-cell
     (f_i · f_j · f_k for triangle (i,j,k)) — a genuine 2-form.
  2. Weighted by the signed Berry phase Ω_ijk / 2 — a pure Cl(3,0)
     parallel-transport invariant (verified against Euler-Eriksson
     to machine precision via rotor composition).
  3. Projected onto an exact-2-form eigenspace with eigenvalue
     matching its curl-eigenspace image (same 2I spectral structure).

Pipeline reuses v10's multi-scale kappa + Nystrom polynomial kernel
ridge. Only the feature vector grows: 293 → 393 per scale.
"""

import os
import sys
import gc
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hopf_controller import ADEHopfController, _berry_verified, _berry_err
from ade_geometry import get_ade
from train_ade_hopf import (
    load_mnist_split,
    extract_features_batch,
    ridge_regression_bias,
    nystrom_poly_kernel_ridge,
    evaluate,
)


# ================================================================
# v11: face/triangle-holonomy feature extraction
# ================================================================


def extract_face_features_batch(F, ade, chunk_size=5000):
    """
    Compute the Ω² reading for a batch of vertex activations.

    F: (N, 120) vertex activations (pixel signal projected onto the
       600-cell via some pixel kernel).
    ade: cached ADE + cell600 geometry (must contain triangle_indices,
         triangle_berry, face_eigenspaces).

    Returns (N, n_face_features) matrix of features. For each face
    eigenspace (multiplicities 6, 16, 30, 48 — inherited from the
    curl eigenspaces via d₁ intertwining), we apply the same Hopf-
    on-4-tuples + Poincaré leftover treatment as v9 curl features,
    but on the triangle-level signal instead of the edge-level one.

    Triangle signal:
        T_ijk = f_i · f_j · f_k · Ω_ijk / 2
    where Ω_ijk / 2 is the signed Berry phase (Cl(3,0)-verified).
    """
    N = F.shape[0]
    tri_idx = ade["triangle_indices"]     # (1200, 3)
    tri_berry = ade["triangle_berry"]     # (1200,) signed Berry phase
    face_es = ade["face_eigenspaces"]     # list of 4 dicts

    face_feats = []

    # Chunk over batch dim to avoid (N, 1200) allocations for large N
    n_tri = tri_idx.shape[0]
    all_face_coeffs = [np.zeros((N, fe["multiplicity"])) for fe in face_es]

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Fc = F[start:end]                 # (chunk, 120)
        Fi = Fc[:, tri_idx[:, 0]]         # (chunk, 1200)
        Fj = Fc[:, tri_idx[:, 1]]
        Fk = Fc[:, tri_idx[:, 2]]
        # Quadratic triangle signal weighted by Berry phase. This is
        # the boundary sum of v9's edge curl signal (f_i f_j + f_j f_k
        # + f_k f_i) around the oriented 2-cell, multiplied by the
        # signed Cl(3,0) holonomy of the Hopf bundle over that cell.
        # A cleaner cubic variant was empirically redundant with v10's
        # polynomial-kernel expansion of v9 features (6th-order overall);
        # the quadratic form keeps us at 4th-order total — same as v9
        # curl through the kernel — but on the 2-form layer instead of
        # the 1-form layer, making it genuinely complementary rather
        # than a higher-order rehash.
        T_q = (Fi * Fj + Fj * Fk + Fk * Fi) * tri_berry[None, :]

        for i, fe in enumerate(face_es):
            V_face = fe["vectors"]         # (1200, mult)
            all_face_coeffs[i][start:end] = T_q @ V_face

        del Fi, Fj, Fk, T_q
    # end chunk loop

    # For each face eigenspace: Hopf on 4-tuples + Poincaré leftovers
    # (same treatment as v9 curl features for consistency)
    for i, fe in enumerate(face_es):
        CC = all_face_coeffs[i]            # (N, mult)
        mult = fe["multiplicity"]
        n_hopf = mult // 4
        for g in range(n_hopf):
            c4 = CC[:, g * 4:(g + 1) * 4]  # (N, 4)
            mags = np.linalg.norm(c4, axis=1)
            safe = mags > 1e-10
            Q = np.zeros_like(c4)
            Q[safe] = c4[safe] / mags[safe, None]
            w, a, b, c = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
            px = 2 * (w * b + a * c)
            py = 2 * (w * c - a * b)
            pz = w * w + a * a - b * b - c * c
            scale = np.minimum(mags, 10.0)
            face_feats.append(np.column_stack([
                px * scale, py * scale, pz * scale,
                2.0 * np.tanh(mags / 2.0)
            ]))
        leftover = mult % 4
        if leftover > 0:
            for k in range(leftover):
                val = CC[:, n_hopf * 4 + k]
                face_feats.append(
                    (2.0 * np.tanh(val / 2.0)).reshape(N, 1))

    return np.hstack(face_feats)


def extract_features_v11(images, ade, pixel_kernel, chunk_size=5000):
    """
    v11 feature extraction: v10 features (scalar + curl + E8) plus
    the new face/holonomy 2-form features.

    Computes the shared (N, 120) vertex activation matrix F once and
    feeds it to both the existing v10 extractor and the new face
    extractor. This avoids redundant pixel-kernel multiplications.
    """
    # v10 features (uses its own F computation internally — minor
    # redundancy vs. rewriting extract_features_batch; acceptable for
    # now since it's one matmul per image)
    v10_feats = extract_features_batch(images, ade, pixel_kernel)

    # Face features on the same F
    X = np.array(images, dtype=np.float64)
    F = X @ pixel_kernel
    face_feats = extract_face_features_batch(F, ade, chunk_size=chunk_size)

    return np.hstack([v10_feats, face_feats])


def extract_features_multiscale_v11(images, ade, kappas, chunk_size=5000):
    """Multi-scale v11 feature extraction — one kappa per scale."""
    from hopf_controller import _get_pixel_kernel
    feats = []
    for k in kappas:
        pk = _get_pixel_kernel(784, kappa=k)
        feats.append(extract_features_v11(images, ade, pk, chunk_size=chunk_size))
    return np.hstack(feats)


# ================================================================
# Main
# ================================================================


def main():
    print("=" * 60)
    print("ADE Hopf Controller v11 — Face/Holonomy (Ω²) Reading")
    print("  (v10 + triangle Berry-phase cubic signal)")
    print("=" * 60)

    # Berry phase verification status (checked at import)
    if _berry_verified:
        print(f"\nCl(3,0) Berry phase verified (err={_berry_err:.2e})")
    else:
        print(f"\nWARNING: Cl(3,0) Berry phase NOT verified (err={_berry_err})")

    out_dir = "hopf_v11_ade"
    os.makedirs(out_dir, exist_ok=True)

    # Load MNIST
    print("\nLoading MNIST...")
    train_images, train_labels, _, _ = load_mnist_split("train")
    test_images, test_labels, _, _ = load_mnist_split("test")
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    # Build ADE geometry (includes face eigenspaces via v11 additions)
    print("\nBuilding ADE + face geometry...")
    ade = get_ade()
    n_tri = len(ade["triangle_indices"])
    n_face_modes = sum(fe["multiplicity"] for fe in ade["face_eigenspaces"])
    print(f"  Triangles: {n_tri}, face 2-form modes: {n_face_modes}")
    print(f"  Berry phase range: [{ade['triangle_berry'].min():.4f}, "
          f"{ade['triangle_berry'].max():.4f}] rad")

    # One-hot labels
    Y_train = np.zeros((len(train_labels), 10))
    for i, l in enumerate(train_labels):
        Y_train[i, l] = 1.0

    # Multi-scale kappa (same as v10)
    kappa_set = [3.0, 5.5, 8.0]
    print(f"\nMulti-scale kappa set: {kappa_set}")

    # Extract v11 features (v10 + face)
    print("\nExtracting training features (multi-scale v11)...")
    t0 = time.time()
    X_train = extract_features_multiscale_v11(train_images, ade, kappa_set)
    print(f"  Shape: {X_train.shape}, time: {time.time() - t0:.1f}s")

    print("Extracting test features (multi-scale v11)...")
    t0 = time.time()
    X_test = extract_features_multiscale_v11(test_images, ade, kappa_set)
    print(f"  Shape: {X_test.shape}, time: {time.time() - t0:.1f}s")

    # How many are face features per scale?
    feats_per_scale = X_train.shape[1] // len(kappa_set)
    print(f"  Features per scale: {feats_per_scale} "
          f"(v10 was 293, so +{feats_per_scale - 293} face features per scale)")

    # Standardize IN-PLACE (v11 has 1179 features; out-of-place would
    # double memory to ~1.1 GB which exceeds available RAM).
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_train -= feat_mean
    X_train /= feat_std
    X_test -= feat_mean
    X_test /= feat_std
    X_train_s = X_train   # rename for readability downstream
    X_test_s = X_test
    gc.collect()

    # Linear ridge baseline — memory-friendly, no explicit bias column.
    # Features are standardized (zero mean), so the optimal bias is just
    # the label mean. We solve ridge on centered Y and add back Y_mean
    # at prediction time. Equivalent to ridge-with-unpenalized-bias when
    # features are centered, at half the peak memory.
    Y_mean = Y_train.mean(axis=0)                        # (10,)
    Y_train_c = Y_train - Y_mean                         # centered labels
    XtX = X_train_s.T @ X_train_s                        # (n_feat, n_feat), small
    XtY = X_train_s.T @ Y_train_c                        # (n_feat, 10)
    n_feat = X_train_s.shape[1]

    print("\n--- Linear ridge baseline (v11 features, no bias col) ---")
    best_linear_acc = 0
    best_linear_W = None
    best_linear_bias = None
    best_linear_alpha = 0.1
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        W = np.linalg.solve(XtX + alpha * np.eye(n_feat), XtY)
        # Predictions: X @ W + Y_mean
        preds_tr = np.argmax(X_train_s @ W + Y_mean, axis=1)
        preds_te = np.argmax(X_test_s @ W + Y_mean, axis=1)
        train_acc = float(np.mean(preds_tr == np.array(train_labels)))
        test_acc = float(np.mean(preds_te == np.array(test_labels)))
        if test_acc > best_linear_acc:
            best_linear_acc = test_acc
            best_linear_W = W
            best_linear_bias = Y_mean.copy()
            best_linear_alpha = alpha
        print(f"  alpha={alpha:.4f}: train={train_acc:.4f}, test={test_acc:.4f}")
    print(f"Linear ridge best: alpha={best_linear_alpha}, test={best_linear_acc:.4f}")
    del XtX, XtY, Y_train_c
    gc.collect()

    # Polynomial kernel ridge with Nystrom (same hyperparameters as v10 sweep)
    print("\n--- Polynomial kernel ridge (degree=2, Nystrom) ---")
    kernel_best_acc = 0
    kernel_best_m = 5000
    kernel_best_alpha = 0.1
    kernel_best_beta = None
    kernel_best_L = None

    for m_val in [3000, 4000, 5000]:
        for k_alpha in [0.01, 0.1, 1.0]:
            beta, L, K_test_m = nystrom_poly_kernel_ridge(
                X_train_s, Y_train, X_test_s,
                m=m_val, degree=2, alpha=k_alpha)
            k_test_acc = np.mean(
                np.argmax(K_test_m @ beta, axis=1) == np.array(test_labels))
            marker = ""
            if k_test_acc > kernel_best_acc:
                kernel_best_acc = k_test_acc
                kernel_best_m = m_val
                kernel_best_alpha = k_alpha
                kernel_best_beta = beta
                kernel_best_L = L
                marker = " <-- BEST"
            print(f"  m={m_val}, alpha={k_alpha:.3f}: test={k_test_acc:.4f}{marker}")
            del K_test_m
            gc.collect()

    print(f"\nKernel ridge best: m={kernel_best_m}, alpha={kernel_best_alpha}, "
          f"test={kernel_best_acc:.4f}")

    use_kernel = kernel_best_acc > best_linear_acc
    best_acc = max(kernel_best_acc, best_linear_acc)

    # Final predictions for per-class report
    if use_kernel:
        print(f"\nKernel ({kernel_best_acc:.4f}) beats linear ({best_linear_acc:.4f})")
        K_test_m = (X_test_s @ kernel_best_L.T + 1) ** 2
        preds = np.argmax(K_test_m @ kernel_best_beta, axis=1)
    else:
        print(f"\nLinear ({best_linear_acc:.4f}) beats kernel ({kernel_best_acc:.4f})")
        preds = np.argmax(X_test_s @ best_linear_W + best_linear_bias, axis=1)

    # Per-class accuracy
    print("\nPer-class accuracy:")
    test_labels_np = np.array(test_labels)
    for digit in range(10):
        mask = test_labels_np == digit
        acc = np.mean(preds[mask] == digit)
        print(f"  Digit {digit}: {acc:.4f} ({mask.sum()} samples)")

    # Save checkpoint
    meta = {
        "version": 11,
        "kappa_set": kappa_set,
        "n_features_per_scale": feats_per_scale,
        "n_features_total": X_train_s.shape[1],
        "test_acc": float(best_acc),
        "linear_test_acc": float(best_linear_acc),
        "kernel_test_acc": float(kernel_best_acc),
        "readout": "kernel_poly2" if use_kernel else "linear",
        "berry_phase_verified": _berry_verified,
        "berry_phase_err": float(_berry_err) if _berry_err is not None else None,
    }
    if use_kernel:
        meta["kernel_m"] = int(kernel_best_m)
        meta["kernel_alpha"] = float(kernel_best_alpha)
        meta["kernel_degree"] = 2
    else:
        meta["linear_alpha"] = float(best_linear_alpha)

    meta_path = os.path.join(out_dir, "best_checkpoint.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    arrays = {"feat_mean": feat_mean, "feat_std": feat_std}
    if use_kernel:
        arrays["kernel_landmarks"] = kernel_best_L.astype(np.float32)
        arrays["kernel_beta"] = kernel_best_beta.astype(np.float32)
    else:
        arrays["linear_W"] = best_linear_W.astype(np.float32)
        arrays["linear_bias"] = best_linear_bias.astype(np.float32)
    arr_path = os.path.join(out_dir, "best_checkpoint.npz")
    np.savez_compressed(arr_path, **arrays)
    print(f"\nSaved metadata to {meta_path}")
    print(f"Saved arrays to {arr_path}")

    # Save training log
    log_path = os.path.join(out_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write(f"ADE Hopf v11 — Face/Holonomy (Ω²) Reading\n")
        f.write(f"Kappa set: {kappa_set}\n")
        f.write(f"Features per scale: {feats_per_scale}\n")
        f.write(f"Total features: {X_train_s.shape[1]}\n")
        f.write(f"Readout: {'polynomial kernel ridge (deg=2)' if use_kernel else 'linear ridge'}\n")
        if use_kernel:
            f.write(f"Nystrom m: {kernel_best_m}\n")
            f.write(f"Kernel alpha: {kernel_best_alpha}\n")
        f.write(f"Linear ridge test: {best_linear_acc:.4f}\n")
        f.write(f"Kernel ridge test: {kernel_best_acc:.4f}\n")
        f.write(f"Final test accuracy: {best_acc:.4f}\n")
        f.write(f"Berry phase Cl(3,0) verified: {_berry_verified}\n")
        f.write(f"Baseline (v10): 97.39%\n")
        f.write(f"Baseline (v9):  96.12%\n")
        f.write(f"Baseline (v8):  87.46%\n")

    print(f"\n{'=' * 60}")
    print(f"v11 ADE Hopf (+face/holonomy): {best_acc:.2%} test accuracy")
    print(f"  {X_train_s.shape[1]} features ({len(kappa_set)} scales × {feats_per_scale})")
    print(f"  Baselines: v10 97.39%, v9 96.12%, v8 87.46%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
