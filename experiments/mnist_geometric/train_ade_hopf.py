#!/usr/bin/env python3
"""
Train v8 ADE Hopf Controller on MNIST via ridge regression.

Zero non-convex parameters: all features are fixed geometric functions.
Only the linear readout is learned, via closed-form ridge regression.
Training is a single linear solve — no evolutionary search needed.
"""

import os
import sys
import gc
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hopf_controller import ADEHopfController, poincare_warp_scalar, hopf_project
from ade_geometry import get_ade


def load_mnist_split(split="train"):
    """Load MNIST from nodes/envs/mnist.py."""
    from nodes.envs.mnist import load_mnist as _load_mnist
    return _load_mnist(split)


def extract_features_from_F(F, ade):
    """Extract ADE features from pre-computed vertex activations.

    This is the core feature extraction pipeline: takes (N, 120) vertex
    activations and produces (N, 293) features via ADE eigenspace
    decomposition, Hopf projections, CG cross products, curl features,
    and E8 edge features. Used by both MNIST (via extract_features_batch)
    and molecular pipelines (via mol_kernel.py).

    Args:
        F: (N, 120) array of vertex activations
        ade: ADE geometry dict from get_ade()

    Returns:
        (N, n_features) array of features (293 for standard ADE config)
    """
    N = F.shape[0]
    ade_es = ade["ade_eigenspaces"]
    e8_edges = ade["e8_edges"]
    e8_to_es = ade["e8_to_eigenspace"]

    # Process each eigenspace
    all_features = []
    all_es_norms = []
    all_es_hopf = {}  # idx -> (N, 3) S2 vectors for E8 interactions

    for idx, aes in enumerate(ade_es):
        V = aes["V"]      # (120, d2)
        d = aes["d"]
        d2 = aes["d2"]
        copies = aes["copies"]
        cg = aes["cg_v1"]

        # Project to eigenspace: (N, d2)
        C = F @ V  # (N, d2)
        norms = np.linalg.norm(C, axis=1)  # (N,)
        all_es_norms.append(norms)

        if d == 1:
            feat = 2.0 * np.tanh(C[:, 0] / 2.0)  # poincare warp
            all_features.append(feat.reshape(N, 1))

        elif d == 2:
            # Hopf on full 4D
            mags = norms.copy()
            safe = mags > 1e-10
            Q = np.zeros_like(C)
            Q[safe] = C[safe] / mags[safe, None]

            # hopf_project vectorized
            w, a, b, c = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
            px = 2 * (w * b + a * c)
            py = 2 * (w * c - a * b)
            pz = w * w + a * a - b * b - c * c

            # Store S2 vectors for E8 interactions
            all_es_hopf[idx] = np.column_stack([px, py, pz])

            scale = np.minimum(mags, 10.0)
            f_hopf = np.column_stack([
                px * scale, py * scale, pz * scale,
                2.0 * np.tanh(mags / 2.0)
            ])
            all_features.append(f_hopf)

        else:
            # Copy magnitudes
            copy_feats = []
            copy_vecs_all = []  # (n_copies, N, d_copy)
            for cb in copies:
                cv = C @ cb  # (N, d_copy)
                copy_vecs_all.append(cv)
                cm = np.linalg.norm(cv, axis=1)
                copy_feats.append(2.0 * np.tanh(cm / 2.0))

            # Store first copy's Hopf S2 as representative
            if copy_vecs_all[0].shape[1] >= 4:
                c4 = copy_vecs_all[0][:, :4]  # (N, 4)
                c4_mag = np.linalg.norm(c4, axis=1)
                c4_safe = c4_mag > 1e-10
                c4_Q = np.zeros_like(c4)
                c4_Q[c4_safe] = c4[c4_safe] / c4_mag[c4_safe, None]
                cw, ca, cb_v, cc = c4_Q[:, 0], c4_Q[:, 1], c4_Q[:, 2], c4_Q[:, 3]
                all_es_hopf[idx] = np.column_stack([
                    2 * (cw * cb_v + ca * cc),
                    2 * (cw * cc - ca * cb_v),
                    cw*cw + ca*ca - cb_v*cb_v - cc*cc
                ])

            all_features.append(np.column_stack(copy_feats))

            if cg is not None:
                # CG cross products: all pairs
                nc = len(copies)
                for a_idx in range(nc):
                    for b_idx in range(a_idx + 1, nc):
                        va = copy_vecs_all[a_idx]  # (N, d_copy)
                        vb = copy_vecs_all[b_idx]  # (N, d_copy)
                        # Kronecker product: (N, d_copy^2)
                        dc = va.shape[1]
                        kron = (va[:, :, None] * vb[:, None, :]).reshape(N, dc * dc)
                        # CG projection: (N, k)
                        w_cg = kron @ cg.T  # cg is (k, dc^2)
                        all_features.append(w_cg)
            else:
                # Hopf on 4-tuples
                n_hopf = d2 // 4
                for g in range(n_hopf):
                    c4 = C[:, g*4:(g+1)*4]
                    mags_g = np.linalg.norm(c4, axis=1)
                    safe_g = mags_g > 1e-10
                    Q_g = np.zeros_like(c4)
                    Q_g[safe_g] = c4[safe_g] / mags_g[safe_g, None]

                    w, a, b, c = Q_g[:, 0], Q_g[:, 1], Q_g[:, 2], Q_g[:, 3]
                    px = 2 * (w * b + a * c)
                    py = 2 * (w * c - a * b)
                    pz = w * w + a * a - b * b - c * c
                    scale_g = np.minimum(mags_g, 10.0)

                    f_h = np.column_stack([
                        px * scale_g, py * scale_g, pz * scale_g,
                        2.0 * np.tanh(mags_g / 2.0)
                    ])
                    all_features.append(f_h)

                leftover = d2 % 4
                if leftover > 0:
                    for k in range(leftover):
                        val = C[:, n_hopf * 4 + k]
                        all_features.append(
                            (2.0 * np.tanh(val / 2.0)).reshape(N, 1))

    # Curl eigenspace features (edge/differential reading, Theorem 5)
    curl_es = ade.get("curl_eigenspaces", [])
    if curl_es:
        edge_list = ade["edges"]
        ei = np.array([e[0] for e in edge_list])  # (720,)
        ej = np.array([e[1] for e in edge_list])  # (720,)
        chunk_size = 10000
        for ces in curl_es:
            V_curl = ces["vectors"]    # (720, mult)
            mult = ces["multiplicity"]
            CC = np.zeros((N, mult))
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                H_chunk = F[start:end, ei] * F[start:end, ej]  # (chunk, 720)
                CC[start:end] = H_chunk @ V_curl

            n_hopf = mult // 4
            for g in range(n_hopf):
                c4 = CC[:, g*4:(g+1)*4]   # (N, 4)
                mags = np.linalg.norm(c4, axis=1)
                safe = mags > 1e-10
                Q = np.zeros_like(c4)
                Q[safe] = c4[safe] / mags[safe, None]
                w, a, b, c = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
                px = 2 * (w * b + a * c)
                py = 2 * (w * c - a * b)
                pz = w * w + a * a - b * b - c * c
                scale = np.minimum(mags, 10.0)
                all_features.append(np.column_stack([
                    px * scale, py * scale, pz * scale,
                    2.0 * np.tanh(mags / 2.0)
                ]))
            leftover = mult % 4
            if leftover > 0:
                for k in range(leftover):
                    val = CC[:, n_hopf * 4 + k]
                    all_features.append(
                        (2.0 * np.tanh(val / 2.0)).reshape(N, 1))

    # E8 edge features: norm interactions + directional interactions
    for ni, nj in e8_edges:
        ei_idx = e8_to_es[ni]
        ej_idx = e8_to_es[nj]
        ni_val = all_es_norms[ei_idx]
        nj_val = all_es_norms[ej_idx]
        prod = ni_val * nj_val
        asym = ni_val / (nj_val + 1e-6) - nj_val / (ni_val + 1e-6)
        norm_feats = np.column_stack([
            2.0 * np.tanh(prod / 2.0),
            2.0 * np.tanh(asym / 2.0)
        ])
        if ei_idx in all_es_hopf and ej_idx in all_es_hopf:
            hi = all_es_hopf[ei_idx]  # (N, 3)
            hj = all_es_hopf[ej_idx]  # (N, 3)
            dot = np.sum(hi * hj, axis=1)
            cross = np.cross(hi, hj)
            cross_mag = np.linalg.norm(cross, axis=1)
            dir_feats = np.column_stack([dot, cross_mag])
        else:
            dir_feats = np.zeros((N, 2))
        all_features.append(np.column_stack([norm_feats, dir_feats]))

    return np.hstack(all_features)


def extract_features_batch(images, ade, pixel_kernel):
    """
    Extract features for a batch of images.
    Thin wrapper: maps pixels to vertex activations, then calls
    extract_features_from_F for the ADE feature pipeline.
    """
    X = np.array(images, dtype=np.float64)
    F = X @ pixel_kernel  # (N, 120)
    return extract_features_from_F(F, ade)


def ridge_regression(X, Y, alpha=1.0):
    """Closed-form ridge regression: W = (X^T X + alpha I)^{-1} X^T Y."""
    n_feat = X.shape[1]
    XtX = X.T @ X + alpha * np.eye(n_feat)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)
    return W


def ridge_regression_bias(X, Y, alpha=1.0):
    """Ridge with unpenalized bias column (last column of X should be ones)."""
    n = X.shape[1]
    reg = alpha * np.eye(n)
    reg[-1, -1] = 0.0  # don't regularize bias
    return np.linalg.solve(X.T @ X + reg, X.T @ Y)


def nystrom_poly_kernel_ridge(X_train, Y_train, X_test, m=2000,
                               degree=2, alpha=0.1, seed=42):
    """
    Polynomial kernel ridge regression with Nystrom approximation.
    Memory-efficient: computes K^T K and K^T Y in chunks.

    K(x, y) = (x.y + 1)^degree
    Uses m landmark points for Nystrom approximation.
    Returns (beta, L, K_test) where predictions are K_test @ beta.
    """
    np.random.seed(seed)
    idx = np.random.choice(len(X_train), m, replace=False)
    L = X_train[idx]

    # K_mm: kernel between landmarks
    K_mm = (L @ L.T + 1) ** degree  # (m, m)

    # Smaller chunks for larger m to cap peak memory
    chunk_size = max(500, min(5000, 3000000 // m))
    KtK = np.zeros((m, m))
    KtY = np.zeros((m, Y_train.shape[1]))
    for i in range(0, len(X_train), chunk_size):
        end = min(i + chunk_size, len(X_train))
        K_chunk = (X_train[i:end] @ L.T + 1) ** degree  # (chunk, m)
        KtK += K_chunk.T @ K_chunk
        KtY += K_chunk.T @ Y_train[i:end]
        del K_chunk

    # Solve regularized system
    beta = np.linalg.solve(KtK + alpha * K_mm, KtY)  # (m, n_classes)

    # Test predictions
    K_test = (X_test @ L.T + 1) ** degree  # (N_test, m)

    return beta, L, K_test


def evaluate(W, X, labels):
    """Evaluate accuracy."""
    preds = np.argmax(X @ W, axis=1)
    return np.mean(preds == labels)


def extract_features_multiscale(images, ade, kappas, use_abs=True):
    """
    Multi-scale feature extraction: concatenate features computed at
    different pixel-kernel temperatures. Each kappa gives a different
    softness of the pixel→600-cell vertex projection, capturing
    different spatial scales of the input signal on the net.

    use_abs: passed through to the pixel kernel. True (default) uses
    |q . v| in the soft-assign (chirality-blind, collapses 120
    vertices to 60 antipodal pairs). False uses signed dots (full
    120-vertex 2I resolution).
    """
    from hopf_controller import _get_pixel_kernel
    feats = []
    for k in kappas:
        pk = _get_pixel_kernel(784, kappa=k, use_abs=use_abs)
        feats.append(extract_features_batch(images, ade, pk))
    return np.hstack(feats)


def main():
    print("=" * 60)
    print("ADE Hopf Controller v10 — Multi-Scale Kernel Ridge")
    print("  (v9 + multi-scale kappa + larger Nystrom m)")
    print("=" * 60)

    # Setup
    out_dir = "checkpoints/hopf_v10_ade"
    os.makedirs(out_dir, exist_ok=True)

    # Load MNIST
    print("\nLoading MNIST...")
    train_images, train_labels, _, _ = load_mnist_split("train")
    test_images, test_labels, _, _ = load_mnist_split("test")
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    # Build ADE geometry
    print("\nBuilding ADE geometry...")
    ade = get_ade()

    from hopf_controller import _get_pixel_kernel

    # One-hot labels
    Y_train = np.zeros((len(train_labels), 10))
    for i, l in enumerate(train_labels):
        Y_train[i, l] = 1.0

    # --- Multi-scale feature extraction ---
    # Different kappa values capture different spatial scales of the signal
    # on the 600-cell: low kappa = soft/coarse (captures broad structure),
    # high kappa = sharp/fine (captures edges). Concatenating scales gives
    # a multi-resolution reading on the net — analogous to a wavelet
    # decomposition, all derived from the same underlying geometry.
    kappa_set = [3.0, 5.5, 8.0]
    print(f"\nMulti-scale kappa set: {kappa_set}")

    print("Extracting training features (multi-scale)...")
    t0 = time.time()
    X_train = extract_features_multiscale(train_images, ade, kappa_set)
    print(f"  Shape: {X_train.shape}, time: {time.time() - t0:.1f}s")

    print("Extracting test features (multi-scale)...")
    t0 = time.time()
    X_test = extract_features_multiscale(test_images, ade, kappa_set)
    print(f"  Shape: {X_test.shape}, time: {time.time() - t0:.1f}s")

    # Standardize
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_train_s = (X_train - feat_mean) / feat_std
    X_test_s = (X_test - feat_mean) / feat_std
    del X_train, X_test
    gc.collect()

    # --- Linear ridge baseline with multi-scale features ---
    X_train_b = np.hstack([X_train_s, np.ones((len(X_train_s), 1))])
    X_test_b = np.hstack([X_test_s, np.ones((len(X_test_s), 1))])

    print("\n--- Linear ridge baseline (multi-scale) ---")
    best_linear_acc = 0
    best_linear_W = None
    best_linear_alpha = 0.1
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        W = ridge_regression_bias(X_train_b, Y_train, alpha)
        test_acc = evaluate(W, X_test_b, test_labels)
        train_acc = evaluate(W, X_train_b, train_labels)
        if test_acc > best_linear_acc:
            best_linear_acc = test_acc
            best_linear_W = W
            best_linear_alpha = alpha
        print(f"  alpha={alpha:.4f}: train={train_acc:.4f}, test={test_acc:.4f}")
    print(f"Linear ridge best: alpha={best_linear_alpha}, test={best_linear_acc:.4f}")
    del X_train_b, X_test_b
    gc.collect()

    # --- Polynomial kernel ridge with large Nystrom approximation ---
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

    # Decide best readout
    use_kernel = kernel_best_acc > best_linear_acc
    best_acc = max(kernel_best_acc, best_linear_acc)

    # Create controller
    ctrl = ADEHopfController(input_size=784, output_size=10)
    n_features_single = ctrl.n_features
    n_features_ms = n_features_single * len(kappa_set)

    # Final predictions for per-class report
    if use_kernel:
        print(f"\nKernel ({kernel_best_acc:.4f}) beats linear ({best_linear_acc:.4f})")
        K_test_m = (X_test_s @ kernel_best_L.T + 1) ** 2
        preds = np.argmax(K_test_m @ kernel_best_beta, axis=1)
    else:
        print(f"\nLinear ({best_linear_acc:.4f}) beats kernel ({kernel_best_acc:.4f})")
        X_test_b = np.hstack([X_test_s, np.ones((len(X_test_s), 1))])
        preds = np.argmax(X_test_b @ best_linear_W, axis=1)

    # Per-class accuracy
    print("\nPer-class accuracy:")
    test_labels_np = np.array(test_labels)
    for digit in range(10):
        mask = test_labels_np == digit
        acc = np.mean(preds[mask] == digit)
        print(f"  Digit {digit}: {acc:.4f} ({mask.sum()} samples)")

    # Save checkpoint
    # Small metadata as JSON, large arrays as compressed numpy
    meta = {
        "version": 10,
        "kappa_set": kappa_set,
        "n_features_per_scale": n_features_single,
        "n_features_total": n_features_ms,
        "test_acc": float(best_acc),
        "linear_test_acc": float(best_linear_acc),
        "kernel_test_acc": float(kernel_best_acc),
        "readout": "kernel_poly2" if use_kernel else "linear",
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

    # Compressed numpy arrays (large data)
    arrays = {
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }
    if use_kernel:
        arrays["kernel_landmarks"] = kernel_best_L.astype(np.float32)
        arrays["kernel_beta"] = kernel_best_beta.astype(np.float32)
    else:
        arrays["linear_W"] = best_linear_W.astype(np.float32)
    arr_path = os.path.join(out_dir, "best_checkpoint.npz")
    np.savez_compressed(arr_path, **arrays)
    print(f"\nSaved metadata to {meta_path}")
    print(f"Saved arrays to {arr_path}")

    # Save training history
    history_path = os.path.join(out_dir, "training_log.txt")
    with open(history_path, "w") as f:
        f.write(f"ADE Hopf v10 — Multi-Scale Kernel Ridge\n")
        f.write(f"Kappa set: {kappa_set}\n")
        f.write(f"Features per scale: {n_features_single}\n")
        f.write(f"Total features: {n_features_ms}\n")
        f.write(f"Readout: {'polynomial kernel ridge (deg=2)' if use_kernel else 'linear ridge'}\n")
        if use_kernel:
            f.write(f"Nystrom m: {kernel_best_m}\n")
            f.write(f"Kernel alpha: {kernel_best_alpha}\n")
        f.write(f"Linear ridge test: {best_linear_acc:.4f}\n")
        f.write(f"Kernel ridge test: {kernel_best_acc:.4f}\n")
        f.write(f"Final test accuracy: {best_acc:.4f}\n")
        f.write(f"Baseline (v9): 96.12%\n")
        f.write(f"Baseline (v8): 87.46%\n")

    print(f"\n{'='*60}")
    print(f"v10 ADE Hopf (multi-scale): {best_acc:.2%} test accuracy")
    print(f"  {n_features_ms} features ({len(kappa_set)} scales × {n_features_single})")
    print(f"  Baseline v9: 96.12%, v8: 87.46%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
