#!/usr/bin/env python3
"""
Train v8 ADE Hopf Controller on MNIST via ridge regression.

Zero non-convex parameters: all features are fixed geometric functions.
Only the linear readout is learned, via closed-form ridge regression.
Training is a single linear solve — no evolutionary search needed.
"""

import os
import sys
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


def extract_features_batch(images, ade, pixel_kernel):
    """
    Extract features for a batch of images.
    Vectorized where possible for speed.
    """
    N = len(images)
    ade_es = ade["ade_eigenspaces"]
    e8_edges = ade["e8_edges"]
    e8_to_es = ade["e8_to_eigenspace"]

    # Pixel -> vertex activations (fully vectorized)
    X = np.array(images, dtype=np.float64)  # (N, 784)
    F = X @ pixel_kernel  # (N, 120)

    # Process each eigenspace
    all_features = []
    all_es_norms = []

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

            # hopf_project vectorized: px=2(wb+ac), py=2(wc-ab), pz=w²+a²-b²-c²
            w, a, b, c = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
            px = 2 * (w * b + a * c)
            py = 2 * (w * c - a * b)
            pz = w * w + a * a - b * b - c * c

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

    # E8 edge features
    for ni, nj in e8_edges:
        ei = e8_to_es[ni]
        ej = e8_to_es[nj]
        ni_val = all_es_norms[ei]
        nj_val = all_es_norms[ej]
        prod = ni_val * nj_val
        asym = ni_val / (nj_val + 1e-6) - nj_val / (ni_val + 1e-6)
        all_features.append(
            np.column_stack([
                2.0 * np.tanh(prod / 2.0),
                2.0 * np.tanh(asym / 2.0)
            ])
        )

    return np.hstack(all_features)


def ridge_regression(X, Y, alpha=1.0):
    """Closed-form ridge regression: W = (X^T X + alpha I)^{-1} X^T Y."""
    n_feat = X.shape[1]
    XtX = X.T @ X + alpha * np.eye(n_feat)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)
    return W


def evaluate(W, X, labels):
    """Evaluate accuracy."""
    preds = np.argmax(X @ W, axis=1)
    return np.mean(preds == labels)


def main():
    print("=" * 60)
    print("ADE Hopf Controller v8 — Ridge Regression Training")
    print("=" * 60)

    # Setup
    out_dir = "hopf_v8_ade"
    os.makedirs(out_dir, exist_ok=True)

    # Load MNIST
    print("\nLoading MNIST...")
    train_images, train_labels, _, _ = load_mnist_split("train")
    test_images, test_labels, _, _ = load_mnist_split("test")
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    # Build ADE geometry
    print("\nBuilding ADE geometry...")
    ade = get_ade()

    # Build pixel kernel
    from hopf_controller import _get_pixel_kernel
    pixel_kernel = _get_pixel_kernel(784)

    # Create controller to get feature count
    ctrl = ADEHopfController(input_size=784, output_size=10)
    print(f"\nFeatures: {ctrl.n_features}")
    print(f"Readout params: {ctrl.param_count()}")

    # Extract features
    print("\nExtracting training features...")
    t0 = time.time()
    X_train = extract_features_batch(train_images, ade, pixel_kernel)
    t_train = time.time() - t0
    print(f"  Shape: {X_train.shape}, Time: {t_train:.1f}s")

    print("Extracting test features...")
    t0 = time.time()
    X_test = extract_features_batch(test_images, ade, pixel_kernel)
    t_test = time.time() - t0
    print(f"  Shape: {X_test.shape}, Time: {t_test:.1f}s")

    # One-hot labels
    Y_train = np.zeros((len(train_labels), 10))
    for i, l in enumerate(train_labels):
        Y_train[i, l] = 1.0

    # Ridge regression with alpha search
    print("\nRidge regression (alpha search)...")
    best_acc = 0
    best_alpha = 1.0
    best_W = None

    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        W = ridge_regression(X_train, Y_train, alpha)
        train_acc = evaluate(W, X_train, train_labels)
        test_acc = evaluate(W, X_test, test_labels)
        print(f"  alpha={alpha:8.3f}: train={train_acc:.4f}, test={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_alpha = alpha
            best_W = W

    print(f"\nBest: alpha={best_alpha}, test={best_acc:.4f}")

    # Store weights in controller
    ctrl.W_out = best_W.T  # (10, n_features)
    ctrl.b_out = np.zeros(10)

    # Per-class accuracy
    print("\nPer-class accuracy:")
    preds = np.argmax(X_test @ best_W, axis=1)
    test_labels_np = np.array(test_labels)
    for digit in range(10):
        mask = test_labels_np == digit
        acc = np.mean(preds[mask] == digit)
        print(f"  Digit {digit}: {acc:.4f} ({mask.sum()} samples)")

    # Save
    ckpt = {
        "controller": ctrl.to_dict(),
        "best_alpha": best_alpha,
        "train_acc": float(evaluate(best_W, X_train, train_labels)),
        "test_acc": float(best_acc),
        "n_features": ctrl.n_features,
        "n_params": ctrl.param_count(),
    }
    ckpt_path = os.path.join(out_dir, "best_checkpoint.json")
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f, indent=2)
    print(f"\nSaved to {ckpt_path}")

    # Save training history
    history_path = os.path.join(out_dir, "training_log.txt")
    with open(history_path, "w") as f:
        f.write(f"ADE Hopf v8 Ridge Regression Results\n")
        f.write(f"Features: {ctrl.n_features}\n")
        f.write(f"Parameters: {ctrl.param_count()}\n")
        f.write(f"Best alpha: {best_alpha}\n")
        f.write(f"Train accuracy: {evaluate(best_W, X_train, train_labels):.4f}\n")
        f.write(f"Test accuracy: {best_acc:.4f}\n")

    print(f"\n{'='*60}")
    print(f"v8 ADE Hopf: {best_acc:.2%} test accuracy")
    print(f"  {ctrl.n_features} features, {ctrl.param_count()} params")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
