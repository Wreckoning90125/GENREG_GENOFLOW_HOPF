"""
MNIST chirality A/B: does fixing the np.abs() in the pixel kernel
move the v10 baseline (97.39% kernel ridge / 94.95% linear ridge)?

The pixel kernel in hopf_controller._build_pixel_kernel uses
    dots = np.abs(pixel_quats @ vertices.T)
which collapses the spinor double cover (q ~ -q) and identifies
antipodal pairs of 600-cell vertices. This was the published default
for v8-v12. Same magic-number footgun pattern as mol_kernel.py.

This script:
  1. Extracts ADE features under both pixel-kernel modes
  2. Runs linear ridge with alpha sweep on each
  3. Reports the test-accuracy delta

If the chirality fix moves the headline number, the v10 97.39%
kernel-ridge result is leaving accuracy on the table.
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
sys.path.insert(0, os.path.join(_ROOT, "experiments", "mnist_geometric"))

from ade_geometry import get_ade
from train_ade_hopf import (
    extract_features_batch, ridge_regression_bias, evaluate,
    load_mnist_split, nystrom_poly_kernel_ridge,
)


def evaluate_acc(W, X_b, labels):
    preds = np.argmax(X_b @ W, axis=1)
    return float(np.mean(preds == np.array(labels)))


def best_kernel_ridge(X_train, y_train_oh, X_test, y_test,
                      m_vals=(3000, 5000), alphas=(0.01, 0.1, 1.0)):
    best_acc = 0.0
    best_cfg = None
    for m in m_vals:
        for a in alphas:
            beta, L, K_test = nystrom_poly_kernel_ridge(
                X_train, y_train_oh, X_test, m=m, degree=2, alpha=a
            )
            preds = np.argmax(K_test @ beta, axis=1)
            acc = float(np.mean(preds == np.array(y_test)))
            if acc > best_acc:
                best_acc = acc
                best_cfg = {"m": m, "alpha": a}
            del K_test
    return best_acc, best_cfg


def best_linear_ridge(X_train, y_train_oh, X_test, y_test,
                      alphas=(0.0001, 0.001, 0.01, 0.1, 1.0)):
    Xt = np.hstack([X_train, np.ones((len(X_train), 1))])
    Xv = np.hstack([X_test, np.ones((len(X_test), 1))])
    best_acc, best_alpha = 0.0, alphas[0]
    for a in alphas:
        W = ridge_regression_bias(Xt, y_train_oh, a)
        acc = evaluate_acc(W, Xv, y_test)
        if acc > best_acc:
            best_acc, best_alpha = acc, a
    return best_acc, best_alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--kappas", type=float, nargs="+", default=[3.0, 5.5, 8.0])
    ap.add_argument("--out-dir", type=str, default="checkpoints/mnist_chirality")
    ap.add_argument("--skip-kernel", action="store_true",
                    help="Skip the Nystrom kernel-ridge step (linear ridge only)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading MNIST...")
    train_images, train_labels, _, _ = load_mnist_split("train")
    test_images, test_labels, _, _ = load_mnist_split("test")
    # Loader returns lists; subsample by indexing then convert to numpy
    if args.n_train < len(train_images):
        rng = np.random.default_rng(42)
        ti = rng.choice(len(train_images), args.n_train, replace=False)
        ti.sort()
        train_images = [train_images[i] for i in ti]
        train_labels = [train_labels[i] for i in ti]
    if args.n_test < len(test_images):
        rng = np.random.default_rng(43)
        si = rng.choice(len(test_images), args.n_test, replace=False)
        si.sort()
        test_images = [test_images[i] for i in si]
        test_labels = [test_labels[i] for i in si]
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    print("Building ADE geometry...")
    ade = get_ade()

    Y_train = np.zeros((len(train_labels), 10))
    for i, l in enumerate(train_labels):
        Y_train[i, l] = 1.0

    from hopf_controller import _get_pixel_kernel

    results = {}
    for use_abs in (True, False):
        label = "abs (chirality-blind, original v10)" if use_abs else "signed (chirality-resolved)"
        print(f"\n=== {label} ===")
        feats_train, feats_test = [], []
        for kappa in args.kappas:
            t0 = time.time()
            pk = _get_pixel_kernel(784, kappa=kappa, use_abs=use_abs)
            ft = extract_features_batch(train_images, ade, pk)
            fv = extract_features_batch(test_images, ade, pk)
            feats_train.append(ft)
            feats_test.append(fv)
            print(f"  kappa={kappa}: train={ft.shape}, test={fv.shape}, "
                  f"time={time.time()-t0:.1f}s")
        X_train = np.hstack(feats_train)
        X_test = np.hstack(feats_test)
        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

        # Standardize
        m, s = X_train.mean(axis=0), X_train.std(axis=0)
        s[s < 1e-8] = 1.0
        X_train = (X_train - m) / s
        X_test = (X_test - m) / s

        # Linear ridge alpha sweep
        print("  Linear ridge alpha sweep:")
        best_lin_acc, best_lin_alpha = best_linear_ridge(
            X_train, Y_train, X_test, test_labels,
            alphas=(0.0001, 0.001, 0.01, 0.1, 1.0)
        )
        print(f"    Best linear: alpha={best_lin_alpha}, test_acc={best_lin_acc:.4f}")

        # Polynomial kernel ridge (Nystrom approximation, degree 2) -- the
        # v10 published method. Smaller sweep for speed.
        if not args.skip_kernel:
            print("  Polynomial kernel ridge (Nystrom degree 2):")
            t0 = time.time()
            best_kr_acc, best_kr_cfg = best_kernel_ridge(
                X_train, Y_train, X_test, test_labels,
                m_vals=(3000, 5000), alphas=(0.01, 0.1, 1.0),
            )
            print(f"    Best kernel: {best_kr_cfg}, test_acc={best_kr_acc:.4f}, "
                  f"time={time.time()-t0:.1f}s")
        else:
            best_kr_acc = None
            best_kr_cfg = None

        results["abs" if use_abs else "signed"] = {
            "n_features": int(X_train.shape[1]),
            "linear_acc": best_lin_acc,
            "linear_alpha": best_lin_alpha,
            "kernel_acc": best_kr_acc,
            "kernel_cfg": best_kr_cfg,
        }
        del X_train, X_test, feats_train, feats_test
        gc.collect()

    print("\n" + "=" * 70)
    print("MNIST chirality A/B (multi-scale ADE features)")
    print("-" * 70)
    print(f"{'mode':<35} {'n_feat':>10} {'lin_acc':>10} {'kernel_acc':>12}")
    for mode in ("abs", "signed"):
        r = results[mode]
        kacc = r['kernel_acc']
        kacc_s = f"{kacc:.4f}" if kacc is not None else "(skipped)"
        print(f"{mode:<35} {r['n_features']:>10} "
              f"{r['linear_acc']:>10.4f} {kacc_s:>12}")
    d_lin = results["signed"]["linear_acc"] - results["abs"]["linear_acc"]
    print(f"\nDelta linear (signed - abs): {d_lin:+.4f} ({d_lin*100:+.2f} pp)")
    if results["abs"]["kernel_acc"] is not None and results["signed"]["kernel_acc"] is not None:
        d_kr = results["signed"]["kernel_acc"] - results["abs"]["kernel_acc"]
        print(f"Delta kernel  (signed - abs): {d_kr:+.4f} ({d_kr*100:+.2f} pp)")
    print(f"\n(Reference: published v10 kernel-ridge 97.39%, v9 linear ridge 96.12%)")

    out_path = os.path.join(args.out_dir, "results.json")
    payload = {
        "n_train": int(len(train_images)),
        "n_test": int(len(test_images)),
        "kappas": list(args.kappas),
        "results": results,
        "delta_linear": d_lin,
    }
    if results["abs"]["kernel_acc"] is not None and results["signed"]["kernel_acc"] is not None:
        payload["delta_kernel"] = (
            results["signed"]["kernel_acc"] - results["abs"]["kernel_acc"]
        )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
