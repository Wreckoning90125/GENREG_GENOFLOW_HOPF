#!/usr/bin/env python3
"""
GENREG × Hopf MNIST — Rotor-Only ES with Analytic Readout

Separates convex from non-convex optimization:
  - Readout weights: solved analytically via ridge regression (convex)
  - Rotors: searched via ES (non-convex, only 90 DOF)

This is MUCH more efficient than searching all 1330 params.
The ridge regression guarantees optimal readout for any rotor configuration.
ES only needs to find good rotors — a 90-dimensional search.

Usage:
    python train_hopf_rotor_es.py [--generations 200] [--pop-size 100]
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np

from hopf_controller import (VertexHopfController, qmul, qnormalize,
                              hopf_project, poincare_warp_scalar,
                              _get_pixel_kernel, _get_geo)
from nodes.envs.mnist import load_mnist


# ================================================================
# Vectorized Hopf operations
# ================================================================

def qmul_batch(Q, r):
    """Multiply batch of quaternions Q (n, 4) by single quaternion r (4,)."""
    w1, x1, y1, z1 = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    w2, x2, y2, z2 = r
    return np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def hopf_project_batch(Q):
    """Hopf project batch of quaternions Q (n, 4) → (n, 3)."""
    w, a, b, c = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    return np.column_stack([
        2*(w*b + a*c),
        2*(w*c - a*b),
        w*w + a*a - b*b - c*c
    ])


def compute_features_batch(f_all, rotors):
    """
    Compute Hopf features for all samples given rotors.

    Args:
        f_all: (n_samples, 120) vertex activations
        rotors: list of 30 quaternions (4,)

    Returns:
        F: (n_samples, 120) feature matrix
    """
    n = f_all.shape[0]
    n_groups = len(rotors)
    F = np.empty((n, n_groups * 4))

    for g in range(n_groups):
        c4 = f_all[:, g*4:(g+1)*4]  # (n, 4)
        mag = np.linalg.norm(c4, axis=1)  # (n,)
        base = g * 4

        # Normalize
        safe_mag = np.maximum(mag, 1e-10)
        Q = c4 / safe_mag[:, None]  # (n, 4)

        # Apply rotor
        Q_rot = qmul_batch(Q, rotors[g])
        # Normalize after multiplication
        norms = np.linalg.norm(Q_rot, axis=1, keepdims=True)
        Q_rot = Q_rot / np.maximum(norms, 1e-12)

        # Hopf project
        P = hopf_project_batch(Q_rot)  # (n, 3)

        # Scale by magnitude
        scale = np.minimum(mag, 10.0)[:, None]
        F[:, base:base+3] = P * scale

        # Poincaré-warped magnitude
        F[:, base+3] = 2.0 * np.tanh(mag / 2.0)

    return F


def rotors_to_flat(rotors):
    """Flatten 30 rotors to a (120,) vector."""
    return np.concatenate(rotors)


def flat_to_rotors(flat, n_groups=30):
    """Unflatten (120,) vector to list of 30 normalized quaternions."""
    rotors = []
    for g in range(n_groups):
        q = flat[g*4:(g+1)*4].copy()
        n = np.linalg.norm(q)
        if n > 1e-12:
            q /= n
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        rotors.append(q)
    return rotors


# ================================================================
# Adam optimizer
# ================================================================

class Adam:
    def __init__(self, n_params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def compute_ranks(fitnesses):
    n = len(fitnesses)
    ranks = np.zeros(n)
    sorted_idx = np.argsort(fitnesses)
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank
    return ranks / (n - 1) - 0.5


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GENREG × Hopf MNIST — Rotor-Only ES")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=20000,
                        help="Training samples for ridge regression")
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Rotor perturbation noise")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str,
                        default="checkpoints/hopf_v7_rotor_es")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-rotated", action="store_true")
    args = parser.parse_args()

    if args.pop_size % 2 != 0:
        args.pop_size += 1

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print("=" * 60)
    print("GENREG x Hopf MNIST — Rotor-Only ES")
    print("=" * 60)
    print(f"Population:     {args.pop_size}")
    print(f"Generations:    {args.generations}")
    print(f"Train samples:  {args.n_train}")
    print(f"Test samples:   {args.n_test}")
    print(f"Sigma:          {args.sigma}")
    print(f"Learning rate:  {args.lr}")
    print(f"Ridge lambda:   {args.ridge_lambda}")

    # Load data and pre-compute vertex activations
    geo = _get_geo()
    kernel = _get_pixel_kernel(784)

    images, labels, _, _ = load_mnist("train")
    test_images, test_labels, _, _ = load_mnist("test")

    print("Pre-computing vertex activations...")
    train_idx = random.sample(range(len(images)), args.n_train)
    f_train = np.array([images[i] for i in train_idx]) @ kernel
    y_train = np.array([labels[i] for i in train_idx])

    f_test = np.array(test_images[:args.n_test]) @ kernel
    y_test = np.array(test_labels[:args.n_test])

    # Also prepare rotated test set
    f_test_rotated = None
    if args.test_rotated:
        from nodes.envs.mnist import _rotate_image
        rotated = []
        for i in range(args.n_test):
            angle = random.uniform(-180, 180)
            rot_img = _rotate_image(test_images[i], 28, 28, angle)
            rotated.append(rot_img)
        f_test_rotated = np.array(rotated) @ kernel

    # One-hot targets
    Y_oh = np.zeros((args.n_train, 10))
    for i, y in enumerate(y_train):
        Y_oh[i, y] = 1.0

    n_groups = 30
    n_rotor_params = n_groups * 4  # 120
    n_features = n_groups * 4  # 120

    print(f"Rotor params:   {n_rotor_params} ({n_groups * 3} DOF)")
    print(f"Features:       {n_features}")
    print(f"Readout params: {n_features * 10 + 10} (solved by ridge)")
    print(f"Total params:   {n_rotor_params + n_features * 10 + 10}")

    def evaluate_rotors(rotors):
        """Compute features → ridge regression → test accuracy."""
        F = compute_features_batch(f_train, rotors)
        # Ridge regression
        W = np.linalg.solve(F.T @ F + args.ridge_lambda * np.eye(n_features),
                            F.T @ Y_oh)
        # Test accuracy
        F_t = compute_features_batch(f_test, rotors)
        pred = np.argmax(F_t @ W, axis=1)
        return np.mean(pred == y_test), W

    # Initialize with identity rotors
    theta = np.tile([1.0, 0.0, 0.0, 0.0], n_groups)
    optimizer = Adam(n_rotor_params, lr=args.lr)

    # Initial evaluation
    init_rotors = flat_to_rotors(theta)
    init_acc, init_W = evaluate_rotors(init_rotors)
    print(f"Initial accuracy (identity rotors): {init_acc:.4f}")
    print("=" * 60)

    os.makedirs(args.save_dir, exist_ok=True)
    best_test_acc = init_acc
    history = []
    half_pop = args.pop_size // 2

    for gen in range(args.generations):
        t0 = time.time()

        # Generate antithetic perturbations (only rotor params)
        epsilons = [np.random.randn(n_rotor_params) for _ in range(half_pop)]

        fitnesses_plus = []
        fitnesses_minus = []

        for eps in epsilons:
            # +ε
            rotors_p = flat_to_rotors(theta + args.sigma * eps)
            F_p = compute_features_batch(f_train, rotors_p)
            W_p = np.linalg.solve(
                F_p.T @ F_p + args.ridge_lambda * np.eye(n_features),
                F_p.T @ Y_oh)
            pred_p = np.argmax(F_p @ W_p, axis=1)
            fitnesses_plus.append(np.mean(pred_p == y_train))

            # -ε
            rotors_m = flat_to_rotors(theta - args.sigma * eps)
            F_m = compute_features_batch(f_train, rotors_m)
            W_m = np.linalg.solve(
                F_m.T @ F_m + args.ridge_lambda * np.eye(n_features),
                F_m.T @ Y_oh)
            pred_m = np.argmax(F_m @ W_m, axis=1)
            fitnesses_minus.append(np.mean(pred_m == y_train))

        all_fitnesses = fitnesses_plus + fitnesses_minus
        best_fit = max(all_fitnesses)
        avg_fit = sum(all_fitnesses) / len(all_fitnesses)

        # Rank-based fitness shaping
        ranks = compute_ranks(all_fitnesses)
        ranks_plus = ranks[:half_pop]
        ranks_minus = ranks[half_pop:]

        # ES gradient
        grad = np.zeros(n_rotor_params)
        for i, eps in enumerate(epsilons):
            grad += (ranks_plus[i] - ranks_minus[i]) * eps
        grad /= half_pop * args.sigma

        update = optimizer.step(grad)
        theta += update

        elapsed = time.time() - t0
        print(f"Gen {gen:4d} | train_best={best_fit:.4f} avg={avg_fit:.4f} "
              f"|grad|={np.linalg.norm(grad):.2f} | {elapsed:.1f}s")

        # Test evaluation
        if gen % args.eval_interval == 0 or gen == args.generations - 1:
            cur_rotors = flat_to_rotors(theta)
            test_acc, W_best = evaluate_rotors(cur_rotors)
            line = f"         TEST: {test_acc:.4f}"

            rot_acc = None
            if f_test_rotated is not None:
                F_rot = compute_features_batch(f_test_rotated, cur_rotors)
                pred_rot = np.argmax(F_rot @ W_best, axis=1)
                rot_acc = np.mean(pred_rot == y_test[:len(pred_rot)])
                line += f" | ROTATED: {rot_acc:.4f} | DROP: {test_acc - rot_acc:.4f}"

            print(line)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # Build and save controller
                ctrl = VertexHopfController(784, 16, 10)
                ctrl.rotors = cur_rotors
                ctrl.W_out = W_best.T.copy()
                ctrl.b_out = np.zeros(10)
                save_path = os.path.join(args.save_dir, "best_hopf_mnist.json")
                with open(save_path, "w") as f:
                    json.dump(ctrl.to_dict(), f)

            entry = {"gen": gen, "train_best": best_fit, "train_avg": avg_fit,
                     "test": test_acc}
            if rot_acc is not None:
                entry["test_rotated"] = rot_acc
            history.append(entry)

    # Final
    print("=" * 60)
    print("Training Complete")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Rotor params: {n_rotor_params}")
    print(f"Total params: {n_rotor_params + n_features * 10 + 10}")
    print("=" * 60)

    hist_path = os.path.join(args.save_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {hist_path}")


if __name__ == "__main__":
    main()
