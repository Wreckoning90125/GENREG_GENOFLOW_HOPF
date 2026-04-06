#!/usr/bin/env python3
"""
GENREG × Hopf MNIST — Evolution Strategies Optimizer

Uses OpenAI-style Evolution Strategies instead of genetic algorithm.
ES treats ALL evaluations as gradient signal, not just top 20%.
Antithetic sampling + Adam optimizer + rank normalization.

Key difference from GA:
  - GA: evaluate 50, keep 10, mutate → wastes 80% of compute
  - ES: evaluate 50, use ALL to estimate gradient → 5x more signal

Usage:
    python train_hopf_es.py [--generations 500] [--pop-size 50]
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np

from hopf_controller import (HopfController, VertexHopfController,
                              hopf_project, poincare_warp_scalar,
                              _get_pixel_kernel)
from nodes.envs.mnist import MNISTEnvironment, load_mnist


# ================================================================
# Evaluation (direct pixel path, no Genome overhead)
# ================================================================

def evaluate_controller(controller, env):
    """Evaluate controller on one episode. Returns accuracy."""
    env.reset()
    while True:
        pixels = env.get_pixel_signals()
        logits = controller.forward(pixels)
        action = logits.index(max(logits))
        _, done = env.step(action)
        if done:
            break
    return env.get_accuracy()


# ================================================================
# Rank-based fitness shaping (OpenAI ES paper)
# ================================================================

def compute_ranks(fitnesses):
    """Convert fitnesses to rank-based utilities (centered, normalized)."""
    n = len(fitnesses)
    ranks = np.zeros(n)
    sorted_idx = np.argsort(fitnesses)
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank
    # Normalize to [-0.5, 0.5] centered
    ranks = ranks / (n - 1) - 0.5
    return ranks


# ================================================================
# Adam optimizer
# ================================================================

def warm_start_vertex_hopf(controller, n_samples=10000, lam=0.01):
    """
    Initialize readout weights via ridge regression on identity-rotor Hopf features.
    This gives ~82% test accuracy immediately, providing a warm start for ES.
    """
    images, labels, _, _ = load_mnist("train")
    kernel = _get_pixel_kernel(controller.input_size)

    # Compute Hopf features with identity rotors
    idx = random.sample(range(len(images)), min(n_samples, len(images)))
    n = len(idx)
    n_feat = controller.n_features
    F = np.empty((n, n_feat))

    for s, i in enumerate(idx):
        x = np.asarray(images[i])
        f = x @ kernel
        for g in range(controller.N_GROUPS):
            c4 = f[g*4:(g+1)*4]
            mag = np.linalg.norm(c4)
            base = g * 4
            if mag > 1e-10:
                q = c4 / mag
                p = hopf_project(q)
                scale = min(mag, 10.0)
                F[s, base] = p[0] * scale
                F[s, base+1] = p[1] * scale
                F[s, base+2] = p[2] * scale
            else:
                F[s, base:base+3] = 0.0
            F[s, base+3] = poincare_warp_scalar(mag)

    y = np.array([labels[i] for i in idx])
    Y_oh = np.zeros((n, 10))
    for i, yi in enumerate(y):
        Y_oh[i, yi] = 1.0

    # Ridge regression: W = (F^T F + λI)^{-1} F^T Y
    W = np.linalg.solve(F.T @ F + lam * np.eye(n_feat), F.T @ Y_oh)
    controller.W_out = W.T.copy()  # (10, n_feat)
    controller.b_out = np.zeros(10)

    # Set rotors to identity
    for i in range(controller.N_GROUPS):
        controller.rotors[i] = np.array([1.0, 0.0, 0.0, 0.0])

    # Report warm start accuracy
    pred = np.argmax(F @ W, axis=1)
    train_acc = np.mean(pred == y)
    print(f"Warm start: ridge on {n} samples, train acc = {train_acc:.4f}")
    return controller


class Adam:
    def __init__(self, n_params, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
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


# ================================================================
# Main ES Training Loop
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="GENREG × Hopf MNIST — ES")
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--pop-size", type=int, default=50,
                        help="Must be even (antithetic sampling)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="MNIST digits per evaluation")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Noise standard deviation")
    parser.add_argument("--lr", type=float, default=0.02,
                        help="Adam learning rate")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--arch", type=str, default="vertex",
                        choices=["spectral", "vertex"],
                        help="spectral=v6 (50 features), vertex=v7 (120 features)")
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--save-dir", type=str,
                        default="checkpoints/hopf_v6_es")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-rotated", action="store_true")
    parser.add_argument("--warm-start", action="store_true",
                        help="Initialize readout with ridge regression (v7 only)")
    parser.add_argument("--warm-samples", type=int, default=10000,
                        help="Training samples for warm start ridge regression")
    args = parser.parse_args()

    if args.pop_size % 2 != 0:
        args.pop_size += 1  # need even for antithetic

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print("=" * 60)
    print("GENREG x Hopf MNIST — Evolution Strategies")
    print("=" * 60)
    print(f"Architecture:   {args.arch}")
    print(f"Population:     {args.pop_size} (antithetic pairs)")
    print(f"Generations:    {args.generations}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Sigma:          {args.sigma}")
    print(f"Learning rate:  {args.lr}")

    # Environments
    train_env = MNISTEnvironment(split="train", batch_size=args.batch_size)
    test_env = MNISTEnvironment(split="test", batch_size=1000)
    test_env_rotated = None
    if args.test_rotated:
        test_env_rotated = MNISTEnvironment(split="test", batch_size=1000,
                                             rotate=True)

    # Initialize parent controller
    if args.arch == "vertex":
        parent = VertexHopfController(784, args.hidden_size, 10)
        if args.warm_start:
            warm_start_vertex_hopf(parent, n_samples=args.warm_samples)
    else:
        parent = HopfController(784, args.hidden_size, 10)
    n_params = len(parent.to_flat())

    print(f"Param count:    {parent.param_count()}")
    print(f"Effective DOF:  {parent.effective_dof()}")
    print(f"Features:       {parent.n_features}")
    print(f"Flat params:    {n_params}")
    print("=" * 60)

    # Adam optimizer
    optimizer = Adam(n_params, lr=args.lr)
    theta = parent.to_flat()

    # Template controller for evaluation (reused to avoid re-init)
    template = parent.clone()

    os.makedirs(args.save_dir, exist_ok=True)
    best_test_acc = 0.0
    history = []
    half_pop = args.pop_size // 2

    for gen in range(args.generations):
        t0 = time.time()

        # Generate antithetic perturbations
        epsilons = [np.random.randn(n_params) for _ in range(half_pop)]

        # Evaluate all perturbations (both +ε and -ε)
        fitnesses_plus = []
        fitnesses_minus = []

        for eps in epsilons:
            # Positive perturbation
            template.from_flat(theta + args.sigma * eps)
            acc_plus = evaluate_controller(template, train_env)
            fitnesses_plus.append(acc_plus)

            # Negative perturbation
            template.from_flat(theta - args.sigma * eps)
            acc_minus = evaluate_controller(template, train_env)
            fitnesses_minus.append(acc_minus)

        # All fitnesses for rank normalization
        all_fitnesses = fitnesses_plus + fitnesses_minus
        best_acc = max(all_fitnesses)
        avg_acc = sum(all_fitnesses) / len(all_fitnesses)

        # Rank-based fitness shaping
        ranks = compute_ranks(all_fitnesses)
        ranks_plus = ranks[:half_pop]
        ranks_minus = ranks[half_pop:]

        # ES gradient estimate using antithetic pairs
        grad = np.zeros(n_params)
        for i, eps in enumerate(epsilons):
            grad += (ranks_plus[i] - ranks_minus[i]) * eps
        grad /= half_pop * args.sigma

        # Adam update
        update = optimizer.step(grad)
        theta += update

        elapsed = time.time() - t0
        print(f"Gen {gen:4d} | best={best_acc:.4f} avg={avg_acc:.4f} "
              f"|grad|={np.linalg.norm(grad):.4f} | {elapsed:.1f}s")

        # Test evaluation
        if gen % args.eval_interval == 0 or gen == args.generations - 1:
            parent.from_flat(theta)
            test_acc = evaluate_controller(parent, test_env)
            line = f"         TEST: {test_acc:.4f}"

            rot_acc = None
            if test_env_rotated:
                rot_acc = evaluate_controller(parent, test_env_rotated)
                line += f" | ROTATED: {rot_acc:.4f} | DROP: {test_acc - rot_acc:.4f}"

            print(line)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_path = os.path.join(args.save_dir, "best_hopf_mnist.json")
                with open(save_path, "w") as f:
                    json.dump(parent.to_dict(), f)

            entry = {"gen": gen, "train_best": best_acc, "train_avg": avg_acc,
                     "test": test_acc}
            if rot_acc is not None:
                entry["test_rotated"] = rot_acc
            history.append(entry)

    # Final
    print("=" * 60)
    print("Training Complete")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Parameters: {parent.param_count()}")
    print("=" * 60)

    hist_path = os.path.join(args.save_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {hist_path}")


if __name__ == "__main__":
    main()
