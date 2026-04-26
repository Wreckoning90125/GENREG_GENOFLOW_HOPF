#!/usr/bin/env python3
"""
GENREG × Hopf MNIST Training Script — v6

Evolves a Hopf geometric controller to classify MNIST digits.
Uses GENREG's trust-based evolutionary engine with the HopfController
computing entirely on S³ via Hopf fibration geometry.

v6 changes:
  - Per-eigenspace Hopf projections (50 features vs 36)
  - Crossover between parents (uniform, 30% rate)
  - Adaptive mutation scale (decays over training)
  - Half-scale readout mutation (preserves learned structure)
  - Direct pixel path (skip dict overhead)

Usage:
    python train_hopf_mnist.py [--generations 500] [--pop-size 50] [--batch-size 100]

Comparison targets:
    - GENREG_ALPHA_MNIST (MLP): 81.47% with 50,890 params
    - This (Hopf v6): ~582 params
"""

import argparse
import json
import math
import os
import random
import time

from hopf_controller import HopfController
from genreg_genome import Genome
from genreg_proteins import (
    SensorProtein, TrendProtein, TrustModifierProtein, run_protein_cascade
)
from nodes.envs.mnist import MNISTEnvironment


# ================================================================
# MNIST-specific protein network
# ================================================================

def mnist_proteins():
    """
    Minimal protein network for MNIST.
    Trust = accuracy reward. Simple and direct.
    """
    proteins = []

    # Sensor: track accuracy
    proteins.append(SensorProtein("accuracy"))
    proteins.append(SensorProtein("last_correct"))

    # Trust modifier: reward correct predictions directly
    trust_correct = TrustModifierProtein("trust_correct")
    trust_correct.bind_inputs(["last_correct"])
    trust_correct.params["gain"] = 1.0
    trust_correct.params["scale"] = 10.0
    trust_correct.params["decay"] = 0.0  # no decay — immediate reward
    proteins.append(trust_correct)

    return proteins


# ================================================================
# Evaluation
# ================================================================

def evaluate_genome(genome, env, signal_order):
    """
    Evaluate a single genome on MNIST.
    Returns accuracy (0.0-1.0).
    """
    genome.reset()
    genome.signal_order = signal_order
    signals = env.reset()

    while True:
        # Direct pixel path — skip dict overhead
        pixels = env.get_pixel_signals()
        logits = genome.controller.forward(pixels)
        action = logits.index(max(logits))

        # Step environment with classification
        signals, done = env.step(action)

        # Run proteins for trust update (uses metadata signals, not pixels)
        outputs, trust_delta = run_protein_cascade(genome.proteins, signals)
        genome.trust += trust_delta

        if done:
            break

    return env.get_accuracy()


def evaluate_population(genomes, env, signal_order):
    """Evaluate all genomes, return list of (genome_idx, accuracy)."""
    results = []
    for i, g in enumerate(genomes):
        acc = evaluate_genome(g, env, signal_order)
        results.append((i, acc))
    return results


# ================================================================
# Evolution with crossover + adaptive mutation
# ================================================================

def evolve(genomes, survival_rate=0.2, mutation_rate=0.15, mutation_scale=0.2,
           trust_inheritance=0.5, crossover_rate=0.3):
    """
    One generation of evolution with crossover.

    - Truncation selection (top survival_rate)
    - Elitism (best preserved unchanged)
    - Crossover: with probability crossover_rate, blend two parents
    - Mutation: Gaussian perturbation on all offspring
    """
    # Sort by trust (descending)
    genomes.sort(key=lambda g: g.trust, reverse=True)

    n_survivors = max(1, int(len(genomes) * survival_rate))
    survivors = genomes[:n_survivors]

    new_genomes = []

    # Elitism: keep best unchanged
    new_genomes.append(survivors[0].clone())

    # Fill with crossover/mutated offspring
    while len(new_genomes) < len(genomes):
        parent1 = random.choice(survivors)

        if random.random() < crossover_rate and n_survivors > 1:
            # Crossover: blend two parents
            parent2 = random.choice(survivors)
            while parent2 is parent1 and n_survivors > 1:
                parent2 = random.choice(survivors)
            child_ctrl = parent1.controller.crossover(parent2.controller)
            child = parent1.clone()
            child.controller = child_ctrl
        else:
            child = parent1.clone()

        child.mutate(mutation_rate, mutation_scale)
        child.trust = parent1.trust * trust_inheritance
        new_genomes.append(child)

    return new_genomes


# ================================================================
# Main Training Loop
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="GENREG × Hopf MNIST Training v6")
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100,
                        help="MNIST digits per episode")
    parser.add_argument("--hidden-size", type=int, default=16,
                        help="Quaternion hidden units (actual = hidden_size//4)")
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--mutation-scale", type=float, default=0.2)
    parser.add_argument("--survival-rate", type=float, default=0.2)
    parser.add_argument("--crossover-rate", type=float, default=0.3)
    parser.add_argument("--eval-interval", type=int, default=25,
                        help="Generations between test set evaluations")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="checkpoints/hopf_v6_spectral")
    parser.add_argument("--controller", type=str, default="hopf",
                        choices=["hopf", "mlp"],
                        help="Controller type (hopf or mlp for comparison)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N training images (faster testing)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-rotated", action="store_true",
                        help="Also test on randomly rotated digits (zero-shot)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive mutation scale decay")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 60)
    print("GENREG x Hopf MNIST Training v6")
    print("=" * 60)
    print(f"Controller:     {args.controller}")
    print(f"Population:     {args.pop_size}")
    print(f"Generations:    {args.generations}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Hidden size:    {args.hidden_size}")
    print(f"Mutation:       rate={args.mutation_rate}, scale={args.mutation_scale}")
    print(f"Crossover:      rate={args.crossover_rate}")
    print(f"Survival rate:  {args.survival_rate}")
    print(f"Adaptive scale: {'OFF' if args.no_adaptive else 'ON (2x→0.5x)'}")

    # Create environments
    train_env = MNISTEnvironment(split="train", batch_size=args.batch_size,
                                  subset_size=args.subset)
    test_env = MNISTEnvironment(split="test", batch_size=1000)
    if args.test_rotated:
        test_env_rotated = MNISTEnvironment(split="test", batch_size=1000, rotate=True)

    # Input/output sizes for MNIST
    input_size = 784  # 28x28 pixels
    output_size = 10  # 10 digit classes
    signal_order = [f"pixel_{i:03d}" for i in range(input_size)]

    # Create initial population
    genomes = []
    for _ in range(args.pop_size):
        g = Genome(
            proteins=mnist_proteins(),
            controller_type=args.controller,
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=output_size,
        )
        g.signal_order = signal_order
        genomes.append(g)

    # Report parameter count
    sample = genomes[0].controller
    if hasattr(sample, 'param_count'):
        print(f"Param count:    {sample.param_count()}")
        print(f"Effective DOF:  {sample.effective_dof()}")
        print(f"Features:       {sample.n_features}")
    else:
        pc = (sample.input_size * sample.hidden_size + sample.hidden_size +
              sample.hidden_size * sample.output_size + sample.output_size)
        print(f"Param count:    {pc}")
    print("=" * 60)

    os.makedirs(args.save_dir, exist_ok=True)
    best_test_acc = 0.0
    history = []

    base_scale = args.mutation_scale

    for gen in range(args.generations):
        t0 = time.time()

        # Adaptive mutation scale: broad early, fine late
        if args.no_adaptive:
            scale = base_scale
        else:
            progress = gen / max(args.generations - 1, 1)
            scale = base_scale * (2.0 - 1.5 * progress)

        # Evaluate population on training batch
        results = evaluate_population(genomes, train_env, signal_order)
        accs = [r[1] for r in results]

        best_acc = max(accs)
        avg_acc = sum(accs) / len(accs)
        best_idx = accs.index(best_acc)
        best_trust = genomes[best_idx].trust

        elapsed = time.time() - t0

        # Log
        print(f"Gen {gen:4d} | best={best_acc:.4f} avg={avg_acc:.4f} "
              f"trust={best_trust:.2f} scale={scale:.3f} | {elapsed:.1f}s")

        # Periodic test set evaluation
        if gen % args.eval_interval == 0 or gen == args.generations - 1:
            best_genome = genomes[best_idx]
            test_acc = evaluate_genome(best_genome, test_env, signal_order)
            line = f"         TEST: {test_acc:.4f}"

            if args.test_rotated:
                rot_acc = evaluate_genome(best_genome, test_env_rotated, signal_order)
                line += f" | ROTATED: {rot_acc:.4f} | DROP: {test_acc - rot_acc:.4f}"

            print(line)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # Save best
                save_path = os.path.join(args.save_dir, "best_hopf_mnist.json")
                with open(save_path, "w") as f:
                    json.dump(best_genome.to_dict(), f)

            entry = {"gen": gen, "train_best": best_acc, "train_avg": avg_acc,
                     "test": test_acc, "trust": best_trust,
                     "mutation_scale": scale}
            if args.test_rotated:
                entry["test_rotated"] = rot_acc
            history.append(entry)

        # Save periodic checkpoint
        if gen % args.save_interval == 0 and gen > 0:
            ckpt_path = os.path.join(args.save_dir, f"gen_{gen:05d}.json")
            with open(ckpt_path, "w") as f:
                json.dump({
                    "generation": gen,
                    "best_accuracy": best_acc,
                    "best_test_accuracy": best_test_acc,
                    "genome": genomes[best_idx].to_dict(),
                    "history": history,
                }, f)

        # Evolve
        genomes = evolve(
            genomes,
            survival_rate=args.survival_rate,
            mutation_rate=args.mutation_rate,
            mutation_scale=scale,
            crossover_rate=args.crossover_rate,
        )

    # Final summary
    print("=" * 60)
    print("Training Complete")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Controller: {args.controller}")
    if hasattr(genomes[0].controller, 'param_count'):
        print(f"Parameters: {genomes[0].controller.param_count()}")
    print("=" * 60)

    # Save history
    hist_path = os.path.join(args.save_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {hist_path}")


if __name__ == "__main__":
    main()
