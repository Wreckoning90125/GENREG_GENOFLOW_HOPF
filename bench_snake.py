"""
Snake A/B: HopfController vs vanilla MLP Controller, head-to-head under
the same GENREG evolutionary loop.

Both controller types are already integrated into Genome via
controller_type in {"mlp", "hopf"}. This script runs both populations
in parallel, same seed, same population size, same generations, same
mutation parameters, on the same Snake environment, and reports:

    - best food eaten (per generation + final)
    - best trust (per generation + final)
    - parameter count
    - convergence (generations to reach a trust threshold)
    - per-controller per-seed variance

Default config: 5 seeds x 50 generations x 50 population x 200
steps_per_life. Reduce via CLI flags if too slow.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genreg_genome import Genome
from genreg_population import Population
from nodes.envs.snake import SnakeEnvironment
from genreg_proteins import SensorProtein


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def count_params(controller):
    """Return the parameter count of a controller (mlp or hopf)."""
    if hasattr(controller, "to_flat"):
        # HopfController exposes to_flat
        return int(controller.to_flat().size)
    # MLP Controller: count weights and biases
    n = (
        controller.hidden_size * controller.input_size
        + controller.hidden_size
        + controller.output_size * controller.hidden_size
        + controller.output_size
    )
    return n


def make_population(size, controller_type, seed, signal_order):
    """Build a population of size N with the given controller type and a
    consistent set of default proteins. signal_order ensures every
    genome reads the same signal vector in the same order."""
    set_all_seeds(seed)
    pop = Population(size=size)
    for g in pop.genomes:
        # Replace controller with the correct type
        from genreg_controller import Controller
        from hopf_controller import HopfController
        if controller_type == "hopf":
            g.controller = HopfController(input_size=11, hidden_size=16,
                                           output_size=4)
        else:
            g.controller = Controller(input_size=11, hidden_size=16,
                                       output_size=4)
        g.controller_type = controller_type
        g.signal_order = signal_order
    return pop


def run_one_population(pop, env, n_generations, steps_per_life,
                       randomize_food=True, label=""):
    """Run n_generations of evolution; return per-generation curves."""
    history = {
        "best_trust": [],
        "avg_trust": [],
        "best_food": [],
        "avg_food": [],
    }
    for gen in range(n_generations):
        # New food layout each generation but same for all genomes in this gen
        if randomize_food:
            env.reset()
        pop.evaluate(env, steps_per_life=steps_per_life)
        stats = pop._update_stats() if hasattr(pop, "_update_stats") else None
        s = pop.get_stats()
        history["best_trust"].append(s["best_trust"])
        history["avg_trust"].append(s["avg_trust"])
        history["best_food"].append(s["best_food"])
        history["avg_food"].append(s["avg_food"])
        if (gen + 1) % max(1, n_generations // 10) == 0:
            print(f"  [{label}] gen {gen+1}/{n_generations}: "
                  f"best_food={s['best_food']:.0f} avg_food={s['avg_food']:.2f} "
                  f"best_trust={s['best_trust']:.1f} avg_trust={s['avg_trust']:.2f}")
        pop.evolve()
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-generations", type=int, default=50)
    ap.add_argument("--population-size", type=int, default=50)
    ap.add_argument("--steps-per-life", type=int, default=200)
    ap.add_argument("--out-dir", type=str, default="checkpoints/snake_ab")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 1 seed, 5 gens, 10 pop")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [1]
        args.n_generations = 5
        args.population_size = 10
        args.steps_per_life = 100

    os.makedirs(args.out_dir, exist_ok=True)

    # Standard signal order matching default proteins
    signal_order = [
        "steps_alive", "energy", "dist_to_food", "head_x", "head_y",
        "food_x", "food_y", "food_dx", "food_dy", "near_wall", "alive",
    ]

    # Param-count comparison (one-shot, same for all seeds)
    set_all_seeds(0)
    from genreg_controller import Controller
    from hopf_controller import HopfController
    mlp_params = count_params(Controller(11, 16, 4))
    hopf_params = count_params(HopfController(11, 16, 4))
    print(f"\nParameter counts: MLP={mlp_params}, Hopf={hopf_params}")

    results_by_seed = {"mlp": [], "hopf": []}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        for ctype in ["mlp", "hopf"]:
            print(f"\n--- {ctype.upper()} controller ---")
            t0 = time.time()
            env = SnakeEnvironment(grid_size=10)
            pop = make_population(args.population_size, ctype, seed, signal_order)
            history = run_one_population(
                pop, env, args.n_generations, args.steps_per_life, label=ctype
            )
            elapsed = time.time() - t0
            final = {
                "seed": seed,
                "controller": ctype,
                "final_best_trust": history["best_trust"][-1],
                "final_avg_trust": history["avg_trust"][-1],
                "final_best_food": history["best_food"][-1],
                "final_avg_food": history["avg_food"][-1],
                "max_best_food_ever": max(history["best_food"]),
                "max_best_trust_ever": max(history["best_trust"]),
                "elapsed_s": elapsed,
                "history": history,
            }
            results_by_seed[ctype].append(final)
            print(f"  Done in {elapsed:.1f}s. final best_food={final['final_best_food']:.0f} "
                  f"max_ever={final['max_best_food_ever']:.0f}")

    # === Aggregate across seeds ===
    print("\n" + "=" * 70)
    print("Aggregate results (mean +/- std across seeds)")
    print("=" * 70)
    print(f"{'metric':<22} {'MLP':>20} {'Hopf':>20}")
    print("-" * 70)
    for metric in ("final_best_trust", "final_avg_trust",
                   "final_best_food", "final_avg_food",
                   "max_best_food_ever", "max_best_trust_ever"):
        m_vals = [r[metric] for r in results_by_seed["mlp"]]
        h_vals = [r[metric] for r in results_by_seed["hopf"]]
        m_mean, m_std = np.mean(m_vals), np.std(m_vals)
        h_mean, h_std = np.mean(h_vals), np.std(h_vals)
        delta = h_mean - m_mean
        marker = ""
        if metric.endswith("food") or metric.endswith("trust"):
            if delta > m_std:
                marker = "  <-- Hopf better"
            elif delta < -m_std:
                marker = "  <-- MLP better"
        print(f"{metric:<22} "
              f"{m_mean:>10.2f} +/- {m_std:>5.2f}  "
              f"{h_mean:>10.2f} +/- {h_std:>5.2f}{marker}")

    out = {
        "config": {
            "seeds": args.seeds,
            "n_generations": args.n_generations,
            "population_size": args.population_size,
            "steps_per_life": args.steps_per_life,
        },
        "param_counts": {"mlp": mlp_params, "hopf": hopf_params},
        "results_by_seed": results_by_seed,
    }
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
