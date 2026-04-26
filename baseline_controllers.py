"""
Trivial baseline controllers for the Snake A/B benchmark.

These are NOT learned -- they are fixed-policy reference points that
ground the absolute scale of the food / trust metrics.

  RandomController -- uniform random over the 4 actions. Provides the
      floor for "no signal at all".

  GreedyController  -- picks the action that most reduces dist_to_food
      this step (with random tiebreak). Provides a strong fixed-policy
      ceiling: it knows where food is and goes there directly. Doesn't
      handle walls or tail intelligently; will die quickly if blocked.

Both implement the same interface as Controller / HopfController /
SnakeHopfController so they slot into the Genome / Population /
bench_snake harness without changes.
"""
from __future__ import annotations

import random


class RandomController:
    """Uniform-random over output_size actions. No learning."""

    def __init__(self, output_size=4):
        self.output_size = output_size

    def select_action(self, signals, signal_order=None):
        return random.randint(0, self.output_size - 1)

    def mutate(self, rate=0.1, scale=0.3):
        pass

    def crossover(self, other):
        return self.clone()

    def clone(self):
        return RandomController(self.output_size)

    def to_dict(self):
        return {"type": "random", "output_size": self.output_size}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("output_size", 4))

    def n_params(self):
        return 0


class GreedyController:
    """Greedy heuristic: take the action that maximally reduces
    Manhattan distance to food.

    Reads food_dx, food_dy from signals and picks the action whose
    direction best matches the food direction. Random tiebreak.
    No learning, no parameters.

    Action convention (matches Snake env):
        0 = up    (-y)
        1 = down  (+y)
        2 = left  (-x)
        3 = right (+x)
    """

    def __init__(self, output_size=4):
        self.output_size = output_size

    def select_action(self, signals, signal_order=None):
        dx = float(signals.get("food_dx", 0.0))
        dy = float(signals.get("food_dy", 0.0))
        # Score each action by alignment with food direction
        scores = [
            -dy,  # up: -y axis
            +dy,  # down: +y axis
            -dx,  # left: -x axis
            +dx,  # right: +x axis
        ]
        max_s = max(scores)
        # Random tiebreak among equal-score actions
        candidates = [i for i, s in enumerate(scores) if s == max_s]
        return random.choice(candidates)

    def mutate(self, rate=0.1, scale=0.3):
        pass

    def crossover(self, other):
        return self.clone()

    def clone(self):
        return GreedyController(self.output_size)

    def to_dict(self):
        return {"type": "greedy", "output_size": self.output_size}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("output_size", 4))

    def n_params(self):
        return 0
