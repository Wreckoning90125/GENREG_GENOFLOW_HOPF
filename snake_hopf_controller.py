"""
Snake-aware Hopf controller. Uses the geometric machinery as the
LEARNING substrate, not as decoration:

  1. Multi-channel directional embedding maps Snake's signals onto
     the 600-cell respecting their actual spatial structure (food
     direction is a vector on R^2 lifted to S^3, not pixel intensity
     on a 4x3 grid).

  2. Hopf-decagon action selector: the 600-cell's 12 great-circle
     decagons partition the 120 vertices into 12 fibers under a
     C_10 left action. Each fiber's centroid projects to a point on
     S^2 via the Hopf map, classifying the fiber by its cardinal
     direction (N/S/E/W). Fiber strengths (sum of activations within
     each fiber) become the action logits via a fixed 12 -> 4 group
     quotient.

  3. Combined readout: a learned linear readout on top of the ADE
     features ADDS to the geometric decagon action logits. The
     learnable mix scalar `alpha` controls the balance. The geometric
     prior is "free" (zero learned params); the readout adds learned
     refinement.

This is the geometric machinery doing the work: the input embedding,
the action geometry, and the eigenspace decomposition are all
structural; only the rotor angles, McKay couplings, readout weights,
and the alpha mix are learned via the GENREG evolutionary loop.
"""
from __future__ import annotations

import math

import numpy as np

from cell600 import get_geometry
from hopf_controller import HopfController, _get_geo, hopf_project
from hopf_decagon import hopf_decagon_partition, fiber_label


# Cached geometric structures
_DECAGON_CACHE = {}


def _decagon_action_map(grid_size=10):
    """Build the fixed 12-fiber -> 4-action map.

    For each Hopf decagon fiber (12 total), compute the average S^2
    direction by Hopf-projecting the centroid of the fiber's vertices
    on S^3 to S^2. Classify by largest |x| or |y| component into one
    of {0=up (-y), 1=down (+y), 2=left (-x), 3=right (+x)} matching
    Snake's action conventions.

    Returns:
        fiber_to_action: (12,) array of action indices in {0,1,2,3}
        fiber_directions: (12, 3) array of S^2 direction representatives
    """
    if "fiber_to_action" in _DECAGON_CACHE:
        return _DECAGON_CACHE["fiber_to_action"], _DECAGON_CACHE["fiber_dirs"]

    geo = _get_geo()
    vertices = geo["vertices"]  # (120, 4) on S^3
    orbits, _, _ = hopf_decagon_partition()  # (12, 10) vertex indices

    fiber_dirs = np.zeros((12, 3))
    for f_idx in range(12):
        # Average S^2 direction: Hopf-project each vertex, average,
        # renormalize. The fiber spans a great circle so the centroid
        # of the projected points may not be zero (and its direction
        # is the fiber's "axis" in S^2).
        s2_pts = []
        for v_idx in orbits[f_idx]:
            p = hopf_project(vertices[v_idx])
            s2_pts.append(p)
        s2_pts = np.array(s2_pts)
        # Use the spectral axis: the unit eigenvector of the outer-
        # product covariance matrix associated with the largest
        # eigenvalue. For a great circle this points along the fiber's
        # rotation axis.
        cov = s2_pts.T @ s2_pts
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, -1]  # largest eigenvalue's eigenvector
        if np.linalg.norm(axis) > 1e-10:
            axis = axis / np.linalg.norm(axis)
        fiber_dirs[f_idx] = axis

    # Classify each fiber by its xy-plane angle, divided into 4 cardinal
    # quadrants. Snake actions: 0=up (-y), 1=down (+y), 2=left (-x),
    # 3=right (+x). We use signed (xy) coords plus z as a continuous
    # angle: theta = atan2(y, x), so right=0, up=-pi/2, left=+/-pi,
    # down=+pi/2. Bin into 4 cardinal-direction quadrants centered on
    # 0, +pi/2, +/-pi, -pi/2.
    fiber_to_action = np.zeros(12, dtype=int)
    for f_idx in range(12):
        ax, ay, az = fiber_dirs[f_idx]
        # Use the signed direction in xy; if both ax and ay are tiny
        # the axis is essentially along z and we use z's sign as
        # tiebreaker (rare for this 600-cell geometry).
        nxy = math.sqrt(ax * ax + ay * ay)
        if nxy < 1e-6:
            fiber_to_action[f_idx] = 0 if az < 0 else 1
            continue
        theta = math.atan2(ay, ax)
        # Quadrant boundaries: pi/4 to 3pi/4 -> down (+y); -3pi/4 to
        # -pi/4 -> up (-y); -pi/4 to pi/4 -> right (+x); else -> left.
        if -math.pi / 4 <= theta < math.pi / 4:
            fiber_to_action[f_idx] = 3  # right
        elif math.pi / 4 <= theta < 3 * math.pi / 4:
            fiber_to_action[f_idx] = 1  # down
        elif -3 * math.pi / 4 <= theta < -math.pi / 4:
            fiber_to_action[f_idx] = 0  # up
        else:
            fiber_to_action[f_idx] = 2  # left

    _DECAGON_CACHE["fiber_to_action"] = fiber_to_action
    _DECAGON_CACHE["fiber_dirs"] = fiber_dirs
    _DECAGON_CACHE["orbits"] = orbits
    return fiber_to_action, fiber_dirs


def _vertex_cardinal_affinities():
    """Return a (120, 4) matrix of vertex affinities for the 4 cardinal
    actions, computed geometrically via the Hopf projection.

    Each 600-cell vertex projects to a point on S^2; we compute its
    cosine similarity to each cardinal target direction:
        UP    = (0, -1, 0)    DOWN  = (0, +1, 0)
        LEFT  = (-1, 0, 0)    RIGHT = (+1, 0, 0)

    This gives a fixed (no learned params) geometric mapping from
    vertex activation to action logits via:
        action_logits = activation @ affinities

    The mapping respects the C_4 symmetry of the cardinal actions
    exactly (rotating S^2 by pi/2 in xy permutes the cardinal columns
    accordingly).
    """
    if "vertex_affinities" in _DECAGON_CACHE:
        return _DECAGON_CACHE["vertex_affinities"]
    vertices = _get_geo()["vertices"]
    cardinals = np.array([
        [0.0, -1.0, 0.0],   # UP    (action 0)
        [0.0, +1.0, 0.0],   # DOWN  (action 1)
        [-1.0, 0.0, 0.0],   # LEFT  (action 2)
        [+1.0, 0.0, 0.0],   # RIGHT (action 3)
    ])
    s2_pts = np.array([hopf_project(v) for v in vertices])  # (120, 3)
    affinities = s2_pts @ cardinals.T  # (120, 4)
    _DECAGON_CACHE["vertex_affinities"] = affinities
    return affinities


def _geometric_action_logits(activation, n_actions=4):
    """Pure-geometry action logits (no learned params).

    Two contributions, summed:
      1. Vertex-cardinal affinity: each vertex's activation contributes
         to each cardinal action proportional to that vertex's Hopf-
         projected cosine similarity with the action's S^2 target.
      2. Hopf-decagon fiber strength: each of the 12 fibers' summed
         activation contributes to its assigned cardinal action class
         (assigned by xy-plane angle of the fiber axis).

    Both are 4-fold-action-symmetric (the geometric prior treats the 4
    cardinal actions on equal footing).
    """
    # Contribution 1: vertex-cardinal affinity
    affinities = _vertex_cardinal_affinities()
    affinity_logits = activation @ affinities  # (4,)

    # Contribution 2: decagon fiber strength
    if "orbits" not in _DECAGON_CACHE:
        _decagon_action_map()
    orbits = _DECAGON_CACHE["orbits"]
    fiber_to_action, _ = _decagon_action_map()
    fiber_strengths = np.zeros(12)
    for f_idx in range(12):
        for v_idx in orbits[f_idx]:
            fiber_strengths[f_idx] += activation[v_idx]
    fiber_logits = np.zeros(n_actions)
    for f_idx in range(12):
        fiber_logits[fiber_to_action[f_idx]] += fiber_strengths[f_idx]

    return affinity_logits + 0.3 * fiber_logits


def _hopf_section_xyz(ux, uy, uz):
    """Standard Hopf section S^2 -> S^3 for an S^2 point (ux, uy, uz)."""
    w = 1.0 + uz
    n = math.sqrt(w * w + ux * ux + uy * uy)
    if n < 1e-10:
        return np.array([0.0, 0.0, 1.0, 0.0])
    return np.array([w / n, 0.0, ux / n, uy / n])


def _vmf_assign(q, vertices, kappa):
    """Signed-vMF soft assignment of unit quaternion q to 120 vertices.
    Signed (no abs) preserves the spinor double cover, encoding orientation."""
    dots = vertices @ q
    scaled = kappa * dots - (kappa * dots).max()
    e = np.exp(scaled)
    return e / e.sum()


def snake_signals_to_activation(signals, kappa=4.0, grid_size=10):
    """Multi-channel directional embedding of Snake signals into a single
    120-vertex activation on the 600-cell.

    Channels (all summed into a single (120,) activation):
        food_dir : food direction vector lifted to S^2 (z=0) -> S^3
        head_pos : head grid coords stereographic-projected to S^2 -> S^3
        food_pos : same for food coords
        wall     : 4 cardinal direction bumps when near_wall, panic
                   weighted by 1/energy
        status   : uniform scalar background weighted by alive * energy
    """
    vertices = _get_geo()["vertices"]
    gs = grid_size
    f = np.zeros(120)

    # Food direction (unit vector in plane)
    dx = float(signals.get("food_dx", 0.0))
    dy = float(signals.get("food_dy", 0.0))
    norm = math.sqrt(dx * dx + dy * dy)
    if norm > 1e-6:
        ux, uy = dx / norm, dy / norm
        q = _hopf_section_xyz(ux, uy, 0.0)
        dist = max(float(signals.get("dist_to_food", 1.0)), 1.0)
        f += _vmf_assign(q, vertices, kappa) * (1.0 / dist)

    # Head position (stereographic to S^2)
    hx = float(signals.get("head_x", 0.0)) / max(gs - 1, 1) * 2.0 - 1.0
    hy = float(signals.get("head_y", 0.0)) / max(gs - 1, 1) * 2.0 - 1.0
    r2 = hx * hx + hy * hy
    denom = 1.0 + r2
    X, Y, Z = 2 * hx / denom, 2 * hy / denom, (r2 - 1.0) / denom
    f += _vmf_assign(_hopf_section_xyz(X, Y, Z), vertices, kappa) * 0.5

    # Food position
    fx = float(signals.get("food_x", 0.0)) / max(gs - 1, 1) * 2.0 - 1.0
    fy = float(signals.get("food_y", 0.0)) / max(gs - 1, 1) * 2.0 - 1.0
    r2f = fx * fx + fy * fy
    denomf = 1.0 + r2f
    Xf, Yf, Zf = 2 * fx / denomf, 2 * fy / denomf, (r2f - 1.0) / denomf
    f += _vmf_assign(_hopf_section_xyz(Xf, Yf, Zf), vertices, kappa) * 0.5

    # Wall danger: 4 cardinal bumps, panic-weighted
    if float(signals.get("near_wall", 0.0)) > 0.5:
        energy = float(signals.get("energy", 25.0))
        panic = 1.0 / max(1.0, energy)
        for ax_x, ax_y in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]:
            f += _vmf_assign(_hopf_section_xyz(ax_x, ax_y, 0.0),
                             vertices, kappa) * panic

    # Status background
    energy_frac = float(signals.get("energy", 25.0)) / 25.0
    alive = float(signals.get("alive", 1.0))
    f += np.full(120, (energy_frac * 0.5 + alive * 0.5) / 120.0)

    return f


class SnakeHopfController:
    """Snake controller using the geometric machinery as the substrate.

    Composition:
      - directional embedding (signals -> 120-vertex activation)
      - HopfController features (rotors, McKay, readout)
      - Hopf-decagon action prior (12 fibers -> 4 cardinal actions)
      - learned alpha mix between geometric prior and learned readout

    Total learned parameters: HopfController params (~210 with hidden=4)
    + alpha (1) ~= 211. Comparable to MLP at hidden=16 (260 params).
    """

    def __init__(self, output_size=4, hidden_size=16, kappa=4.0, grid_size=10,
                  alpha_init=1.0):
        self.output_size = output_size
        self.kappa = kappa
        self.grid_size = grid_size
        # Single underlying HopfController for the eigenspace machinery
        # + readout. Inputs are 120 (we feed activation directly).
        self.hopf = HopfController(input_size=120, hidden_size=hidden_size,
                                    output_size=output_size)
        # Learnable mix weight: alpha * geometric_prior + readout_logits
        self.alpha = float(alpha_init)

    def forward(self, signals):
        f = snake_signals_to_activation(signals, kappa=self.kappa,
                                         grid_size=self.grid_size)
        # Geometric action prior from Hopf machinery (vertex-cardinal
        # affinity + decagon fiber strength)
        prior = _geometric_action_logits(f, n_actions=self.output_size)
        # Learned readout on geometric features
        features = self.hopf.features_from_activation(f)
        readout = self.hopf.W_out @ features + self.hopf.b_out
        return self.alpha * prior + readout

    def select_action(self, signals, signal_order=None):
        return int(np.argmax(self.forward(signals)))

    def mutate(self, rate=0.1, scale=0.3):
        self.hopf.mutate(rate, scale)
        # Mutate alpha mix
        import random as _rnd
        if _rnd.random() < rate:
            self.alpha += _rnd.gauss(0.0, scale * 0.5)

    def crossover(self, other):
        child = self.clone()
        if hasattr(self.hopf, "crossover"):
            child.hopf = self.hopf.crossover(other.hopf)
        return child

    def clone(self):
        c = SnakeHopfController.__new__(SnakeHopfController)
        c.output_size = self.output_size
        c.kappa = self.kappa
        c.grid_size = self.grid_size
        c.hopf = self.hopf.clone()
        c.alpha = float(self.alpha)
        return c

    def to_dict(self):
        return {
            "type": "snake_hopf",
            "output_size": self.output_size,
            "kappa": self.kappa,
            "grid_size": self.grid_size,
            "alpha": self.alpha,
            "hopf": self.hopf.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        c = cls(output_size=d["output_size"], kappa=d.get("kappa", 4.0),
                grid_size=d.get("grid_size", 10),
                alpha_init=d.get("alpha", 1.0))
        c.hopf = HopfController.from_dict(d["hopf"])
        return c

    def to_flat(self):
        return np.concatenate([self.hopf.to_flat(), np.array([self.alpha])])

    def n_params(self):
        return int(self.to_flat().size)
