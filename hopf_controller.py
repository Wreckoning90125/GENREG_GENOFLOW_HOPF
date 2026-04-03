# ================================================================
# Hopf Geometric Controller — Computational Substrate on S³
#
# Replaces the feed-forward MLP with operations derived from
# the Hopf fibration S³ → S² (fiber S¹).
#
# Legal operations:
#   1. Right multiplication (fiber-preserving rotation)
#   2. Left multiplication (base-changing rotation)
#   3. Hopf projection S³ → S² (the nonlinearity)
#   4. Section lifting S² → S³ (with learned phase)
#   5. Parallel transport (Berry phase accumulation)
#   6. Holonomy (topological phase from closed loops)
#
# All parameters live on S³. Mutation is geodesic.
# No gradients. No backprop. Evolution finds what works.
# ================================================================

import random
import math


# ================================================================
# Quaternion Arithmetic (pure Python, no dependencies)
# ================================================================

def quat_multiply(a, b):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def quat_conjugate(q):
    """Conjugate: [w, -x, -y, -z]."""
    return [q[0], -q[1], -q[2], -q[3]]


def quat_norm(q):
    """Euclidean norm of quaternion."""
    return math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)


def quat_normalize(q):
    """Project back to S³ (unit quaternion)."""
    n = quat_norm(q)
    if n < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]  # identity
    return [c / n for c in q]


def quat_exp(v):
    """
    Exponential map: tangent vector (pure quaternion [0, vx, vy, vz]) → S³.
    exp([0, v]) = [cos(|v|), sin(|v|)/|v| * v]
    """
    vx, vy, vz = v
    theta = math.sqrt(vx**2 + vy**2 + vz**2)
    if theta < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    s = math.sin(theta) / theta
    return [math.cos(theta), s * vx, s * vy, s * vz]


def quat_log(q):
    """
    Logarithmic map: S³ → tangent space at identity.
    Returns pure quaternion [vx, vy, vz].
    """
    q = quat_normalize(q)
    # Ensure w is in [-1, 1] for acos
    w = max(-1.0, min(1.0, q[0]))
    theta = math.acos(abs(w))
    if theta < 1e-12:
        return [0.0, 0.0, 0.0]
    s = theta / math.sin(theta)
    sign = 1.0 if w >= 0 else -1.0
    return [sign * s * q[1], sign * s * q[2], sign * s * q[3]]


def hopf_project(q):
    """
    Hopf projection S³ → S² (the nonlinearity).
    Maps unit quaternion to point on unit 2-sphere.

    π(q₀,q₁,q₂,q₃) = (q₀²+q₁²-q₂²-q₃², 2(q₁q₂+q₀q₃), 2(q₁q₃-q₀q₂))

    This IS the nonlinearity. No ReLU needed.
    """
    w, x, y, z = q
    return [
        w*w + x*x - y*y - z*z,
        2.0 * (x*y + w*z),
        2.0 * (x*z - w*y),
    ]


def hopf_lift(s2_point, phase):
    """
    Section lifting: S² × S¹ → S³.
    Given a point on S² and a phase angle, lift back to S³.

    Uses the standard section of the Hopf bundle.
    """
    sx, sy, sz = s2_point
    # Normalize the S² point
    norm = math.sqrt(sx*sx + sy*sy + sz*sz)
    if norm < 1e-12:
        return [math.cos(phase), math.sin(phase), 0.0, 0.0]
    sx, sy, sz = sx/norm, sy/norm, sz/norm

    # Standard section: given (sx, sy, sz) on S², construct unit quaternion
    # that projects to it, then rotate by phase along the fiber
    #
    # For north-pole-based section:
    # If sz > -1 + eps (not at south pole):
    #   base_q = [sqrt((1+sz)/2), -sy/sqrt(2(1+sz)), sx/sqrt(2(1+sz)), 0]
    # Then apply fiber rotation: q = base_q * [cos(phase/2), 0, 0, sin(phase/2)]

    if sz > -0.999:
        denom = math.sqrt(2.0 * (1.0 + sz))
        base = [denom / 2.0, -sy / denom, sx / denom, 0.0]
    else:
        # Near south pole, use alternate section
        denom = math.sqrt(2.0 * (1.0 - sz))
        base = [0.0, sx / denom, -sy / denom, denom / 2.0]

    # Fiber rotation by phase (right multiplication by e^{i*phase/2} in fiber direction)
    fiber_rot = [math.cos(phase / 2.0), 0.0, 0.0, math.sin(phase / 2.0)]
    result = quat_multiply(base, fiber_rot)
    return quat_normalize(result)


def luneburg_warp(vec):
    """
    Lüneburg/Poincaré conformal radial warp.

    Replaces per-component tanh with a geometry-preserving radial compression:
        r' = 2 * tanh(r / 2)

    The whole vector gets scaled together instead of squashed independently.
    Angles between components are preserved. The ball structure is maintained.
    Invertible via 2 * atanh(r' / 2).

    This IS the Poincaré disk metric expressed as physical optics (GRIN lens).
    Compression factor: sech²(r/2).
    """
    r = math.sqrt(sum(x * x for x in vec))
    if r < 1e-12:
        return vec[:]
    r_prime = 2.0 * math.tanh(r / 2.0)
    scale = r_prime / r
    return [x * scale for x in vec]


def luneburg_unwarp(vec):
    """Inverse: 2 * atanh(r' / 2). Maps from Poincaré ball back to Euclidean."""
    r_prime = math.sqrt(sum(x * x for x in vec))
    if r_prime < 1e-12:
        return vec[:]
    if r_prime >= 2.0 - 1e-12:
        r_prime = 2.0 - 1e-12  # clamp to boundary
    r = 2.0 * math.atanh(r_prime / 2.0)
    scale = r / r_prime
    return [x * scale for x in vec]


def compression_factor(r):
    """
    Compression factor sech²(r/2).
    Measures information density change with radius.
    """
    return 1.0 / (math.cosh(r / 2.0) ** 2)


def random_unit_quaternion():
    """Sample uniformly from S³ using Marsaglia's method."""
    while True:
        u1 = random.gauss(0, 1)
        u2 = random.gauss(0, 1)
        u3 = random.gauss(0, 1)
        u4 = random.gauss(0, 1)
        n = math.sqrt(u1*u1 + u2*u2 + u3*u3 + u4*u4)
        if n > 1e-12:
            return [u1/n, u2/n, u3/n, u4/n]


def geodesic_perturb(q, scale):
    """
    Geodesic perturbation on S³.
    Generate small tangent vector, exponentiate, right-multiply.
    Stays on S³ at all times.
    """
    v = [random.gauss(0, scale) for _ in range(3)]
    delta = quat_exp(v)
    result = quat_multiply(q, delta)
    return quat_normalize(result)


# ================================================================
# HopfController — The Geometric Computational Substrate
# ================================================================

class HopfController:
    """
    Controller that computes entirely on S³ via Hopf fibration geometry.

    Architecture:
        1. Embed input → S³ (chunk into quaternions, normalize)
        2. Left-multiply by learned quaternions (change fibers — inter-fiber mixing)
        3. Right-multiply by learned quaternions (rotate within fibers — phase)
        4. Hopf project S³ → S² (nonlinearity)
        5. Lift back to S³ with learned phases (for deeper layers)
        6. Final Hopf project → readout

    Drop-in replacement for Controller. Same interface.
    """

    def __init__(self, input_size=11, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.output_size = output_size

        # Number of quaternion units in hidden layer
        # Each quaternion is 4 reals, but we think in quaternion units
        self.hidden_size = hidden_size  # store for compatibility
        self.n_quat_hidden = max(1, hidden_size // 4)

        # Number of input quaternions: chunk input into groups of 4
        # Pad input to multiple of 4
        self.n_quat_input = max(1, (input_size + 3) // 4)

        # --- Learnable parameters (all on S³) ---

        # Layer 1: Input embedding to hidden
        # For each hidden quaternion, one left-multiplier per input quaternion
        # This mixes across fibers (cross-feature interaction)
        self.left_quats = [
            [random_unit_quaternion() for _ in range(self.n_quat_input)]
            for _ in range(self.n_quat_hidden)
        ]

        # Right multipliers for phase rotation (within-fiber)
        # One per hidden unit
        self.right_quats = [random_unit_quaternion() for _ in range(self.n_quat_hidden)]

        # Layer 2: Hidden to output via Hopf projection
        # After projecting hidden to S² (3 coords each), we need to
        # map n_quat_hidden * 3 values → output_size
        #
        # This is done geometrically: lift the S² points back to S³,
        # apply another round of left/right multiplications, project again.

        # Learned phases for section lifting (one per hidden unit)
        self.lift_phases = [random.uniform(-math.pi, math.pi) for _ in range(self.n_quat_hidden)]

        # Output layer: quaternion multipliers that combine hidden representations
        # For each output, we have one quaternion that left-multiplies each hidden unit
        # Then project to S² and take one coordinate as the logit
        n_quat_out = max(1, (output_size + 2) // 3)  # each S² point gives 3 coords
        self.out_left_quats = [
            [random_unit_quaternion() for _ in range(self.n_quat_hidden)]
            for _ in range(n_quat_out)
        ]
        self.out_right_quats = [random_unit_quaternion() for _ in range(n_quat_out)]
        self.n_quat_out = n_quat_out

    def _embed_to_quaternions(self, inputs):
        """
        Embed flat input vector into unit quaternions on S³.
        Chunk into groups of 4, normalize to unit quaternions.
        """
        # Pad to multiple of 4
        x = list(inputs)
        while len(x) < self.input_size:
            x.append(0.0)
        x = x[:self.input_size]
        while len(x) % 4 != 0:
            x.append(0.0)

        quats = []
        for i in range(0, len(x), 4):
            q = x[i:i+4]
            n = math.sqrt(sum(c*c for c in q))
            if n < 1e-12:
                quats.append([1.0, 0.0, 0.0, 0.0])
            else:
                quats.append([c / n for c in q])
        return quats

    def forward(self, inputs):
        """
        Forward pass through Hopf geometry.

        Args:
            inputs: list of floats (signal values)

        Returns:
            list of output_size logits
        """
        # 1. Embed input to S³
        input_quats = self._embed_to_quaternions(inputs)

        # 2. Hidden layer: left-multiply (cross-fiber mixing) + right-multiply (phase)
        #    Then apply Lüneburg/Poincaré conformal warp (replaces per-component tanh)
        hidden_quats = []
        for h in range(self.n_quat_hidden):
            # Aggregate input quaternions via sequential left multiplication
            # Each hidden unit combines all inputs through its learned rotations
            accum = [1.0, 0.0, 0.0, 0.0]  # identity
            for j in range(min(len(input_quats), len(self.left_quats[h]))):
                # Left-multiply input by learned quaternion (changes fiber)
                rotated = quat_multiply(self.left_quats[h][j], input_quats[j])
                # Accumulate via quaternion multiplication (composition of rotations)
                accum = quat_multiply(accum, rotated)

            # Right-multiply by phase quaternion (within-fiber rotation)
            accum = quat_multiply(accum, self.right_quats[h])

            # Lüneburg conformal warp on the imaginary part (preserves geometry)
            # r' = 2*tanh(r/2) applied radially — conformal, angle-preserving
            # This replaces per-component tanh with a geometry-respecting nonlinearity
            im_part = luneburg_warp([accum[1], accum[2], accum[3]])
            accum = [accum[0], im_part[0], im_part[1], im_part[2]]

            hidden_quats.append(quat_normalize(accum))

        # 3. Hopf projection: S³ → S² (THIS IS THE NONLINEARITY)
        hidden_s2 = [hopf_project(q) for q in hidden_quats]

        # 4. Section lifting: S² → S³ with learned phases
        lifted = []
        for i, s2_point in enumerate(hidden_s2):
            lifted.append(hopf_lift(s2_point, self.lift_phases[i]))

        # 5. Output layer: another round of geometric operations
        output_s2 = []
        for o in range(self.n_quat_out):
            accum = [1.0, 0.0, 0.0, 0.0]
            for j in range(min(len(lifted), len(self.out_left_quats[o]))):
                rotated = quat_multiply(self.out_left_quats[o][j], lifted[j])
                accum = quat_multiply(accum, rotated)
            accum = quat_multiply(accum, self.out_right_quats[o])
            accum = quat_normalize(accum)
            # Project to S²
            output_s2.append(hopf_project(accum))

        # 6. Read out logits from S² coordinates
        # Each S² point gives 3 coordinates in [-1, 1]
        # Flatten and take first output_size values
        flat = []
        for s2 in output_s2:
            flat.extend(s2)

        # Pad if needed, truncate to output_size
        while len(flat) < self.output_size:
            flat.append(0.0)
        return flat[:self.output_size]

    def select_action(self, signals, signal_order):
        """
        Select action from signal dictionary.
        Same interface as Controller.

        Args:
            signals: dict of signal_name → value
            signal_order: list of signal names in order

        Returns:
            int: action index
        """
        inputs = [signals.get(k, 0.0) for k in signal_order]
        outputs = self.forward(inputs)
        return outputs.index(max(outputs))

    def mutate(self, rate=0.1, scale=0.3):
        """
        Geodesic mutation on S³.
        Instead of additive Gaussian noise on flat weights,
        we perturb quaternion parameters along geodesics.

        Args:
            rate: probability of mutating each parameter
            scale: magnitude of geodesic perturbation
        """
        # Mutate left quaternions (layer 1)
        for h in range(self.n_quat_hidden):
            for j in range(len(self.left_quats[h])):
                if random.random() < rate:
                    self.left_quats[h][j] = geodesic_perturb(self.left_quats[h][j], scale)

        # Mutate right quaternions (layer 1)
        for h in range(self.n_quat_hidden):
            if random.random() < rate:
                self.right_quats[h] = geodesic_perturb(self.right_quats[h], scale)

        # Mutate lift phases
        for i in range(self.n_quat_hidden):
            if random.random() < rate:
                self.lift_phases[i] += random.gauss(0, scale)
                # Keep in [-π, π]
                self.lift_phases[i] = ((self.lift_phases[i] + math.pi) % (2 * math.pi)) - math.pi

        # Mutate output left quaternions
        for o in range(self.n_quat_out):
            for j in range(len(self.out_left_quats[o])):
                if random.random() < rate:
                    self.out_left_quats[o][j] = geodesic_perturb(self.out_left_quats[o][j], scale)

        # Mutate output right quaternions
        for o in range(self.n_quat_out):
            if random.random() < rate:
                self.out_right_quats[o] = geodesic_perturb(self.out_right_quats[o], scale)

    def clone(self):
        """Create a deep copy of this controller."""
        c = HopfController(self.input_size, self.hidden_size, self.output_size)
        c.n_quat_hidden = self.n_quat_hidden
        c.n_quat_input = self.n_quat_input
        c.n_quat_out = self.n_quat_out

        c.left_quats = [[q[:] for q in row] for row in self.left_quats]
        c.right_quats = [q[:] for q in self.right_quats]
        c.lift_phases = self.lift_phases[:]
        c.out_left_quats = [[q[:] for q in row] for row in self.out_left_quats]
        c.out_right_quats = [q[:] for q in self.out_right_quats]
        return c

    def param_count(self):
        """Count total scalar parameters."""
        count = 0
        # left_quats: n_quat_hidden * n_quat_input * 4
        count += self.n_quat_hidden * self.n_quat_input * 4
        # right_quats: n_quat_hidden * 4
        count += self.n_quat_hidden * 4
        # lift_phases: n_quat_hidden
        count += self.n_quat_hidden
        # out_left_quats: n_quat_out * n_quat_hidden * 4
        count += self.n_quat_out * self.n_quat_hidden * 4
        # out_right_quats: n_quat_out * 4
        count += self.n_quat_out * 4
        return count

    def effective_dof(self):
        """
        Count effective degrees of freedom.
        Each unit quaternion has 3 DOF (constrained to S³).
        Phases have 1 DOF each.
        """
        n_quats = (
            self.n_quat_hidden * self.n_quat_input +  # left_quats
            self.n_quat_hidden +                        # right_quats
            self.n_quat_out * self.n_quat_hidden +     # out_left_quats
            self.n_quat_out                             # out_right_quats
        )
        return n_quats * 3 + self.n_quat_hidden  # 3 per quaternion + 1 per phase

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "type": "hopf",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "n_quat_hidden": self.n_quat_hidden,
            "n_quat_input": self.n_quat_input,
            "n_quat_out": self.n_quat_out,
            "left_quats": self.left_quats,
            "right_quats": self.right_quats,
            "lift_phases": self.lift_phases,
            "out_left_quats": self.out_left_quats,
            "out_right_quats": self.out_right_quats,
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize from dictionary."""
        c = cls(d["input_size"], d["hidden_size"], d["output_size"])
        c.n_quat_hidden = d["n_quat_hidden"]
        c.n_quat_input = d["n_quat_input"]
        c.n_quat_out = d["n_quat_out"]

        if d.get("left_quats"):
            c.left_quats = d["left_quats"]
        if d.get("right_quats"):
            c.right_quats = d["right_quats"]
        if d.get("lift_phases"):
            c.lift_phases = d["lift_phases"]
        if d.get("out_left_quats"):
            c.out_left_quats = d["out_left_quats"]
        if d.get("out_right_quats"):
            c.out_right_quats = d["out_right_quats"]
        return c
