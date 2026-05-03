"""
Bundle-level Berry-phase diagnostic for field-line topology.

Given a traced field line in R^3, map each point through:
    (x, y, z) -> inverse stereographic -> q on S^3 -> hopf_project -> p on S^2

This produces a curve on S^2; for a closed field line (periodic in R^3)
the S^2 curve closes too. Accumulate the Pancharatnam (parallel-transport)
phase along this curve. As a cross-check, fan-triangulate the closed S^2
curve and sum the Clifford-rotor Berry phase per triangle.

The two routes agree to <1e-8 on triangles (verified in hopf_controller).
A discrepancy along a field line therefore signals either a tracing
artifact (open loop, numerical drift) or a singular point of the Hopf
projection (q = -1 stereographic pole).
"""
from __future__ import annotations

import math
import numpy as np

from hopf_seed import stereographic_R3_to_S3
from hopf_controller import (
    hopf_project,
    pancharatnam_phase,
    triangle_berry_clifford,
)


def field_line_to_S2(path, R=1.0):
    """Map a (N, 3) R^3 path to a (N, 3) S^2 path via inv-stereo + Hopf.

    Points at the stereographic pole (q0,q1,q2,q3) with q3 = -1 are
    unreachable from our parameterization (R^3 |x|->inf maps to q3=+1 under
    our convention), so no singularity is expected for bounded paths.
    """
    N = path.shape[0]
    out = np.zeros((N, 3))
    for i in range(N):
        x, y, z = path[i]
        q0, q1, q2, q3 = stereographic_R3_to_S3(x, y, z, R)
        q = np.array([q0, q1, q2, q3])
        out[i] = hopf_project(q)
    return out


def accumulate_pancharatnam(s2_path):
    """Sum pancharatnam_phase(p_i, p_{i+1}) along the S^2 path.

    For a closed path, also include the wrap-around segment p_{N-1} -> p_0.
    Returns the total accumulated phase (radians, unwrapped).
    """
    total = 0.0
    for i in range(len(s2_path) - 1):
        p1 = s2_path[i]
        p2 = s2_path[i + 1]
        total += pancharatnam_phase(p1, p2)
    # Closure
    total += pancharatnam_phase(s2_path[-1], s2_path[0])
    return total


def accumulate_clifford_berry(s2_path, reference=None):
    """Fan-triangulate the closed S^2 loop and sum triangle Berry phases.

    reference: an S^2 point used as the apex of the fan. If None, picks
    an out-of-plane unit vector. Each triangle contributes
    triangle_berry_clifford(reference, p_i, p_{i+1}) / 2; the factor of
    2 is the spinor-double-cover normalization used in hopf_controller
    verify_berry_phase, so this route returns the same Berry-phase
    convention as accumulate_pancharatnam.
    """
    if reference is None:
        reference = np.array([0.0, 0.0, 1.0])
        mean = np.mean(s2_path, axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        if np.linalg.norm(np.cross(reference, mean)) < 1e-3:
            reference = np.array([1.0, 0.0, 0.0])

    total = 0.0
    for i in range(len(s2_path) - 1):
        p1 = s2_path[i]
        p2 = s2_path[i + 1]
        total += triangle_berry_clifford(reference, p1, p2) / 2.0
    total += triangle_berry_clifford(reference, s2_path[-1], s2_path[0]) / 2.0
    return total


def accumulate_along_fieldline(B_fn, x0, R=1.0, max_length=60.0, max_steps=6000):
    """End-to-end: trace from x0, push to S^2, accumulate both Berry routes.

    Returns dict with traced path, S^2 path, and both phase totals.
    """
    from hopf_fieldlines import trace_fieldline

    tr = trace_fieldline(B_fn, x0, max_length=max_length, max_steps=max_steps)
    s2 = field_line_to_S2(tr["path"], R=R)
    phase_panch = accumulate_pancharatnam(s2)
    phase_cliff = accumulate_clifford_berry(s2)
    return {
        "path_R3": tr["path"],
        "path_S2": s2,
        "pancharatnam_total": float(phase_panch),
        "clifford_berry_total": float(phase_cliff),
        "routes_agree": float(abs(phase_panch - phase_cliff)),
        "closed_loop_residual": float(np.linalg.norm(tr["path"][-1] - tr["path"][0])),
    }


def expected_phase_from_winding(
    toroidal_winding, poloidal_winding, convention="pancharatnam"
):
    """Heuristic expected-phase formula for calibration at (1, 1).

    The relation between the Pancharatnam accumulated phase on the S^2
    image and the (omega1, omega2) winding numbers is convention-dependent
    (which Hopf projection, which fiber orientation). We leave this as a
    diagnostic output only: record the observed phase and the winding
    numbers, and defer calibration to the test suite and CLI metadata.

    This function returns None to make the ambiguity explicit.
    """
    return None
