"""
Field-line tracing and rotational-transform recovery for the Hopf seed.

For the generalized-Hopf field with winding ratio (omega1, omega2),
closed field lines wind omega2 times toroidally (around z-axis) while
winding omega1 times poloidally (around the core circle x^2 + y^2 = R^2
in the z=0 plane). The Poincaré section at y = 0, x > 0 therefore
returns to the starting poloidal angle after omega2 toroidal transits;
iota = omega1 / omega2 is recovered by linear fit of accumulated
poloidal angle vs toroidal angle.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def _field_normalized(B_fn):
    """Return a callable (t, x) -> B(x) / |B(x)| for solve_ivp."""
    def rhs(_t, xyz):
        Bx, By, Bz = B_fn(xyz[0], xyz[1], xyz[2])
        vec = np.array([float(Bx), float(By), float(Bz)])
        n = np.linalg.norm(vec)
        if n < 1e-14:
            return np.zeros(3)
        return vec / n
    return rhs


def trace_fieldline(B_fn, x0, max_length=200.0, max_steps=50000, rtol=1e-7, atol=1e-9):
    """Trace a field line by integrating dx/ds = B / |B|.

    x0: (3,) initial point.
    max_length: integrate from s=0 to s=max_length (arc length in the
        normalized field).
    Returns dict with 't' (arc length samples), 'path' (N, 3), and a
    solve_ivp result object.
    """
    rhs = _field_normalized(B_fn)
    sol = solve_ivp(
        rhs,
        t_span=(0.0, max_length),
        y0=np.asarray(x0, dtype=float),
        method="LSODA",
        rtol=rtol,
        atol=atol,
        max_step=max_length / 200.0,
        dense_output=False,
        t_eval=np.linspace(0.0, max_length, max_steps),
    )
    path = sol.y.T  # (N, 3)
    return {"s": sol.t, "path": path, "ok": sol.success, "message": sol.message}


def _toroidal_angle(path):
    """phi(t) = atan2(y, x), unwrapped along the trajectory."""
    phi = np.arctan2(path[:, 1], path[:, 0])
    return np.unwrap(phi)


def _poloidal_angle(path, R_core=1.0):
    """theta(t) = atan2(z, rho - R_core) in the meridional plane, unwrapped."""
    rho = np.sqrt(path[:, 0] ** 2 + path[:, 1] ** 2)
    theta = np.arctan2(path[:, 2], rho - R_core)
    return np.unwrap(theta)


def recover_iota(path, R_core=1.0):
    """Recover iota = d(theta) / d(phi) by linear fit over the trajectory.

    Returns a dict with fit slope, residual, and the unwrapped angles.
    """
    phi = _toroidal_angle(path)
    theta = _poloidal_angle(path, R_core=R_core)
    # Require enough toroidal winding for a clean fit
    span = abs(phi[-1] - phi[0])
    if span < 2.0 * np.pi:
        return {
            "iota": np.nan,
            "residual": np.nan,
            "toroidal_winding": span / (2.0 * np.pi),
            "poloidal_winding": (theta[-1] - theta[0]) / (2.0 * np.pi),
            "phi": phi,
            "theta": theta,
            "message": "insufficient toroidal winding (<2pi)",
        }
    # Linear fit theta = iota * phi + c
    A = np.vstack([phi, np.ones_like(phi)]).T
    (iota, c), res, rank, sv = np.linalg.lstsq(A, theta, rcond=None)
    pred = iota * phi + c
    rms = float(np.sqrt(np.mean((theta - pred) ** 2)))
    return {
        "iota": float(iota),
        "residual": rms,
        "toroidal_winding": float(abs(phi[-1] - phi[0]) / (2.0 * np.pi)),
        "poloidal_winding": float((theta[-1] - theta[0]) / (2.0 * np.pi)),
        "intercept": float(c),
        "phi": phi,
        "theta": theta,
    }


def poincare_section_y0(path, direction="down"):
    """Intersections of path with the half-plane y = 0, x > 0.

    direction: 'down' keeps crossings where y goes from >0 to <0
        (standard toroidal Poincaré convention); 'up' the reverse;
        'both' returns all.
    """
    y = path[:, 1]
    sign = np.sign(y)
    # crossings: sign change between consecutive samples
    keep = []
    for i in range(len(y) - 1):
        if sign[i] == 0 or sign[i + 1] == 0:
            continue
        if sign[i] == sign[i + 1]:
            continue
        if direction == "down" and not (sign[i] > 0 and sign[i + 1] < 0):
            continue
        if direction == "up" and not (sign[i] < 0 and sign[i + 1] > 0):
            continue
        # linear interpolation in y
        t = y[i] / (y[i] - y[i + 1])
        p = (1.0 - t) * path[i] + t * path[i + 1]
        if p[0] > 0:
            keep.append(p)
    return np.array(keep) if keep else np.zeros((0, 3))


def detect_axis(B_fn, R_init=1.0, max_iter=50, tol=1e-6):
    """Locate the core axis (O-point of the Poincaré section) by tracing
    and requiring that the poloidal motion has zero mean displacement.

    For the Hopf field this is exactly the unit circle x^2 + y^2 = R^2,
    z = 0. The method below is a cheap validation, not a general
    axis-finder.
    """
    # Start from the expected Hopf core, perturb slightly, check that
    # the poloidal oscillation amplitude is small.
    x0 = np.array([R_init, 0.0, 0.0])
    result = trace_fieldline(B_fn, x0, max_length=30.0, max_steps=3000)
    rho = np.sqrt(result["path"][:, 0] ** 2 + result["path"][:, 1] ** 2)
    z = result["path"][:, 2]
    drift = float(np.max(np.abs(rho - R_init)) + np.max(np.abs(z)))
    return {"R_axis": R_init, "drift": drift, "converged": drift < 1e-2}
