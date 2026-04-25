"""
End-to-end Layer 1 runner: build a generalized-Hopf seed, verify it two
ways (Cartesian grid + 600-cell de Rham operators), diagnose it via
field-line tracing and Berry-phase accumulation, write HDF5 + VTK + JSON.

Usage:
    python hopf_layer1_cli.py --omega1 3 --omega2 2 --resolution 128 \
        --bbox -2 2 -2 2 -2 2 --R 1.0 --out-dir ./seeds/hopf_3_2 \
        --n-fieldlines 4 --n-transits 40 [--skip-verify] [--skip-berry]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict

import numpy as np


def _git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_and_verify_grid(omega1, omega2, R, bbox, resolution):
    from hopf_grid import (
        build_grid,
        sample_seed_on_grid,
        grid_divergence_interior_max,
        curl_A_minus_B_max,
        grid_helicity,
        boundary_flux,
    )
    grid = build_grid(bbox, resolution)
    B, A = sample_seed_on_grid(grid, omega1, omega2, R)
    residuals = {
        "max_abs_divB": grid_divergence_interior_max(B, grid["dx"]),
        "max_abs_curlA_minus_B": curl_A_minus_B_max(A, B, grid["dx"]),
        "helicity_grid": grid_helicity(A, B, grid["dx"]),
        "boundary_flux": boundary_flux(B, grid["dx"]),
    }
    return grid, B, A, residuals


def run_600cell_witness(omega1, omega2, R):
    from hopf_600cell_witness import run_witness
    return run_witness(omega1, omega2, R=R)


def run_fieldline_diagnostics(
    omega1, omega2, R, n_fieldlines=4, max_length=60.0, max_steps=6000, seed=42
):
    from hopf_seed import seed_field
    from hopf_fieldlines import trace_fieldline, recover_iota
    from hopf_berry_diagnostic import (
        field_line_to_S2,
        accumulate_pancharatnam,
        accumulate_clifford_berry,
    )

    def Bfn(x, y, z):
        return seed_field(x, y, z, omega1, omega2, R)

    rng = np.random.default_rng(seed)
    runs = []
    for k in range(n_fieldlines):
        # Sample start points on nested tori: offset from core circle (R, 0, 0)
        # in the y = 0 plane by (dr, 0, dz) with small (dr, dz).
        dr = 0.10 + 0.05 * k
        dz = 0.08 + 0.03 * k
        x0 = np.array([R + dr, 0.0, dz])
        tr = trace_fieldline(Bfn, x0, max_length=max_length, max_steps=max_steps)
        iota = recover_iota(tr["path"], R_core=R)
        s2 = field_line_to_S2(tr["path"], R=R)
        phase_panch = accumulate_pancharatnam(s2)
        phase_cliff = accumulate_clifford_berry(s2)
        runs.append(
            {
                "x0": x0.tolist(),
                "traced_ok": bool(tr["ok"]),
                "toroidal_winding": iota["toroidal_winding"],
                "poloidal_winding": iota["poloidal_winding"],
                "iota_fit": iota["iota"],
                "iota_fit_residual": iota["residual"],
                "pancharatnam_phase": float(phase_panch),
                "clifford_berry_phase": float(phase_cliff),
                "berry_routes_agree": float(abs(phase_panch - phase_cliff)),
                "closed_loop_residual": float(
                    np.linalg.norm(tr["path"][-1] - tr["path"][0])
                ),
            }
        )
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omega1", type=int, required=True)
    p.add_argument("--omega2", type=int, required=True)
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument(
        "--bbox", type=float, nargs=6,
        default=[-2.0, 2.0, -2.0, 2.0, -2.0, 2.0],
        help="xmin xmax ymin ymax zmin zmax",
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--n-fieldlines", type=int, default=4)
    p.add_argument("--max-length", type=float, default=60.0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--skip-verify", action="store_true", help="skip 600-cell witness")
    p.add_argument("--skip-berry", action="store_true", help="skip field-line / Berry")
    p.add_argument("--no-hdf5", action="store_true")
    p.add_argument("--no-vtk", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from hopf_seed import analytic_iota, analytic_linking_number, analytic_helicity

    print(f"[hopf_layer1] omega1={args.omega1}, omega2={args.omega2}, R={args.R}")
    print(f"[hopf_layer1] bbox={tuple(args.bbox)}, resolution={args.resolution}")
    H_an = analytic_helicity(args.omega1, args.omega2, args.R)
    print(f"[hopf_layer1] iota={analytic_iota(args.omega1, args.omega2)}  "
          f"linking={analytic_linking_number(args.omega1, args.omega2)}  "
          f"H_analytic={H_an:.6f}")

    print("[hopf_layer1] building grid + sampling seed...")
    grid, B, A, grid_residuals = build_and_verify_grid(
        args.omega1, args.omega2, args.R, tuple(args.bbox), args.resolution
    )
    for k, v in grid_residuals.items():
        print(f"  {k}: {v:.4e}")

    witness = None
    if not args.skip_verify:
        print("[hopf_layer1] running 600-cell second-witness...")
        witness = run_600cell_witness(args.omega1, args.omega2, args.R)
        for k, v in witness.items():
            print(f"  {k}: {v}")

    fieldlines = None
    if not args.skip_berry:
        print(f"[hopf_layer1] tracing {args.n_fieldlines} field lines + Berry "
              "accumulation...")
        fieldlines = run_fieldline_diagnostics(
            args.omega1, args.omega2, args.R,
            n_fieldlines=args.n_fieldlines,
            max_length=args.max_length,
            max_steps=args.max_steps,
        )
        for r in fieldlines:
            print(
                f"  x0={r['x0']} iota_fit={r['iota_fit']:.4f} "
                f"berry_panch={r['pancharatnam_phase']:.4f} "
                f"agree={r['berry_routes_agree']:.2e}"
            )

    # Assemble metadata. Report grid helicity vs the closed-form value.
    H_grid = grid_residuals["helicity_grid"]
    helicity_relative_error = abs(H_grid - H_an) / abs(H_an) if H_an != 0 else 0.0
    print(f"[hopf_layer1] H_grid={H_grid:.6f}  H_analytic={H_an:.6f}  "
          f"rel_err={helicity_relative_error:.4e}")

    meta: Dict[str, Any] = {
        "omega1": int(args.omega1),
        "omega2": int(args.omega2),
        "R": float(args.R),
        "bbox": list(args.bbox),
        "resolution": int(args.resolution),
        "iota": float(analytic_iota(args.omega1, args.omega2)),
        "linking_number": int(analytic_linking_number(args.omega1, args.omega2)),
        "analytic_helicity": float(H_an),
        "helicity_relative_error": float(helicity_relative_error),
        "residuals": grid_residuals,
        "600cell_witness": witness,
        "fieldline_diagnostics": fieldlines,
        "commit_hash": _git_commit_hash(),
    }

    # Writes
    if not args.no_hdf5:
        from hopf_io import write_hdf5
        h5 = os.path.join(args.out_dir, "seed.h5")
        print(f"[hopf_layer1] writing HDF5 -> {h5}")
        write_hdf5(h5, grid, B, A, meta)

    if not args.no_vtk:
        from hopf_io import write_vtk
        vti = os.path.join(args.out_dir, "seed.vti")
        print(f"[hopf_layer1] writing VTK -> {vti}")
        write_vtk(vti, grid, B, A)

    from hopf_io import write_metadata_json
    mj = os.path.join(args.out_dir, "meta.json")
    print(f"[hopf_layer1] writing meta -> {mj}")
    write_metadata_json(mj, meta)

    print("[hopf_layer1] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
