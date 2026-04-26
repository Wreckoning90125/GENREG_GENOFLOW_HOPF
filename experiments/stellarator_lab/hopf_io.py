"""
I/O for Layer 1 Hopf-seed artifacts.

write_hdf5:    /B, /A, /grid/{x,y,z}, attrs on /, /B, /A. Consumed by
               GLEMuR-class MHD codes.
write_vtk:     pyvista.ImageData -> .vti; Paraview-consumable.
write_metadata_json: all numerical residuals + analytic invariants in a
               small sidecar JSON for machine inspection.
"""
from __future__ import annotations

import json
import os

import numpy as np


def write_hdf5(path, grid, B, A, meta):
    """Write seed field + vector potential to HDF5.

    Layout:
        /grid/x   (N,)   grid["x"]
        /grid/y   (N,)
        /grid/z   (N,)
        /B        (3, N, N, N)
        /A        (3, N, N, N)
        attrs on /: omega1, omega2, R, bbox, resolution, iota,
                    linking_number, reference_helicity, commit_hash,
                    residuals_json
    """
    import h5py

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("grid")
        g.create_dataset("x", data=grid["x"])
        g.create_dataset("y", data=grid["y"])
        g.create_dataset("z", data=grid["z"])
        f.create_dataset("B", data=B, compression="gzip", compression_opts=4)
        f.create_dataset("A", data=A, compression="gzip", compression_opts=4)

        # scalar / short-vector attrs
        for key in (
            "omega1",
            "omega2",
            "R",
            "resolution",
            "iota",
            "linking_number",
            "reference_helicity",
        ):
            if key in meta:
                f.attrs[key] = meta[key]
        if "bbox" in meta:
            f.attrs["bbox"] = np.asarray(meta["bbox"])
        if "commit_hash" in meta:
            f.attrs["commit_hash"] = str(meta["commit_hash"])
        # Full meta dict as JSON attr for completeness
        f.attrs["meta_json"] = json.dumps(_jsonable(meta))


def read_hdf5(path):
    """Inverse of write_hdf5. Returns (grid, B, A, meta)."""
    import h5py

    with h5py.File(path, "r") as f:
        grid = {
            "x": f["grid/x"][:],
            "y": f["grid/y"][:],
            "z": f["grid/z"][:],
        }
        B = f["B"][:]
        A = f["A"][:]
        meta = json.loads(f.attrs["meta_json"])
    grid["dx"] = float(grid["x"][1] - grid["x"][0])
    grid["shape"] = B.shape[1:]
    grid["bbox"] = (
        float(grid["x"][0]), float(grid["x"][-1]),
        float(grid["y"][0]), float(grid["y"][-1]),
        float(grid["z"][0]), float(grid["z"][-1]),
    )
    return grid, B, A, meta


def write_vtk(path, grid, B, A):
    """Write seed field as a structured .vti for Paraview preview.

    ImageData convention: pyvista uses Fortran-ordered flat arrays. Our
    B shape is (3, N, N, N) with (i, j, k) matching (x, y, z) axes; we
    reshape to (N*N*N, 3) in Fortran order so Paraview sees the right
    geometry.
    """
    import pyvista as pv

    Nx, Ny, Nz = B.shape[1], B.shape[2], B.shape[3]
    origin = (float(grid["x"][0]), float(grid["y"][0]), float(grid["z"][0]))
    spacing = (float(grid["dx"]), float(grid["dx"]), float(grid["dx"]))
    img = pv.ImageData(dimensions=(Nx, Ny, Nz), spacing=spacing, origin=origin)

    # Flatten in Fortran order to match ImageData indexing
    B_flat = np.stack(
        [B[0].ravel(order="F"), B[1].ravel(order="F"), B[2].ravel(order="F")],
        axis=1,
    )
    A_flat = np.stack(
        [A[0].ravel(order="F"), A[1].ravel(order="F"), A[2].ravel(order="F")],
        axis=1,
    )
    img.point_data["B"] = B_flat
    img.point_data["A"] = A_flat
    img.point_data["Bmag"] = np.linalg.norm(B_flat, axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    img.save(path)


def _jsonable(obj):
    """Recursively convert numpy scalars / arrays to Python primitives."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_metadata_json(path, meta):
    """Sidecar JSON dump of the meta dict."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(meta), f, indent=2, sort_keys=True)
