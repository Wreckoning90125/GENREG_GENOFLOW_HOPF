#!/usr/bin/env python3
"""
QM9 dataset loader for the 600-cell geometric feature extractor.

Downloads, parses, and splits QM9 (134k small molecules, up to 29
atoms each, 19 quantum mechanical properties). Standard SchNet split:
110k train / 10k val / ~10.8k test.

If download fails (e.g., no network access), provides a synthetic
molecule generator for pipeline validation.
"""

import os
import math
import tarfile
import numpy as np

# ================================================================
# Constants
# ================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qm9")

ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
ATOM_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
N_ATOM_CHANNELS = len(ATOM_TYPES)

HARTREE_TO_MEV = 27211.386
HARTREE_TO_EV = 27.211386
BOHR_TO_ANGSTROM = 0.529177

# QM9 property indices (0-based in the parsed property array)
# Properties as listed in the dsgdb9nsd.xyz header line (after gdb tag and index)
PROPERTY_NAMES = [
    "tag",       # 0 - gdb tag (string, not used)
    "index",     # 1 - molecule index
    "A",         # 2 - rotational constant A (GHz)
    "B",         # 3 - rotational constant B (GHz)
    "C",         # 4 - rotational constant C (GHz)
    "mu",        # 5 - dipole moment (Debye)
    "alpha",     # 6 - isotropic polarizability (Bohr^3)
    "homo",      # 7 - HOMO energy (Hartree)
    "lumo",      # 8 - LUMO energy (Hartree)
    "gap",       # 9 - HOMO-LUMO gap (Hartree)
    "R2",        # 10 - electronic spatial extent (Bohr^2)
    "zpve",      # 11 - zero-point vibrational energy (Hartree)
    "U0",        # 12 - internal energy at 0K (Hartree)
    "U",         # 13 - internal energy at 298.15K (Hartree)
    "H",         # 14 - enthalpy at 298.15K (Hartree)
    "G",         # 15 - free energy at 298.15K (Hartree)
    "Cv",        # 16 - heat capacity at 298.15K (cal/(mol K))
]

# Indices of properties to predict (0-based into the 15-column property array)
# CSV columns: A(0), B(1), C(2), mu(3), alpha(4), homo(5), lumo(6), gap(7),
#              r2(8), zpve(9), u0(10), u298(11), h298(12), g298(13), cv(14)
TARGET_INDICES = {
    "gap":   7,   # HOMO-LUMO gap (Hartree)
    "homo":  5,   # HOMO energy (Hartree)
    "lumo":  6,   # LUMO energy (Hartree)
    "U0":    10,  # Internal energy at 0K (Hartree)
    "mu":    3,   # Dipole moment (Debye)
    "alpha": 4,   # Polarizability (Bohr^3)
    "Cv":    14,  # Heat capacity (cal/(mol K))
}

# Units for MAE reporting
TARGET_UNITS = {
    "gap": ("meV", HARTREE_TO_MEV),
    "homo": ("meV", HARTREE_TO_MEV),
    "lumo": ("meV", HARTREE_TO_MEV),
    "U0": ("meV", HARTREE_TO_MEV),
    "mu": ("D", 1.0),
    "alpha": ("Bohr^3", 1.0),
    "Cv": ("cal/(mol K)", 1.0),
}

# QM9 download URLs
QM9_URL = "https://springernature.figshare.com/ndownloader/files/3195389"
EXCLUDE_URL = "https://springernature.figshare.com/ndownloader/files/3195404"

# Known excluded molecule indices (3054 molecules with convergence issues)
N_TOTAL_QM9 = 133885
N_EXCLUDED = 3054
N_VALID_QM9 = N_TOTAL_QM9 - N_EXCLUDED


# ================================================================
# QM9 parsing (SDF + CSV from DeepChem mirror)
# ================================================================

def parse_sdf(sdf_path):
    """Parse gdb9.sdf into a list of (atoms, coords) tuples.

    Returns:
        list of (atoms, coords) where atoms is a list of element symbols
        and coords is an (N_atoms, 3) numpy array in Angstroms.
    """
    molecules = []
    with open(sdf_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Skip molecule name and software lines
        if i + 3 >= len(lines):
            break
        mol_name = lines[i].strip()
        i += 3  # skip name, software, blank line

        # Counts line: "N_atoms N_bonds ..."
        counts = lines[i].strip().split()
        n_atoms = int(counts[0])
        i += 1

        # Atom block
        atoms = []
        coords = []
        for _ in range(n_atoms):
            parts = lines[i].strip().split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            elem = parts[3]
            atoms.append(elem)
            coords.append([x, y, z])
            i += 1

        molecules.append((atoms, np.array(coords, dtype=np.float64)))

        # Skip to next molecule (past bond block, M END, $$$$)
        while i < len(lines) and lines[i].strip() != "$$$$":
            i += 1
        i += 1  # skip $$$$

    return molecules


def load_csv_properties(csv_path):
    """Load QM9 properties from gdb9.sdf.csv.

    Returns:
        mol_ids: list of molecule IDs (e.g., 'gdb_1')
        properties: (N, 19) array (A, B, C, mu, alpha, homo, lumo, gap,
                    r2, zpve, u0, u298, h298, g298, cv, u0_atom, u298_atom,
                    h298_atom, g298_atom)
        col_names: list of column names
    """
    mol_ids = []
    all_props = []
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        col_names = header[1:]  # skip mol_id
        for line in f:
            parts = line.strip().split(",")
            mol_ids.append(parts[0])
            all_props.append([float(x) for x in parts[1:]])
    return mol_ids, np.array(all_props, dtype=np.float64), col_names


def parse_xyz(filepath):
    """Parse a single QM9 XYZ file (kept for compatibility with figshare format).

    Returns:
        atoms: list of element symbols (e.g., ['C', 'H', 'H', 'H', 'H'])
        coords: (N_atoms, 3) numpy array in Angstroms
        properties: (15,) numpy array of float properties (A through Cv)
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())

    # Property line: "gdb INDEX PROP1 PROP2 ... PROP15"
    # QM9 uses *^ for scientific notation (Fortran-style)
    prop_line = lines[1].strip().replace("*^", "e")
    parts = prop_line.split()
    # parts[0] = "gdb", parts[1] = index, parts[2:17] = 15 properties
    properties = np.array([float(x) for x in parts[2:17]], dtype=np.float64)

    atoms = []
    coords = []
    for i in range(2, 2 + n_atoms):
        atom_parts = lines[i].strip().replace("*^", "e").split()
        atoms.append(atom_parts[0])
        coords.append([float(atom_parts[1]), float(atom_parts[2]),
                        float(atom_parts[3])])

    return atoms, np.array(coords, dtype=np.float64), properties


# ================================================================
# QM9 loading
# ================================================================

def download_qm9(data_dir=None):
    """Download QM9 dataset from DeepChem S3 mirror (SDF + CSV format)."""
    import urllib.request

    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    tar_path = os.path.join(data_dir, "gdb9.tar.gz")
    extract_dir = os.path.join(data_dir, "xyz")

    if not os.path.exists(tar_path):
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
        print(f"Downloading QM9 from DeepChem S3 ({url})...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        with open(tar_path, "wb") as f:
            f.write(data)
        print(f"  Downloaded {len(data)} bytes")

    if not os.path.exists(extract_dir) or \
       not os.path.exists(os.path.join(extract_dir, "gdb9.sdf")):
        print("Extracting...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)

    sdf_path = os.path.join(extract_dir, "gdb9.sdf")
    csv_path = os.path.join(extract_dir, "gdb9.sdf.csv")
    return sdf_path, csv_path


def load_qm9(data_dir=None):
    """Load the full QM9 dataset from DeepChem SDF + CSV format.

    Returns:
        all_coords: list of (N_atoms, 3) arrays in Angstroms
        all_atoms: list of lists of element symbols
        all_properties: (N_molecules, 15) array (A through Cv)
        all_indices: list of integer molecule indices
    """
    if data_dir is None:
        data_dir = DATA_DIR

    sdf_path, csv_path = download_qm9(data_dir)

    # Parse structures from SDF
    print("Parsing SDF structures...")
    molecules = parse_sdf(sdf_path)
    print(f"  Parsed {len(molecules)} molecules from SDF")

    # Parse properties from CSV
    print("Parsing CSV properties...")
    mol_ids, all_props_raw, col_names = load_csv_properties(csv_path)
    print(f"  Parsed {len(mol_ids)} property rows from CSV")
    print(f"  Columns: {col_names[:7]}... ({len(col_names)} total)")

    # Use the first 15 property columns (A through Cv),
    # skip the _atom columns which are atomization energies
    all_properties = all_props_raw[:, :15]

    # Build output lists (only molecules present in both SDF and CSV)
    n_use = min(len(molecules), len(mol_ids))
    all_coords = []
    all_atoms = []
    all_indices = []

    for i in range(n_use):
        atoms, coords = molecules[i]
        all_coords.append(coords)
        all_atoms.append(atoms)
        # Extract index from mol_id (e.g., 'gdb_1' -> 1)
        try:
            idx = int(mol_ids[i].split("_")[1])
        except (IndexError, ValueError):
            idx = i + 1
        all_indices.append(idx)

    all_properties = all_properties[:n_use]
    print(f"Loaded {n_use} QM9 molecules")
    print(f"  Atom types in dataset: "
          f"{sorted(set(a for mol in all_atoms for a in mol))}")
    print(f"  Max atoms per molecule: "
          f"{max(len(a) for a in all_atoms)}")
    return all_coords, all_atoms, all_properties, all_indices


# ================================================================
# Train / val / test split
# ================================================================

def get_splits(n_total, n_train=110000, n_val=10000, seed=42):
    """Standard QM9 split (SchNet convention).

    Returns:
        (train_idx, val_idx, test_idx) as numpy arrays of integer indices
        into the loaded molecule list (not QM9 molecule IDs).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ================================================================
# Synthetic molecule generator (for pipeline testing without QM9)
# ================================================================

def generate_synthetic_molecules(n_molecules=1000, seed=42):
    """Generate synthetic molecules for pipeline testing.

    Creates random 3D point clouds with atom types from {H, C, N, O, F}
    and synthetic properties that depend on molecular geometry in an
    SO(3)-invariant way. This tests the pipeline; real QM9 numbers
    require downloading the actual dataset.

    Returns same format as load_qm9: (coords, atoms, properties, indices)
    """
    rng = np.random.default_rng(seed)
    atom_symbols = ["H", "C", "N", "O", "F"]

    all_coords = []
    all_atoms = []
    all_properties = []
    all_indices = []

    for i in range(n_molecules):
        n_atoms = rng.integers(3, 20)

        # Random 3D positions (Angstrom scale)
        coords = rng.standard_normal((n_atoms, 3)) * 1.5

        # Center on center of mass (equal masses for simplicity)
        coords -= coords.mean(axis=0)

        # Random atom types (biased toward C and H like real molecules)
        weights = [0.4, 0.3, 0.1, 0.1, 0.1]
        atom_types = rng.choice(atom_symbols, size=n_atoms, p=weights)

        # Synthetic SO(3)-invariant properties based on geometry:
        distances = np.linalg.norm(coords, axis=1)
        pairwise_dists = np.linalg.norm(
            coords[:, None, :] - coords[None, :, :], axis=2)

        # 15 synthetic properties (matching QM9 property count)
        props = np.zeros(15)
        props[0] = np.mean(distances)                          # "A" - avg dist
        props[1] = np.std(distances)                           # "B" - std dist
        props[2] = np.max(distances)                           # "C" - max dist
        props[3] = float(n_atoms) * 0.1                        # "mu"
        props[4] = np.sum(distances ** 2)                      # "alpha"
        props[5] = -np.mean(pairwise_dists) * 0.01             # "homo"
        props[6] = -np.min(pairwise_dists[pairwise_dists > 0]) * 0.01  # "lumo"
        props[7] = props[6] - props[5]                         # "gap"
        props[8] = np.sum(distances ** 2)                      # "R2"
        props[9] = np.sum(1.0 / (pairwise_dists[pairwise_dists > 0.1])) * 0.001  # "zpve"
        props[10] = -np.sum(pairwise_dists) * 0.001            # "U0"
        props[11] = props[10] + rng.standard_normal() * 0.0001  # "U"
        props[12] = props[10] + rng.standard_normal() * 0.0001  # "H"
        props[13] = props[10] - 0.01 * n_atoms                 # "G"
        props[14] = float(n_atoms) * 3.0                       # "Cv"

        all_coords.append(coords)
        all_atoms.append(list(atom_types))
        all_properties.append(props)
        all_indices.append(i + 1)

    all_properties = np.array(all_properties, dtype=np.float64)
    print(f"Generated {n_molecules} synthetic molecules for pipeline testing")
    return all_coords, all_atoms, all_properties, all_indices
