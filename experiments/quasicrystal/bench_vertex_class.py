"""Head-to-head: feature pipelines for vertex-class classification on
the icosahedral quasicrystal.

The task: each QC vertex has a "vertex class" determined by its
perp-space coordinate (the cut-and-project ground truth). This is
the local-environment label — vertices with the same class have
geometrically similar local neighborhoods up to icosahedral
rotation. A learner that only sees physical-space neighbor positions
must reconstruct this classification from local geometry alone.

Four feature pipelines compared at matched compute:

  - `naive`         : K nearest distances, sorted (K features)
  - `steinhardt`    : Q_l for l ∈ {2, 4, 6, 8, 10, 12}, 6 features.
                       Standard rotation-invariant local-order feature.
  - `icosahedral`   : Q_l restricted to icosahedrally-invariant
                       l ∈ {6, 10, 12}, 3 features.
  - `cell600_ade`   : 9 ADE eigenspace norms, derived from vMF
                       soft-assignment of neighbor directions onto
                       the 120 vertices of the 600-cell. Uses the
                       existing `ade_geometry.get_ade()` machinery.

Baseline classifier: scikit-learn-style Ridge logistic regression
implemented from scratch with numpy (no scipy/sklearn dep). Same
classifier across all four feature sets; the only difference is the
features.

Reports per-pipeline test accuracy, mean squared error of the
distance-to-nearest-class-centroid metric, and number of features.
Output: experiments/quasicrystal/checkpoints/vertex_class_bench.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.quasicrystal.cnp import generate, vertex_class
from experiments.quasicrystal.features import (
    nearest_distance_features,
    steinhardt_Q,
    icosahedral_features,
    cell600_irrep_features,
    cell600_irrep_coeffs,
    ssg_irrep_features,
)


N_CLASSES = 6
TRAIN_FRAC = 0.7


def _train_test_split(X, y, frac, rng):
    n = len(y)
    perm = rng.permutation(n)
    n_train = int(frac * n)
    return X[perm[:n_train]], y[perm[:n_train]], X[perm[n_train:]], y[perm[n_train:]]


def _standardize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X_train - mu) / sd, (X_test - mu) / sd


def _ridge_logreg_fit(X, y, n_classes, lam=1.0, n_iter=200, lr=0.5):
    """Multinomial logistic regression with L2 ridge, gradient descent.

    No sklearn / scipy dep. Returns (W, b) of shape ((d, C), (C,)).
    """
    n, d = X.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y]
    for _ in range(n_iter):
        z = X @ W + b
        z -= z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        p = ez / ez.sum(axis=1, keepdims=True)
        # Cross-entropy gradient + ridge.
        grad_W = X.T @ (p - Y) / n + lam * W
        grad_b = (p - Y).mean(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b


def _ridge_logreg_acc(W, b, X, y):
    z = X @ W + b
    return float(np.mean(z.argmax(axis=1) == y))


def evaluate_pipeline(name: str, X: np.ndarray, y: np.ndarray, seed: int = 0,
                      n_classes: int = N_CLASSES) -> dict:
    rng = np.random.default_rng(seed)
    X_train, y_train, X_test, y_test = _train_test_split(X, y, TRAIN_FRAC, rng)
    X_train, X_test = _standardize(X_train, X_test)

    # Tune lambda on a held-out validation chunk inside the train set.
    rng2 = np.random.default_rng(seed + 1)
    perm = rng2.permutation(len(y_train))
    cut = int(0.8 * len(y_train))
    Xtr = X_train[perm[:cut]]; ytr = y_train[perm[:cut]]
    Xva = X_train[perm[cut:]]; yva = y_train[perm[cut:]]

    best_lam = None
    best_acc = -1
    for lam in (0.001, 0.01, 0.1, 1.0):
        W, b = _ridge_logreg_fit(Xtr, ytr, n_classes, lam=lam, n_iter=400, lr=0.5)
        acc = _ridge_logreg_acc(W, b, Xva, yva)
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    # Refit with best lam on full train, evaluate on test.
    W, b = _ridge_logreg_fit(X_train, y_train, n_classes, lam=best_lam,
                              n_iter=600, lr=0.5)
    train_acc = _ridge_logreg_acc(W, b, X_train, y_train)
    test_acc = _ridge_logreg_acc(W, b, X_test, y_test)
    return {
        "name": name,
        "n_features": int(X.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "best_lam": best_lam,
        "val_acc": best_acc,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


def main(out_path: str, cutoff: int = 5, n_classes: int = N_CLASSES,
          n_neighbors: int = 12, seeds=(0, 1, 2, 3, 4)):
    print(f"[bench-qc] generating QC at cutoff={cutoff}...")
    qc = generate(cutoff=cutoff)
    y = vertex_class(qc, n_classes=n_classes)
    print(f"[bench-qc] {len(qc.physical)} vertices; class counts {np.bincount(y).tolist()}")

    print(f"[bench-qc] computing features (n_neighbors={n_neighbors})...")
    t = time.time()
    X_naive = nearest_distance_features(qc, n_neighbors)
    t_naive = time.time() - t

    t = time.time()
    X_stein = steinhardt_Q(qc, n_neighbors, ls=(2, 4, 6, 8, 10, 12))
    t_stein = time.time() - t

    t = time.time()
    X_icos = icosahedral_features(qc, n_neighbors)
    t_icos = time.time() - t

    t = time.time()
    X_ade = cell600_irrep_features(qc, n_neighbors)
    t_ade = time.time() - t

    t = time.time()
    X_ade_coeffs = cell600_irrep_coeffs(qc, n_neighbors)
    t_ade_coeffs = time.time() - t

    t = time.time()
    X_ssg = ssg_irrep_features(qc, n_neighbors)
    t_ssg = time.time() - t

    print(f"[bench-qc] feature times: naive={t_naive:.2f}s, "
          f"steinhardt={t_stein:.2f}s, icos={t_icos:.2f}s, "
          f"ade={t_ade:.2f}s, ade_coeffs={t_ade_coeffs:.2f}s, "
          f"ssg={t_ssg:.2f}s")

    pipelines = [
        ("naive", X_naive),
        ("steinhardt", X_stein),
        ("icosahedral", X_icos),
        ("cell600_ade_norms", X_ade),
        ("cell600_ade_coeffs", X_ade_coeffs),
        ("ssg_irrep (A+T1+H)", X_ssg),
        ("steinhardt+icosahedral", np.concatenate([X_stein, X_icos], axis=1)),
        ("naive+steinhardt", np.concatenate([X_naive, X_stein], axis=1)),
        ("naive+ade_coeffs", np.concatenate([X_naive, X_ade_coeffs], axis=1)),
        ("naive+ssg", np.concatenate([X_naive, X_ssg], axis=1)),
    ]

    results = {}
    for name, X in pipelines:
        runs = []
        for s in seeds:
            res = evaluate_pipeline(name, X, y, seed=s)
            runs.append(res)
        accs = np.array([r["test_acc"] for r in runs])
        results[name] = {
            "n_features": int(X.shape[1]),
            "test_acc_mean": float(accs.mean()),
            "test_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "test_acc_per_seed": accs.tolist(),
            "runs": runs,
        }
        print(f"[bench-qc] {name:30s} ({X.shape[1]:2d} feat): "
              f"test_acc {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")

    out = {
        "n_vertices": int(len(qc.physical)),
        "n_classes": n_classes,
        "n_neighbors": n_neighbors,
        "cutoff": cutoff,
        "seeds": list(seeds),
        "feature_times_s": {
            "naive": t_naive, "steinhardt": t_stein,
            "icosahedral": t_icos, "cell600_ade": t_ade,
        },
        "results": results,
        "class_counts": np.bincount(y).tolist(),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[bench-qc] wrote {out_path}")
    return out


def _ridge_regress(X_train, y_train, X_test, y_test, lam=1e-2):
    n, d = X_train.shape
    A = X_train.T @ X_train + lam * np.eye(d)
    w = np.linalg.solve(A, X_train.T @ y_train)
    pred_train = X_train @ w
    pred_test = X_test @ w
    ss_res = float(np.sum((y_test - pred_test) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "r2_test": r2,
        "rmse_test": float(np.sqrt(((y_test - pred_test) ** 2).mean())),
        "rmse_train": float(np.sqrt(((y_train - pred_train) ** 2).mean())),
    }


def stabilizer_class_classification(out_path: str, cutoff: int = 5,
                                     n_neighbors: int = 12,
                                     seeds=(0, 1, 2, 3, 4)):
    """Classify each QC vertex by its SSG orbit *size*.

    Each vertex lies in an SSG-orbit (the icosahedral group I acting
    on its Z⁶ coord); the orbit size ∈ {1, 12, 20, 30, 60}
    corresponds to the stabilizer order |I|/|orbit| ∈ {60, 5, 3, 2,
    1} — the local rotational symmetry of the vertex's neighborhood.
    A 5-fold-axis vertex (size-12 orbit) has a locally C₅-symmetric
    environment; a generic vertex has trivial stabilizer.

    This is an SSG-canonical label, defined directly from the
    superspace-group action. Rotation-invariant features detecting
    local point-group order (Steinhardt, icosahedral Q_l) should
    pick it up; direction-aware features should add nothing
    (the label is itself rotation-invariant).
    """
    from experiments.quasicrystal.ssg_decomposition import ssg_rotation_matrices
    print(f"[bench-qc-stab] generating QC at cutoff={cutoff}...")
    qc = generate(cutoff=cutoff)
    SSG = ssg_rotation_matrices()
    N = len(qc.integer_coords)
    ic = qc.integer_coords.astype(np.int64)

    print(f"[bench-qc-stab] computing SSG orbits over {N} vertices…")
    vert_index = {tuple(int(x) for x in v): i for i, v in enumerate(ic)}
    orbit_size = np.zeros(N, dtype=np.int32)
    visited = np.zeros(N, dtype=bool)
    for start in range(N):
        if visited[start]:
            continue
        orbit_vecs = set()
        for M in SSG:
            orbit_vecs.add(tuple(int(x) for x in M @ ic[start]))
        members = [vert_index[v] for v in orbit_vecs if v in vert_index]
        size = len(orbit_vecs)  # orbit size in Z^6 (not restricted to QC)
        for m in members:
            orbit_size[m] = size
            visited[m] = True

    # Binary label: 1 if vertex is on a high-symmetry axis (orbit size
    # < 60, i.e., non-trivial stabilizer), 0 if generic (size = 60).
    # 96.3% of vertices are generic, so we balance the classes by
    # undersampling the generic class.
    high_sym_mask = orbit_size < 60
    high_sym_idx = np.where(high_sym_mask)[0]
    generic_idx = np.where(~high_sym_mask)[0]
    n_balance = min(len(high_sym_idx), len(generic_idx))

    rng_bal = np.random.default_rng(42)
    keep_generic = rng_bal.choice(generic_idx, size=n_balance, replace=False)
    keep = np.concatenate([high_sym_idx, keep_generic])
    y = np.zeros(len(keep), dtype=np.int32)
    y[: len(high_sym_idx)] = 1  # high-symmetry = positive class
    print(f"[bench-qc-stab] balanced dataset: "
          f"{len(high_sym_idx)} high-symmetry vertices + "
          f"{n_balance} downsampled generic vertices "
          f"(total {len(keep)}, 50/50)")

    # Compute all feature pipelines on the *full* QC, then index.
    X_naive_full = nearest_distance_features(qc, n_neighbors)
    X_stein_full = steinhardt_Q(qc, n_neighbors, ls=(2, 4, 6, 8, 10, 12))
    X_icos_full = icosahedral_features(qc, n_neighbors)
    X_ade_norms_full = cell600_irrep_features(qc, n_neighbors)
    X_ssg_full = ssg_irrep_features(qc, n_neighbors)

    pipelines = [
        ("naive (rot-inv)", X_naive_full[keep]),
        ("steinhardt (rot-inv)", X_stein_full[keep]),
        ("icosahedral Q_l (rot-inv)", X_icos_full[keep]),
        ("cell600_ade_norms (rot-inv)", X_ade_norms_full[keep]),
        ("ssg_irrep A+T1+H (direction-aware)", X_ssg_full[keep]),
        ("steinhardt+icos", np.concatenate([X_stein_full[keep], X_icos_full[keep]], axis=1)),
    ]

    results = {}
    for name, X in pipelines:
        runs = []
        for s in seeds:
            res = evaluate_pipeline(name, X, y, seed=s, n_classes=2)
            runs.append(res)
        accs = np.array([r["test_acc"] for r in runs])
        results[name] = {
            "n_features": int(X.shape[1]),
            "test_acc_mean": float(accs.mean()),
            "test_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "test_acc_per_seed": accs.tolist(),
        }
        print(f"[bench-qc-stab] {name:42s} ({X.shape[1]:2d} feat): "
              f"acc {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")

    out = {
        "task": "stabilizer_class_classification_balanced_binary",
        "n_total_vertices": int(N),
        "n_high_symmetry": int(len(high_sym_idx)),
        "n_balanced_dataset": int(len(keep)),
        "n_neighbors": n_neighbors,
        "cutoff": cutoff,
        "label_meaning": "1 = on high-symmetry axis (stab > 1), 0 = generic",
        "results": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def perp_coordinate_regression(out_path: str, cutoff: int = 5,
                                n_neighbors: int = 12,
                                seeds=(0, 1, 2, 3, 4)):
    """Regression task: predict the perp-space x-coordinate of each
    vertex from its local physical-space environment.

    The target is NOT rotation-invariant — rotating the QC by an
    icosahedral element rotates the perp space too, in lockstep.
    This is the dual of the vertex-class task: rotation-invariant
    features should fail (no signal); direction-aware features
    (e.g. the full 600-cell ADE coefficient vector) should succeed.

    Reports R² on a held-out 30%.
    """
    print(f"[bench-qc-reg] generating QC at cutoff={cutoff}...")
    qc = generate(cutoff=cutoff)
    target = qc.perpendicular[:, 0].astype(np.float64)
    target -= target.mean()  # zero-center

    print(f"[bench-qc-reg] {len(qc.physical)} vertices, target std {target.std():.4f}")

    X_naive = nearest_distance_features(qc, n_neighbors)
    X_stein = steinhardt_Q(qc, n_neighbors, ls=(2, 4, 6, 8, 10, 12))
    X_ade_norms = cell600_irrep_features(qc, n_neighbors)
    X_ade_coeffs = cell600_irrep_coeffs(qc, n_neighbors)
    X_ssg = ssg_irrep_features(qc, n_neighbors)

    pipelines = [
        ("naive (rot-inv)", X_naive),
        ("steinhardt (rot-inv)", X_stein),
        ("cell600_ade_norms (rot-inv)", X_ade_norms),
        ("cell600_ade_coeffs (direction-aware)", X_ade_coeffs),
        ("ssg_irrep A+T1+H (direction-aware)", X_ssg),
        ("naive + ade_coeffs", np.concatenate([X_naive, X_ade_coeffs], axis=1)),
        ("naive + ssg", np.concatenate([X_naive, X_ssg], axis=1)),
    ]

    results = {}
    for name, X in pipelines:
        runs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            X_train, y_train, X_test, y_test = _train_test_split(X, target, TRAIN_FRAC, rng)
            X_train, X_test = _standardize(X_train, X_test)
            res = _ridge_regress(X_train, y_train, X_test, y_test, lam=1e-2)
            runs.append(res)
        r2s = np.array([r["r2_test"] for r in runs])
        results[name] = {
            "n_features": int(X.shape[1]),
            "r2_test_mean": float(r2s.mean()),
            "r2_test_std": float(r2s.std(ddof=1)) if len(r2s) > 1 else 0.0,
            "r2_test_per_seed": r2s.tolist(),
            "runs": runs,
        }
        print(f"[bench-qc-reg] {name:42s} ({X.shape[1]:3d} feat): "
              f"R²={r2s.mean():+.4f} ± {r2s.std(ddof=1):.4f}")

    out = {
        "task": "perp_coordinate_regression",
        "n_vertices": int(len(qc.physical)),
        "n_neighbors": n_neighbors,
        "cutoff": cutoff,
        "seeds": list(seeds),
        "results": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    out_cls = Path("experiments/quasicrystal/checkpoints/vertex_class_bench.json")
    main(str(out_cls))
    print()
    out_reg = Path("experiments/quasicrystal/checkpoints/perp_regression_bench.json")
    perp_coordinate_regression(str(out_reg))
    print()
    out_stab = Path("experiments/quasicrystal/checkpoints/stabilizer_bench.json")
    stabilizer_class_classification(str(out_stab))
