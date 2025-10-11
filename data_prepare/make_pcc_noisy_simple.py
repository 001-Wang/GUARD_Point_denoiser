#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pcc_noisy_simple.py
Create PointCloud-C–style noisy test data (add_global/add_local, severities 1–5)
from a ShapeNetPart test list, with minimal CLI usage.

Key features:
- Accepts your JSON list (entries like "shape_data/02691156/xxxx") and a --dataset-root.
- Automatically rewrites each entry: "shape_data/<rest>" -> "<dataset-root>/<rest>".
- Tries both "<path>" and "<path>.txt".
- Outputs to: <output-root>/<mode>_s<sev>/<relative_path>.txt (relative to dataset root).

Defaults:
- Keep point count constant (replace a fraction with outliers).
- Outlier labels = -1 (ignore), outlier normals = 0.
- Severities: ratio = [0.05, 0.10, 0.20, 0.30, 0.40].

Usage (single line):
  python make_pcc_noisy_simple.py --list "data/.../shuffled_test_file_list.json" \
      --dataset-root "data/shapenetcore_partanno_segmentation_benchmark_v0_normal" \
      --output-root "data/shapenet_c_noisy"

Author: ChatGPT
"""

import argparse
import json
from pathlib import Path
import numpy as np
import random
import sys

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


def parse_args():
    ap = argparse.ArgumentParser(description="Make PointCloud-C noisy ShapeNetPart test sets.")
    ap.add_argument("--list", required=True, type=str, help="JSON file with test file paths (usually 'shape_data/...').")
    ap.add_argument("--dataset-root", required=True, type=str, help="Root folder of the clean dataset.")
    ap.add_argument("--output-root", required=True, type=str, help="Where to write noisy datasets.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true", help="Append outliers (increase N) instead of replacing.")
    ap.add_argument("--label-mode", default="ignore", choices=["ignore", "nearest"], help="Labels for injected outliers.")
    ap.add_argument("--zero-normals", action="store_true", default=True, help="Set normals of outliers to zero.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


SEV_RATIOS = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.30, 5: 0.40}
SEV_BBOX_EXP = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.30, 5: 0.40}  # add_global
SEV_BLOB_STD = {1: 0.010, 2: 0.015, 3: 0.020, 4: 0.030, 5: 0.040}  # add_local


def sev_num_blobs(N, s):
    frac = {1: 0.01, 2: 0.015, 3: 0.02, 4: 0.035, 5: 0.05}[s]
    return int(np.clip(int(np.ceil(frac * N)), 4, 64))


def load_list(fp: Path):
    data = json.loads(fp.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("List JSON must be an array of paths.")
    # normalize slashes
    return [p.replace("\\", "/") for p in data]


def resolve_path(dataset_root: Path, item: str):
    # Rewrite "shape_data/<rest>" --> dataset_root/<rest>
    p = item
    if p.startswith("shape_data/"):
        rel = p[len("shape_data/"):]
        abs1 = dataset_root / rel
    else:
        # treat as relative to dataset_root
        abs1 = dataset_root / p
    if abs1.exists():
        return abs1
    # try with .txt
    if abs1.suffix.lower() != ".txt":
        abs2 = abs1.with_suffix(".txt")
        if abs2.exists():
            return abs2
    return None


def bbox_stats(xyz):
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    return mn, mx, diag


def sample_add_global(n_out, mn, mx, expansion, rng):
    size = mx - mn
    mn_e = mn - expansion * size
    mx_e = mx + expansion * size
    return rng.random((n_out, 3)) * (mx_e - mn_e) + mn_e


def sample_add_local(xyz, n_out, num_blobs, std_scale, rng):
    if n_out <= 0:
        return np.empty((0, 3), dtype=float)
    N = xyz.shape[0]
    centers = xyz[rng.choice(N, size=min(N, max(1, num_blobs)), replace=False)]
    _, _, diag = bbox_stats(xyz)
    std = std_scale * diag
    per = np.full(len(centers), n_out // len(centers), dtype=int)
    per[: (n_out % len(centers))] += 1
    out = []
    for c, m in zip(centers, per):
        if m <= 0:
            continue
        noise = rng.normal(0.0, std, size=(m, 3))
        out.append(c + noise)
    return np.vstack(out) if out else np.empty((0, 3), dtype=float)


def assign_labels_normals(xyz, nrm, lbl, out_xyz, mode="ignore", zero_normals=True):
    n_out = out_xyz.shape[0]
    if n_out == 0:
        return np.zeros((0, 3), dtype=float), np.full((0,), -1, dtype=int)
    if mode == "ignore":
        return np.zeros((n_out, 3), dtype=float), np.full((n_out,), -1, dtype=int)
    # nearest
    if NearestNeighbors is None:
        d2 = ((out_xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
        idx = np.argmin(d2, axis=1)
    else:
        nbrs = NearestNeighbors(n_neighbors=1).fit(xyz)
        idx = nbrs.kneighbors(out_xyz, return_distance=False).reshape(-1)
    out_lbl = lbl[idx].astype(int)
    out_nrm = np.zeros((n_out, 3), dtype=float) if zero_normals else nrm[idx].astype(float)
    return out_nrm, out_lbl


def write_noisy(abs_in: Path, abs_out: Path, mode: str, sev: int, append: bool,
                label_mode: str, zero_normals: bool, rng: np.random.Generator, verbose=False):
    arr = np.loadtxt(abs_in, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"{abs_in} needs >=7 columns (x y z nx ny nz label).")
    xyz = arr[:, :3].astype(float)
    nrm = arr[:, 3:6].astype(float)
    lbl = arr[:, 6].astype(int)

    N = xyz.shape[0]
    ratio = SEV_RATIOS[sev]
    n_out = int(np.round(ratio * N))

    mn, mx, _ = bbox_stats(xyz)

    if mode == "add_global":
        out_xyz = sample_add_global(n_out, mn, mx, SEV_BBOX_EXP[sev], rng)
    else:
        out_xyz = sample_add_local(xyz, n_out, sev_num_blobs(N, sev), SEV_BLOB_STD[sev], rng)

    out_nrm, out_lbl = assign_labels_normals(xyz, nrm, lbl, out_xyz, mode=label_mode, zero_normals=zero_normals)

    if append:
        new_xyz = np.vstack([xyz, out_xyz])
        new_nrm = np.vstack([nrm, out_nrm])
        new_lbl = np.concatenate([lbl, out_lbl])
    else:
        # replace random subset to keep N
        idx = rng.choice(N, size=n_out, replace=False) if n_out > 0 else np.array([], dtype=int)
        new_xyz = xyz.copy()
        new_nrm = nrm.copy()
        new_lbl = lbl.copy()
        if n_out > 0:
            new_xyz[idx] = out_xyz
            new_nrm[idx] = out_nrm
            new_lbl[idx] = out_lbl

    out_arr = np.hstack([new_xyz, new_nrm, new_lbl.reshape(-1, 1)])
    abs_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(abs_out, out_arr, fmt="%.6f %.6f %.6f %.6f %.6f %.6f %d")
    if verbose:
        print(f"[OK] {mode} s{sev}: {abs_in} -> {abs_out} | N={N} out={n_out}")


def main():
    args = parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    lst_path = Path(args.list).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_root).resolve()

    items = load_list(lst_path)

    modes = ["add_global", "add_local"]
    sevs = [1, 2, 3, 4, 5]

    total = 0
    miss = 0

    for item in items:
        abs_in = resolve_path(dataset_root, item)
        if abs_in is None or not abs_in.exists():
            if args.verbose:
                print(f"[MISS] {item}")
            miss += 1
            continue

        # Build a relative path (w.r.t dataset root) for output mirroring
        try:
            rel = abs_in.relative_to(dataset_root)
        except ValueError:
            # fallback: use filename only
            rel = abs_in.name

        for mode in modes:
            for s in sevs:
                abs_out = out_root / f"{mode}_s{s}" / rel
                write_noisy(abs_in, abs_out, mode, s,
                            append=args.append,
                            label_mode=args.label_mode,
                            zero_normals=args.zero_normals,
                            rng=rng, verbose=args.verbose)
                total += 1

    print(f"Done. Wrote {total} files. Missing inputs: {miss}.")
    print("Output root:", out_root)
    print("Subfolders created:", ", ".join([f"add_global_s{i}" for i in range(1, 6)]) + ", " +
          ", ".join([f"add_local_s{i}" for i in range(1, 6)]))


if __name__ == "__main__":
    main()
