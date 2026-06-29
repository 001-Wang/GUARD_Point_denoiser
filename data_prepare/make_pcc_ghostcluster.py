#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD

=======
"""
make_pcc_ghostcluster.py
Append-only "rigid ghost cluster" noise generator for ShapeNetPart-style point clouds.
- Reads a JSON list of relative file paths (same style as your existing script).
- Resolves each entry under --dataset-root (tries "<path>" then "<path>.txt").
- Appends cloned clusters transformed by a random rigid transform (R, t).
- Writes to <output-root>/add_ghostcluster_s<sev>/<relative_path>.txt
- Point format preserved: (x y z label) OR (x y z nx ny nz label).

Default label policy matches common PointCloud-C practice (noise label = -1).
You can switch to label copying with: --label-mode copy

Usage:
  python make_pcc_ghostcluster.py \
    --json test_list.json \
    --dataset-root data/shapenetcore_partanno_segmentation_benchmark_v0_normal \
    --output-root data/shapenet_c_add \
    --severity 3 \
    --clusters 1 --ratio 0.25 --rot-deg 10 --trans-range 0.05 --jitter-std 0.0

"""
>>>>>>> 8c1fcb01970756c50538b0deed66253798650c9b

import os
import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path

def load_txt_pcd(path: Path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] == 7:
        xyz = arr[:, :3].astype(np.float32)
        nrm = arr[:, 3:6].astype(np.float32)
        lbl = arr[:, 6].astype(np.int64)
    elif arr.shape[1] == 4:
        xyz = arr[:, :3].astype(np.float32)
        nrm = None
        lbl = arr[:, 3].astype(np.int64)
    else:
        raise ValueError(f"Unsupported columns ({arr.shape[1]}) in {path}")
    return xyz, nrm, lbl

def save_txt_pcd(path: Path, xyz, lbl, nrm=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if nrm is None:
        out = np.concatenate([xyz.astype(np.float32), lbl.reshape(-1,1).astype(np.int64)], axis=1)
        np.savetxt(path, out, fmt="%.6f %.6f %.6f %d")
    else:
        out = np.concatenate([xyz.astype(np.float32), nrm.astype(np.float32), lbl.reshape(-1,1).astype(np.int64)], axis=1)
        np.savetxt(path, out, fmt="%.6f %.6f %.6f %.6f %.6f %.6f %d")

def random_rotation(deg, rng):
    theta = math.radians(rng.uniform(-deg, deg))
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    ux, uy, uz = axis
    c, s = math.cos(theta), math.sin(theta)
    C = 1 - c
    R = np.array([
        [c+ux*ux*C,     ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s, c+uy*uy*C,     uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, c+uz*uz*C    ]
    ], dtype=np.float32)
    return R

def choose_part_indices(lbl, prefer_large=True, rng=None):
    uniq, counts = np.unique(lbl, return_counts=True)
    if rng is None:
        rng = np.random.default_rng()
    if prefer_large:
        probs = counts / counts.sum()
        chosen = rng.choice(uniq, p=probs)
    else:
        chosen = rng.choice(uniq)
    return np.where(lbl == chosen)[0]

def make_ghost_clusters(xyz, lbl, nrm, *, clusters=1, ratio=0.2,
                        rot_deg=10.0, trans_range=0.05, jitter_std=0.0,
                        label_mode="minus1", rng=None, zero_normals=False):
    """
    label_mode: 'minus1' (default) or 'copy'
      - minus1: appended points get label -1 (matches common PCC policy)
      - copy  : appended points keep the source labels
    """
    if rng is None:
        rng = np.random.default_rng()
    N = xyz.shape[0]

    all_xyz = [xyz]
    all_lbl = [lbl]
    all_nrm = [nrm] if nrm is not None else None

    for _ in range(max(1, int(clusters))):
        idx_part = choose_part_indices(lbl, prefer_large=True, rng=rng)
        k = max(1, int(round(len(idx_part) * float(ratio))))
        k = min(k, len(idx_part))  # guard
        src_idx = rng.choice(idx_part, size=k, replace=False)

        src_xyz = xyz[src_idx].copy()
        src_lbl = lbl[src_idx].copy()
        src_nrm = nrm[src_idx].copy() if nrm is not None else None

        # Rigid transform
        R = random_rotation(rot_deg, rng)
        t = rng.uniform(-trans_range, trans_range, size=(1,3)).astype(np.float32)

        dst_xyz = (src_xyz @ R.T) + t
        if jitter_std > 0:
            dst_xyz = dst_xyz + rng.normal(0.0, jitter_std, size=dst_xyz.shape).astype(np.float32)

        if nrm is not None:
            if zero_normals:
                dst_nrm = np.zeros_like(src_nrm, dtype=np.float32)
            else:
                dst_nrm = (src_nrm @ R.T)
                dst_nrm /= (np.linalg.norm(dst_nrm, axis=1, keepdims=True) + 1e-9)
        else:
            dst_nrm = None

        if label_mode == "minus1":
            dst_lbl = np.full_like(src_lbl, fill_value=-1)
        elif label_mode == "copy":
            dst_lbl = src_lbl
        else:
            raise ValueError("label_mode must be 'minus1' or 'copy'")

        all_xyz.append(dst_xyz)
        all_lbl.append(dst_lbl)
        if all_nrm is not None:
            all_nrm.append(dst_nrm)

    new_xyz = np.concatenate(all_xyz, axis=0)
    new_lbl = np.concatenate(all_lbl, axis=0)
    new_nrm = np.concatenate(all_nrm, axis=0) if all_nrm is not None else None
    return new_xyz, new_lbl, new_nrm

def parse_args():
    p = argparse.ArgumentParser(description="Append-only rigid ghost cluster noise")
    p.add_argument("--json", type=str, required=True, help="JSON file listing test entries (relative paths).")
    p.add_argument("--dataset-root", type=str, required=True, help="Dataset root dir.")
    p.add_argument("--output-root", type=str, required=True, help="Output root dir.")
    p.add_argument("--severity", type=int, default=3, help="1..5 controls schedule scaling.")
    p.add_argument("--clusters", type=int, default=1, help="Base number of ghost clusters at severity=3.")
    p.add_argument("--ratio", type=float, default=0.25, help="Fraction of chosen-part points to clone per cluster.")
    p.add_argument("--rot-deg", type=float, default=30, help="Rotation bound in degrees (±).")
    p.add_argument("--trans-range", type=float, default=0.05, help="Uniform translation range per axis (±).")
    p.add_argument("--jitter-std", type=float, default=0.0, help="Gaussian jitter std for cloned points.")
    p.add_argument("--label-mode", type=str, default="minus1", choices=["minus1", "copy"],
                   help="minus1: new points labeled -1 (default). copy: keep source labels.")
    p.add_argument("--zero-normals", action="store_true", help="Zero normals of cloned points (if normals exist).")
    p.add_argument("--seed", type=int, default=2025, help="Random seed.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.json, "r") as f:
        rel_list = json.load(f)

    out_root = Path(args.output_root)
    ds_root  = Path(args.dataset_root)
    sev = max(1, min(5, int(args.severity)))
    out_dir = out_root / f"add_ghostcluster_s{sev}"

    # Severity schedule (conservative)
    schedule = {
        1: dict(mult_c=1.0, mult_r=0.5, mult_rt=0.5, mult_tr=0.5, mult_j=0.5),
        2: dict(mult_c=1.0, mult_r=0.75, mult_rt=0.75, mult_tr=0.75, mult_j=0.75),
        3: dict(mult_c=1.0, mult_r=1.0, mult_rt=1.0, mult_tr=1.0, mult_j=1.0),
        4: dict(mult_c=1.5, mult_r=1.25, mult_rt=1.25, mult_tr=1.25, mult_j=1.25),
        5: dict(mult_c=2.0, mult_r=1.5, mult_rt=1.5, mult_tr=1.5, mult_j=1.5),
    }[sev]

    total, miss = 0, 0
    for rel in rel_list:
        # Map "shape_data/<rest>" to "<dataset-root>/<rest>" (compat with your list format)
        if isinstance(rel, str) and rel.startswith("shape_data/"):
            rel = rel.replace("shape_data/", "")
        src = ds_root / rel
        if not src.exists():
            if src.with_suffix(".txt").exists():
                src = src.with_suffix(".txt")
            else:
                if args.verbose:
                    print("[MISS]", src)
                miss += 1
                continue

        xyz, nrm, lbl = load_txt_pcd(src)

        cfg = dict(
            clusters=int(round(args.clusters * schedule["mult_c"])),
            ratio=min(0.6, args.ratio * schedule["mult_r"]),
            rot_deg=args.rot_deg * schedule["mult_rt"],
            trans_range=args.trans_range * schedule["mult_tr"],
            jitter_std=args.jitter_std * schedule["mult_j"],
        )

        new_xyz, new_lbl, new_nrm = make_ghost_clusters(
            xyz, lbl, nrm,
            clusters=cfg["clusters"], ratio=cfg["ratio"],
            rot_deg=cfg["rot_deg"], trans_range=cfg["trans_range"],
            jitter_std=cfg["jitter_std"], label_mode=args.label_mode,
            rng=rng, zero_normals=args.zero_normals
        )

        dst = (out_dir / Path(rel)).with_suffix(".txt")
        save_txt_pcd(dst, new_xyz, new_lbl, nrm=new_nrm)
        total += 1
        if args.verbose and (total % 50 == 0):
            print(f"[{total}] wrote", dst)

    print(f"[GhostCluster] Done. Wrote {total} files. Missing inputs: {miss}.")
    print("Output root:", out_root)
    print("Subfolder created:", out_dir.relative_to(out_root))

if __name__ == "__main__":
    main()
