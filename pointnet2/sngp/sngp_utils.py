import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, math, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal import find_peaks
import logging, sys
# -----------------------------
# 2) Utilities
# -----------------------------
import numpy as np

from typing import Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_train_cache(path: Path, s2_all: np.ndarray, pe_all: np.ndarray, logger=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Optionally store sorted arrays for O(1) percentile via indexing
    np.savez_compressed(path, s2=np.sort(s2_all), pe=np.sort(pe_all))
    if logger: logger.info(f"[cache] saved TRAIN arrays to {path} (s2={s2_all.size}, pe={pe_all.size})")

def load_train_cache(path: Path, logger=None):
    obj = np.load(path)
    s2 = obj['s2']; pe = obj['pe']
    if logger: logger.info(f"[cache] loaded TRAIN arrays from {path} (s2={s2.size}, pe={pe.size})")
    return s2, pe

def q_from_sorted(arr_sorted: np.ndarray, q: float) -> float:
    """Fast percentile from sorted array (q in [0,1])."""
    if arr_sorted.size == 0: return float('nan')
    idx = min(arr_sorted.size-1, max(0, int(round(q*(arr_sorted.size-1)))))
    return float(arr_sorted[idx])



# --- replace your existing load_mask_with_coords with this one ---
# --- DROP-IN REPLACEMENT ---
def load_mask_with_coords(edge_dir: Path, sid: str,
                          *,                        # keyword-only
                          positive_labels=(1,),
                          allow_subdirs: bool = False,
                          exts = (".txt", ".npy")):
    """
    Try full sid first, then base (sid.split('_')[0]); optional one-level subdir search;
    supports multiple extensions. Returns:
      (mask_pts[N,3], mask_lbl[N], used_path or None, tried_paths, meta_dict)
    meta_dict: {'path': str|None, 'shape': tuple|None, 'raw_label_values': dict, 'bin_label_counts': dict}
    """
    tried = []

    # search names: sid and base
    candidates = [sid]
    base = sid.split("_")[0]
    if base != sid:
        candidates.append(base)

    # where to search
    search_dirs = [edge_dir]
    if allow_subdirs:
        for d in edge_dir.glob("*"):
            if d.is_dir():
                search_dirs.append(d)

    def _read_array(p: Path):
        if p.suffix.lower() == ".npy":
            arr = np.load(p, allow_pickle=False)
        else:  # .txt fallback
            try:
                arr = np.loadtxt(p, dtype=np.float32)
            except Exception:
                arr = np.genfromtxt(p, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    for d in search_dirs:
        for name in candidates:
            for ext in exts:
                p = d / f"{name}{ext}"
                tried.append(str(p))
                if not p.is_file():
                    continue
                arr = _read_array(p)
                if arr.size == 0 or arr.shape[1] < 4:
                    continue
                mask_pts = arr[:, :3].astype(np.float32)

                raw_lbl = arr[:, -1]
                if np.issubdtype(raw_lbl.dtype, np.floating):
                    mask_lbl_raw = np.rint(raw_lbl).astype(np.int64)
                else:
                    mask_lbl_raw = raw_lbl.astype(np.int64)

                pos_set = set(int(x) for x in positive_labels)
                mask_lbl_bin = np.array([1 if int(v) in pos_set else 0 for v in mask_lbl_raw], dtype=np.int64)

                meta = {
                    "path": str(p),
                    "shape": tuple(arr.shape),
                    "raw_label_values": {int(v): int(c) for v, c in zip(*np.unique(mask_lbl_raw, return_counts=True))},
                    "bin_label_counts":  {int(v): int(c) for v, c in zip(*np.unique(mask_lbl_bin,  return_counts=True))},
                }
                # We return raw labels; your build_* will re-read from disk using resolved path.
                return mask_pts, mask_lbl_raw, str(p), tried, meta

    return None, None, None, tried, {"path": None, "shape": None, "raw_label_values": {}, "bin_label_counts": {}}




def plot_distribution(x, name, percentile, save_dir: Path, bins=100):
    if x.size == 0 or save_dir is None:
        return
    p = float(np.quantile(x, percentile/100.0)) if x.size else float('nan')

    # Histogram
    fig = plt.figure(figsize=(7,4.2))
    plt.hist(x, bins=bins, density=True)
    plt.axvline(p, linestyle="--")
    plt.title(f"{name} distribution (q={percentile}%)")
    plt.xlabel(name); plt.ylabel("density")
    fig.tight_layout()
    fig.savefig(save_dir / f"{name.replace(' ', '_').lower()}_hist_q{int(percentile)}.png", dpi=160)
    plt.close(fig)

    # Empirical CDF
    xs = np.sort(x)
    ys = np.linspace(0, 1, xs.size, endpoint=True)
    fig2 = plt.figure(figsize=(7,4.2))
    plt.plot(xs, ys)
    plt.axvline(p, linestyle="--")
    plt.title(f"{name} empirical CDF (q={percentile}%)")
    plt.xlabel(name); plt.ylabel("F(x)")
    fig2.tight_layout()
    fig2.savefig(save_dir / f"{name.replace(' ', '_').lower()}_cdf_q{int(percentile)}.png", dpi=160)
    plt.close(fig2)


def bh_fdr_threshold_from_scores(
    scores: np.ndarray,
    q: float = 0.35,
    tail: str = "gt",           # 'gt': #(x > s_i)/n (more aggressive) ; 'ge': #(x >= s_i)/n (more conservative)
    method: str = "storey",     # 'bh' (classic Benjamini–Hochberg) or 'storey' (BH with pi0 correction)
    lambda_grid: Optional[np.ndarray] = None,  # for Storey; None → np.linspace(0.4, 0.9, 6)
    jitter_eps: float = 0.0,    # add tiny jitter only when all-ties; else leave scale intact
    seed: Optional[int] = 0,
    return_mask: bool = True
) -> Tuple[float, Optional[np.ndarray]]:
    """
    FDR threshold on *right tail* (larger score = more positive).
    Returns (thr, mask). If no discovery, thr=+inf and mask=None.
    """
    x = np.asarray(scores, float).copy()
    n = x.size
    if n == 0:
        return float("inf"), None

    # Optional tiny jitter only if nearly all equal (prevents undefined ranks while keeping reproducible)
    if jitter_eps and np.std(x) < 1e-12:
        rng = np.random.default_rng(seed)
        x += rng.uniform(-jitter_eps, jitter_eps, size=n)

    xs = np.sort(x)  # ascending

    # Right-tail p-values
    if tail == "gt":
        ub = np.searchsorted(xs, x, side="right")   # count of <=
        tail_count = n - ub                          # count of >
    elif tail == "ge":
        lb = np.searchsorted(xs, x, side="left")    # first index of >=
        tail_count = n - lb                          # count of >=
    else:
        raise ValueError("tail must be 'gt' or 'ge'")

    p = tail_count / n  # smaller p → more significant

    # Optional Storey pi0 correction (less conservative when many true effects)
    if method.lower() == "storey":
        if lambda_grid is None:
            lambda_grid = np.linspace(0.4, 0.9, 6)  # typical grid
        pi0s = []
        for lam in lambda_grid:
            # pi0(lam) ≈ # {p_i > lam} / ( (1-lam) * n )
            pi0_est = np.mean(p > lam) / max(1e-9, (1.0 - lam))
            pi0s.append(pi0_est)
        # use a robust central tendency, clipped to [0.5, 1.0]
        pi0 = float(np.clip(np.median(pi0s), 0.5, 1.0))
        q_eff = q / max(pi0, 1e-9)
    else:
        q_eff = q

    order = np.argsort(p)           # p_(1) ≤ p_(2) ≤ ...
    p_sorted = p[order]
    line = (np.arange(1, n+1) / n) * q_eff

    ok = np.where(p_sorted <= line)[0]
    if ok.size == 0:
        return float("inf"), None

    k = int(ok[-1])
    idx_sel = order[:k+1]           # selected by BH
    thr = float(np.min(x[idx_sel])) # since larger = more positive

    mask = np.zeros(n, dtype=bool)
    mask[idx_sel] = True
    return thr, mask


def select_adaptive(
    scores: np.ndarray,
    aux_tiebreak: Optional[np.ndarray] = None,
    p_min: float = 0.01,
    p_max: float = 0.30,
    q_fdr: float = 0.35,
    tail: str = "gt",
    use_ranks: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Adaptive per-shape selection:
      1) FDR on right tail (BH/Storey) applied to scores (optionally rank-normalized)
      2) Clip selected count into [p_min, p_max] only if outside
      3) Degenerate fallback: use aux_tiebreak Top-k if all 0 or all 1
    Returns (pred_binary, threshold_used_on_fdr_space, selected_count)
    """
    s = np.asarray(scores, float)
    N = s.size
    lo = max(1, int(np.ceil(p_min * N)))
    hi = max(lo, int(np.floor(p_max * N)))

    # Rank-normalize makes FDR scale-invariant across shapes
    if use_ranks:
        # ranks in [0..1]; higher rank = larger score
        # (N+1) in denom prevents rank==1.0 edge case
        r = (np.argsort(np.argsort(s)) + 1) / (N + 1.0)
        s_for_fdr = r
    else:
        s_for_fdr = s

    thr, mask = bh_fdr_threshold_from_scores(
        s_for_fdr, q=q_fdr, tail=tail, method="storey", jitter_eps=1e-12, seed=0, return_mask=True
    )
    if mask is None:
        pred = np.zeros(N, np.int32)
        m = 0
    else:
        pred = mask.astype(np.int32)
        m = int(mask.sum())

    # Clip to [lo, hi] only if needed
    if m < lo:
        # include top lo by original scores
        kth = np.partition(s, -lo)[-lo]
        pred = (s >= kth).astype(np.int32)
        m = lo
    elif m > hi:
        # cap to top hi by original scores
        kth = np.partition(s, -hi)[-hi]
        pred = (s >= kth).astype(np.int32)
        m = hi

    # Degenerate fallback
    if (pred.sum() == 0 or pred.sum() == N) and (aux_tiebreak is not None):
        tgt = max(1, min(N, int(np.clip(m, lo, hi))))
        kth = np.partition(aux_tiebreak, -tgt)[-tgt]
        pred = (aux_tiebreak >= kth).astype(np.int32)
        m = int(pred.sum())

    return pred, float(thr), int(m)


@torch.no_grad()
def calibrate_precision(model, shapes, device, k=64, sample_n=2048):
    """
    用前 k 个样本累积一次 Z^T Z 来点亮 precision（无标签、无反传）。
    sample_n: 每个样本最多取多少点，防 OOM。
    """
    # 1) 正确重置：ridge * I（只填对角线）
    P = model.gp_head.precision
    P.zero_()
    P.diagonal().fill_(float(model.gp_head.ridge))

    # 2) 整体 eval，但允许 head 更新
    model.eval()
    model.gp_head.train()

    print("[calib] start; trace before =", float(torch.trace(P)))
    seen = 0

    for i, item in enumerate(shapes):
        sid, path = item if isinstance(item, tuple) else (item.stem, item)
        pts = load_points(path)
        if pts.shape[0] == 0:
            continue
        if pts.shape[0] > sample_n:
            idx = np.random.choice(pts.shape[0], sample_n, replace=False)
            pts = pts[idx]

        x = (torch.from_numpy(pc_normalize(pts)).float()
                .unsqueeze(0).permute(0,2,1).to(device))

        # ★ 强制更新 precision，不计算 σ²
        _ = model(x, update_precision=True, compute_var=False, force_update=True)
        seen += pts.shape[0]

        if (i + 1) % 8 == 0:
            print(f"[calib] processed {i+1} shapes, trace = {float(torch.trace(P)):.1f}")

        if i + 1 >= k:
            break

    model.eval()
    print(f"[calib] done; points ~{seen}, trace now = {float(torch.trace(P))}")



def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_points(path: Path):
    arr = np.loadtxt(path)
    # use first 3 columns as xyz
    pts = arr[:, :3].astype(np.float32)
    return pts

def pc_normalize(points: np.ndarray):
    """Normalize to zero-mean and unit sphere."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.linalg.norm(points, axis=1))
    if m > 0:
        points = points / m
    return points

import numpy as np, re
from pathlib import Path

# 与 _drop 一致：整型网格量化（0.01 单位）
def quantize_xyzn(arr, scale=100):
    return np.round(arr * scale).astype(np.int32)

def _load_array_txt(path: Path):
    try:
        return np.loadtxt(path, dtype=np.float32)
    except Exception:
        return np.genfromtxt(path, dtype=np.float32)

# 先用完整 sid，再退回 base（sid.split('_')[0]）；支持 .txt/.npy
def load_mask_with_coords_simple(edge_dir: Path, sid: str):
    tried = []
    for name in (sid, sid.split("_")[0]):
        for ext in (".txt", ".npy"):
            p = edge_dir / f"{name}{ext}"
            tried.append(str(p))
            if p.is_file():
                arr = _load_array_txt(p) if p.suffix == ".txt" else np.load(p)
                if arr.ndim == 1: arr = arr[None, :]
                return (arr[:, :3].astype(np.float32),
                        arr[:, -1].astype(np.int64),
                        str(p), tried)
    return None, None, None, tried

def build_edge_gt_match_drop_diag(
    pts_raw: np.ndarray,
    mask_dir: Path,
    sid: str,
    scale: int = 100,
    positive_labels=(2,),   # 想把 1 也当正就写 (1,2)
    verbose: bool = True,
    logger=None
):
    if logger is None:
        logger = logging.getLogger("sngp")

    mask_pts, mask_lbl, hit_path, tried, _ = load_mask_with_coords(Path(mask_dir), sid)
    if mask_pts is None:
        if verbose:
            print(f"[mask] {sid}: NOT FOUND. tried={tried}", flush=True)
        return np.zeros(pts_raw.shape[0], dtype=np.int64)

    pos_set = set(int(v) for v in positive_labels)

    q_mask = quantize_xyzn(mask_pts, scale=scale)
    q_raw  = quantize_xyzn(pts_raw,  scale=scale)

    # ---- 构造查找表（正为王） ----
    lookup = {}
    for k, v in zip(map(tuple, q_mask), mask_lbl):
        if int(v) in pos_set:
            lookup[k] = 1
        else:
            lookup.setdefault(k, 0)

    # 计算匹配率 + 生成 gt
    set_mask = set(map(tuple, q_mask))
    raw_keys = list(map(tuple, q_raw))
    gt = np.fromiter((lookup.get(k, 0) for k in raw_keys), dtype=np.int64)
    return gt


def prf_counts(pred: np.ndarray, gt: np.ndarray):
    TP = int(((pred == 1) & (gt == 1)).sum())
    FP = int(((pred == 1) & (gt == 0)).sum())
    FN = int(((pred == 0) & (gt == 1)).sum())
    return TP, FP, FN

def prf_from_counts(TP, FP, FN):
    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)
    return P, R, F1


def list_shapes_from_json(json_path: Path, pts_dir: Path):

    with open(json_path, "r") as f:
        shape_paths = json.load(f)

    shape_ids = [Path(p.strip('"')).stem for p in shape_paths]
    pairs = []
    for sid in shape_ids:
        p_txt = pts_dir / f"{sid}.txt"
        if p_txt.exists():
            pairs.append((sid, p_txt))
        else:
            # 找不到就跳过；你也可以在这里 raise
            print(f"[WARN] missing point file for sid={sid} under {pts_dir}")
    return pairs

def list_shapes_scan_dir(pts_dir: Path):
    files = sorted([p for p in pts_dir.iterdir() if p.suffix.lower() in ('.txt', '.npy')])
    return [(p.stem, p) for p in files]
