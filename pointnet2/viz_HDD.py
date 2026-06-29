#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

# ---- your repo model ----
from pointnet2.models.sngp_hdd import get_model


# ---------------------- utils ----------------------
def load_point_label(file_path: Path):
    """txt: x y z [label]. Returns (xyz, gt_or_None)."""
    arr = np.loadtxt(str(file_path)).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    xyz = arr[:, :3]
    gt = arr[:, -1].astype(np.int64) if arr.shape[1] > 3 else None
    return xyz, gt

def pc_normalize(points: np.ndarray):
    c = np.mean(points, axis=0)
    pts = points - c
    m = np.max(np.linalg.norm(pts, axis=1) + 1e-12)
    return pts / (m + 1e-12)

def predictive_entropy_from_logits(logits_nc: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits_nc, dim=-1).clamp(min=1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.detach().cpu().numpy().astype(np.float32)

def robust_minmax(x: np.ndarray, lo=0.10, hi=0.90, eps=1e-6) -> np.ndarray:
    a = np.quantile(x, lo)
    b = np.quantile(x, hi)
    if not np.isfinite(a) or not np.isfinite(b):
        return np.zeros_like(x, dtype=np.float32)
    span = b - a
    if span < eps:
        mn, mx = x.min(), x.max()
        span2 = mx - mn
        if span2 < eps:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mn) / max(span2, eps)).clip(0, 1).astype(np.float32)
    return ((x - a) / span).clip(0, 1).astype(np.float32)

def color_map_blue_red01(vals01: np.ndarray) -> np.ndarray:
    v = vals01.reshape(-1, 1)
    c_lo = np.array([[0.20, 0.20, 1.00]], dtype=np.float32)  # blue
    c_hi = np.array([[1.00, 0.10, 0.10]], dtype=np.float32)  # red
    return (c_lo * (1 - v) + c_hi * v).astype(np.float32)

PALETTE = np.array([
    [31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],
    [140,86,75],[227,119,194],[127,127,127],[188,189,34],[23,190,207],
    [174,199,232],[255,187,120],[152,223,138],[255,152,150],[197,176,213],
    [196,156,148],[247,182,210],[199,199,199],[219,219,141],[158,218,229]
], dtype=np.float32) / 255.0

def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    return PALETTE[labels % len(PALETTE)]

def make_pcd(xyz: np.ndarray, rgb: np.ndarray | None = None, uniform_rgb=(0.70, 0.85, 1.00)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
    if rgb is not None:
        rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
        assert rgb.shape == (xyz.shape[0], 3), f"{rgb.shape} vs {(xyz.shape[0],3)}"
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        pcd.paint_uniform_color(uniform_rgb)  # light blue
    return pcd

def show_o3d_multi(geoms, title="Open3D", point_size=3.0, bg=(1,1,1)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=900)
    opt = vis.get_render_option()
    opt.background_color = np.array(bg, dtype=np.float32)
    opt.point_size = float(point_size)
    for g in geoms:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()

def rank01(x: np.ndarray) -> np.ndarray:
    idx = np.argsort(x, kind="mergesort")
    r = np.empty_like(idx, dtype=np.float64)
    r[idx] = np.arange(len(x), dtype=np.float64)
    return (r / max(len(x)-1, 1)).astype(np.float32)

def build_keep_mask_by_rank(key: np.ndarray, q_keep: float) -> np.ndarray:
    n = key.shape[0]
    k = max(1, int(np.floor(q_keep/100.0 * n)))
    order = np.argsort(key, kind="mergesort")  # 稳定
    kept_idx = order[:k]
    mask = np.zeros(n, dtype=bool)
    mask[kept_idx] = True
    return mask

def colorize_for_heat(values: np.ndarray, robust_lo=0.02, robust_hi=0.98) -> np.ndarray:
    v = values.astype(np.float64)
    if (v >= 0).all():
        v = np.log1p(v)
    else:
        v = np.log1p(np.maximum(v - v.min(), 0.0))
    a, b = np.quantile(v, robust_lo), np.quantile(v, robust_hi)
    span = max(b - a, 1e-12)
    v01 = np.clip((v - a) / span, 0.0, 1.0).astype(np.float32)
    if float(v01.max() - v01.min()) < 1e-6:
        v01 = rank01(values)
    return v01


# ---------------------- model io ----------------------
def read_state_dict(ckpt_path: Path):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ["model_state_dict", "state_dict", "network", "net", "model"]:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return obj

def build_model(num_classes: int, device: torch.device):
    return get_model(num_classes=num_classes, normal_channel=False).to(device)

def load_checkpoint(model: nn.Module, ckpt_path: Path):
    sd = read_state_dict(ckpt_path)
    _ = model.load_state_dict(sd, strict=False)
    return model


# ---------------------- main flow ----------------------
def parse_args():
    ap = argparse.ArgumentParser("Directly visualize a single file with SNGP σ² / Entropy / Hybrid heatmaps")
    ap.add_argument("--file", type=str, default=r'data_prepare\hdd_data\01234567\29_0.txt', help="Point cloud txt: x y z [label]")
    ap.add_argument("--ckpt", type=str, default=r'pointnet2\log\sngp_hdd\checkpoints\best_model.pth', help="Model checkpoint .pth")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--precision_cache", type=str, default="pointnet2\log\sngp_hdd\checkpoints\P_clean.pth", help="Optional ΦᵀΦ state to load into model")
    ap.add_argument("--no_calib", action="store_true", help="Skip precision-cache loading")
    ap.add_argument("--q_keep", type=float, default=75.0, help="Keep % (lowest-uncertainty)")
    ap.add_argument("--alpha_hybrid", type=float, default=1, help="Hybrid weight on sigma2 vs entropy")
    ap.add_argument("--q_lo", type=float, default=0.10, help="(unused in ranking) kept for backward compat")
    ap.add_argument("--q_hi", type=float, default=0.90, help="(unused in ranking) kept for backward compat")
    ap.add_argument("--ptsize", type=float, default=3.0)
    ap.add_argument("--save_mask", type=str, default="", help="Optional path to save kept mask .npy")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    fpath = Path(args.file)
    assert fpath.exists(), f"File not found: {fpath}"

    # load points
    xyz, _ = load_point_label(fpath)
    assert xyz.shape[0] > 0, "Empty point cloud."

    # build + load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.num_classes, device)
    model = load_checkpoint(model, Path(args.ckpt))
    model.eval()

    # forward once on full cloud to get logits and sigma^2
    x = torch.from_numpy(pc_normalize(xyz)).float().unsqueeze(0).permute(0, 2, 1).to(device)
    base_logits, sngp_logits, aux = model(x, compute_sigma=True)
    logits_nc = (sngp_logits if sngp_logits is not None else base_logits).squeeze(0)  # (N,C)

    sigma2 = aux["sigma2_geo"].squeeze(0).detach().cpu().numpy().astype(np.float32)  # (N,)
    entropy = predictive_entropy_from_logits(logits_nc)  # (N,)

    # ---------- KEY for KEEP (ranking only) ----------
    if abs(args.alpha_hybrid - 1.0) < 1e-9:
        key_for_mask = sigma2
    else:
        s_r = rank01(sigma2)      
        h_r = rank01(entropy)     
        key_for_mask = args.alpha_hybrid * s_r + (1.0 - args.alpha_hybrid) * h_r

    # ---------- KEEP lowest q% by rank ----------
    kept_mask = build_keep_mask_by_rank(key_for_mask, args.q_keep)
    if args.save_mask:
        np.save(args.save_mask, kept_mask.astype(np.bool_))

    # ---------- HEATMAP for visualization (independent of keep) ----------
    vals_for_color = sigma2 if abs(args.alpha_hybrid - 1.0) < 1e-9 else key_for_mask
    vals01 = colorize_for_heat(vals_for_color, robust_lo=0.02, robust_hi=0.98)
    colors_heat = color_map_blue_red01(vals01)
    pcd_heat = make_pcd(xyz, colors_heat)
    show_o3d_multi([pcd_heat],
                   title=f"Hybrid Heat (whole cloud), α={args.alpha_hybrid:.2f}",
                   point_size=args.ptsize)

    # ---------- segmentation on kept ----------
    kept_idx = np.where(kept_mask)[0]
    if kept_idx.size == 0:
        print("No points kept under current q_keep; nothing to segment.")
        return

    kept_logits = logits_nc[kept_idx]  # (K,C)
    kept_pred = kept_logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
    kept_xyz = xyz[kept_idx]

    geoms = [make_pcd(kept_xyz, labels_to_colors(kept_pred))]

    title = f"Segmentation on Kept (q_keep={args.q_keep:.0f}%) — kept={kept_idx.size}/{xyz.shape[0]}"
    show_o3d_multi(geoms, title=title, point_size=args.ptsize)



    # ---------- quick diagnostics ----------
    raw_mask_cmp = build_keep_mask_by_rank(sigma2, args.q_keep)
    print(f"[cmp] alpha={args.alpha_hybrid:.3f} kept_raw={raw_mask_cmp.sum()} kept_hyb={kept_mask.sum()} "
          f"identical_when_alpha1={(abs(args.alpha_hybrid-1.0)<1e-9) and np.array_equal(raw_mask_cmp, kept_mask)}")
    print(f"[sigma2] min={sigma2.min():.3e} max={sigma2.max():.3e} ptp={np.ptp(sigma2):.3e}")
    print(f"[entropy] min={entropy.min():.3e} max={entropy.max():.3e} ptp={np.ptp(entropy):.3e}")

if __name__ == "__main__":
    main()
