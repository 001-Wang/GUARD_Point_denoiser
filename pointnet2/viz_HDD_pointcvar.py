
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import importlib
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

# ---------------------- I/O utils ----------------------
def load_point_label(file_path: Path):
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

PALETTE = np.array([
    [31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],
    [140,86,75],[227,119,194],[127,127,127],[188,189,34],[23,190,207],
    [174,199,232],[255,187,120],[152,223,138],[255,152,150],[197,176,213],
    [196,156,148],[247,182,210],[199,199,199],[219,219,141],[158,218,229]
], dtype=np.float32) / 255.0

def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    return PALETTE[labels % len(PALETTE)]

def make_pcd(xyz: np.ndarray, rgb: np.ndarray = None, uniform_rgb=(0.70, 0.85, 1.00)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
    if rgb is not None:
        rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
        assert rgb.shape == (xyz.shape[0], 3), f"{rgb.shape} vs {(xyz.shape[0],3)}"
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        pcd.paint_uniform_color(uniform_rgb)
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

# ---------------------- Model helpers ----------------------
def smart_instantiate(model_path: str, num_part: int, normal_channel: bool):
    mod = importlib.import_module(model_path)
    if hasattr(mod, "get_model"):
        for fn in (
            lambda: mod.get_model(num_classes=num_part, normal_channel=normal_channel),
            lambda: mod.get_model(num_classes=num_part),
            lambda: mod.get_model(num_part),
        ):
            try:
                return fn()
            except Exception:
                continue
    for name in ("PointNet2", "DGCNN", "Model", "Net"):
        if hasattr(mod, name):
            return getattr(mod, name)()
    raise RuntimeError(f"Cannot instantiate model from {model_path}")

def read_state_dict(ckpt_path: Path):
    # Allow full unpickling for local, trusted checkpoints.
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ["model_state_dict", "state_dict", "network", "net", "model"]:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return obj

def load_checkpoint(model: nn.Module, ckpt_path: Path):
    sd = read_state_dict(ckpt_path)
    # Strip common prefixes
    sd = {k.replace("module.", "").replace("model.", "").replace("net.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded checkpoint {ckpt_path} (missing={len(missing)} unexpected={len(unexpected)})")
    if missing:
        print("  - missing (showing up to 5):", list(missing)[:5])
    if unexpected:
        print("  - unexpected (up to 5):", list(unexpected)[:5])
    return model

# ---------------------- PointCVaR risk ----------------------

def forward_logits(model: nn.Module, x_bcn: torch.Tensor):
    out = model(x_bcn, compute_sigma=False) if "compute_sigma" in model.forward.__code__.co_varnames else model(x_bcn)
    if isinstance(out, tuple) and len(out) >= 1:
        base_logits = out[0]
    else:
        base_logits = out
    # try to align to (B,N,C)
    if base_logits.dim() == 3 and base_logits.shape[1] == x_bcn.shape[-1]:
        logits_bnc = base_logits
    else:
        logits_bnc = base_logits.permute(0, 2, 1).contiguous()
    return logits_bnc

def grad_risk_norm(model: nn.Module, x_bcn: torch.Tensor) -> torch.Tensor:
    x_req = x_bcn.clone().requires_grad_(True)
    logits_bnc = forward_logits(model, x_req)  # (B,N,C)
    pred_bN = logits_bnc.argmax(dim=-1)        # (B,N)
    chosen = torch.gather(logits_bnc, dim=-1, index=pred_bN.unsqueeze(-1)).squeeze(-1)  # (B,N)
    loss = chosen.sum()
    model.zero_grad(set_to_none=True)
    if x_req.grad is not None:
        x_req.grad.zero_()
    loss.backward()
    g = x_req.grad  # (B,C,N) with C=3 or 6 depending on normal_channel
    risk = torch.sqrt((g ** 2).sum(dim=1) + 1e-12)  # (B,N)
    return risk

# ---------------------- CLI and main ----------------------
def parse_args():
    ap = argparse.ArgumentParser("HDD PointCVaR kept-only visualizer (viz_HDD.py style)")
    ap.add_argument("--file", type=str, required=True, help="Point cloud txt: x y z [label]")
    ap.add_argument("--model", type=str, default="pointnet2.models.pointnet2_hdd",
                    help="Backbone module path with get_model (e.g., pointnet2.models.pointnet2_part_seg_msg)")
    ap.add_argument("--ckpt", type=str, default=r'pointnet2\log\without_normal_hdd\checkpoints\best_model.pth', help="Checkpoint .pth for the chosen backbone")
    ap.add_argument("--num_part", type=int, default=5, help="number of part classes (HDD)")
    ap.add_argument("--normal_channel", action="store_true",
                    help="if set, feed Nx6 (xyz+normals); otherwise Nx3")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--q_keep", type=float, default=75.0, help="Keep percent (lowest-risk)")
    ap.add_argument("--ptsize", type=float, default=3.0)
    ap.add_argument("--no_view", action="store_true")
    ap.add_argument("--save_mask", type=str, default="")
    return ap.parse_args()

def main():
    args = parse_args()
    fpath = Path(args.file)
    assert fpath.exists(), f"File not found: {fpath}"
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # Load data
    xyz, gt = load_point_label(fpath)
    assert xyz.shape[0] > 0, "Empty point cloud."
    N = xyz.shape[0]

    # Build model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = smart_instantiate(args.model, args.num_part, normal_channel=args.normal_channel).to(device).eval()
    model = load_checkpoint(model, ckpt_path)

    # Prepare input (B,C,N) with optional normals
    pts = pc_normalize(xyz).astype(np.float32)
    if args.normal_channel:
        if xyz.shape[1] >= 6:
            feats = np.concatenate([pts, xyz[:, 3:6].astype(np.float32)], axis=1)  # use provided normals
        else:
            feats = np.concatenate([pts, np.zeros_like(pts)], axis=1)  # pad zeros if normals absent
    else:
        feats = pts
    x_bcn = torch.from_numpy(feats.T).unsqueeze(0).to(device)  # (1,C,N)

    # First forward (no grad)
    with torch.no_grad():
        logits_bnc = forward_logits(model, x_bcn)  # (1,N,C)

    # Risk (with grad)
    risk_bN = grad_risk_norm(model, x_bcn)  # (1,N)
    risk = risk_bN[0].detach().cpu().numpy().astype(np.float32)

    # Keep lowest q%
    K = max(1, int(np.floor(args.q_keep / 100.0 * N)))
    kept_idx = np.argsort(risk, kind="mergesort")[:K]
    kept_xyz = xyz[kept_idx]
    kept_pred = logits_bnc[0, kept_idx].argmax(dim=-1).cpu().numpy().astype(np.int64)

    # Show only kept points
    geoms = [make_pcd(kept_xyz, labels_to_colors(kept_pred))]
    title = f"Segmentation on Kept (q_keep={args.q_keep:.0f}%) — kept={kept_idx.size}/{xyz.shape[0]}"
    if not args.no_view:
        show_o3d_multi(geoms, title=title, point_size=args.ptsize)

    # Optional mask save
    if args.save_mask:
        mask = np.zeros(N, dtype=bool)
        mask[kept_idx] = True
        np.save(args.save_mask, mask.astype(np.bool_))

    print(f"[risk] min={risk.min():.3e} max={risk.max():.3e} ptp={np.ptp(risk):.3e}")
    print(f"[keep] K={K}/{N} kept_idx[:5]={kept_idx[:5]}")

if __name__ == "__main__":
    main()
