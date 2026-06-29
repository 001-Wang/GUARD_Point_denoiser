
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize PointCVaR risk & predictions for a single ShapeNetPart file (.txt/.npy/.npz/.pt).
# Steps:
#   1) First forward to get per-point class.
#   2) Compute risk r_i = ||∂s_i/∂x_i|| (logit of predicted class).
#   3) Keep K lowest-risk points.
#   4) Second forward only on the kept K points for final predictions.

import os
import argparse
import importlib
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps as mcm
from pointnet2.test_all_v6 import _smart_load_state_dict as v6_load_state

# allow duplicate OpenMP (Windows) so plotting doesn't crash on mixed MKL builds
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---- eval helpers (v6) ----
import pointnet2.test_all_v6 as v6
from pointnet2.test_all_v6 import (
    run_model_logits,
    grad_risk_norm,
    build_valid_parts,
    predict_labels_with_gt,
)

# ---- dataset helpers ----
from pointnet2.data_utils.ShapeNetDataLoader_test import PartNormalDataset, pc_normalize

# ----------------- Utility: set seed for repeatability -----------------
import open3d as o3d

def show_interactive(xyz, color=None, pred=None, title="PointCVaR Visualization"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if color is not None:
        color = color / 255.0 if color.max() > 1 else color
        pcd.colors = o3d.utility.Vector3dVector(color)
    elif pred is not None:
        cmap = plt.get_cmap("tab20")
        color = np.array([cmap(int(l) % 20)[:3] for l in pred])
        pcd.colors = o3d.utility.Vector3dVector(color)
    else:
        pcd.paint_uniform_color([0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=960, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 3.0
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- I/O helpers -----------------
def load_any_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".csv"):
        arr = np.loadtxt(path).astype(np.float32)
        if arr.ndim == 1: arr = arr[None, :]
        return arr
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = np.array([np.asarray(x) for x in arr], dtype=np.float32)
        if arr.ndim == 1: arr = arr[None, :]
        return arr.astype(np.float32)
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        for k in ("xyz", "points", "data"):
            if k in data:
                arr = np.asarray(data[k])
                break
        else:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"{path} is empty npz")
            arr = np.asarray(data[keys[0]])
        if arr.ndim == 1: arr = arr[None, :]
        return arr.astype(np.float32)
    if ext in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ("xyz", "points", "data"):
                if k in obj:
                    obj = obj[k]
                    break
        if isinstance(obj, torch.Tensor):
            arr = obj.cpu().numpy()
        else:
            arr = np.asarray(obj)
        if arr.ndim == 1: arr = arr[None, :]
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported file extension: {ext}")


def normalize_and_optionally_resample(arr: np.ndarray, use_normals: bool, npoints: int | None):
    if arr.shape[1] < 3:
        raise ValueError("Input has fewer than 3 columns; need at least xyz.")
    has_normals = (arr.shape[1] >= 6) and use_normals
    if has_normals:
        xyz = pc_normalize(arr[:, :3].copy())
        nrm = arr[:, 3:6].copy()
        pts = np.concatenate([xyz, nrm], axis=1)
    else:
        pts = pc_normalize(arr[:, :3].copy())

    if npoints is None:
        return pts.astype(np.float32), has_normals

    N = pts.shape[0]
    M = int(npoints)
    if N >= M:
        choice = np.random.choice(N, M, replace=True)
    else:
        pad = np.random.choice(N, M - N, replace=True)
        choice = np.concatenate([np.arange(N), pad], 0)
    return pts[choice].astype(np.float32), has_normals


def dataset_meta_from_file(file_path: str):
    root = os.path.dirname(os.path.dirname(file_path))
    ds = PartNormalDataset(root=root, split="test", npoints=0, normal_channel=False)
    if getattr(v6, "SEG_CLASSES", None) is None and hasattr(ds, "seg_classes"):
        v6.SEG_CLASSES = ds.seg_classes
    if getattr(v6, "CLASS_CHOICE", None) is None and hasattr(ds, "classes"):
        v6.CLASS_CHOICE = [None] * len(ds.classes)
        for cat, idx in ds.classes.items():
            v6.CLASS_CHOICE[idx] = cat
    synset = os.path.basename(os.path.dirname(file_path))
    inv_cat = {v: k for k, v in ds.cat.items()}
    cat_name = inv_cat.get(synset, list(ds.cat.keys())[0])
    label_idx = ds.classes[cat_name]
    return ds, label_idx, cat_name


# ----------------- viz helpers -----------------
def to_heat_rgb(vals: np.ndarray):
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    denom = (vmax - vmin) if vmax > vmin else 1.0
    norm = (vals - vmin) / denom
    rgb = mcm.get_cmap("viridis")(norm)[:, :3]
    rgb = (rgb * 255.0).round().astype(np.uint8)
    return norm.astype(np.float32), rgb


def scatter_heat(ax, xyz, val, label="Risk (normalized)"):
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=val, s=0.6, cmap="viridis")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, label=label)


def scatter_pred(ax, xyz, lab, num_part=50, title="Kept K Predictions"):
    lab = np.maximum(lab, 0)
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(1, num_part - 1))
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=lab, s=1.0, cmap="tab20", norm=norm)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, label="Predicted Part ID")


# ----------------- model helpers -----------------
def smart_instantiate(model_path: str, num_part: int, normal_channel: bool):
    mod = importlib.import_module(model_path)
    if hasattr(mod, "get_model"):
        tried = []
        for sig in (
            lambda: mod.get_model(num_classes=num_part, normal_channel=normal_channel),
            lambda: mod.get_model(num_classes=num_part),
            lambda: mod.get_model(num_part),
        ):
            try:
                return sig()
            except Exception as e:
                tried.append(repr(e))
        raise RuntimeError("Failed to construct model via get_model: " + " | ".join(tried))
    for name in ("PointNet2", "DGCNN", "Model"):
        if hasattr(mod, name):
            return getattr(mod, name)()
    raise RuntimeError(f"Cannot instantiate model from {model_path}")


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", "").replace("model.", "").replace("net.", ""): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded checkpoint {path} (missing={len(missing)} unexpected={len(unexpected)})")


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("PointCVaR file visualizer (aligned with dataset/eval)")
    ap.add_argument("--file", type=str, required=True, help="data_prepare\\shapenet\\02691156\\xxx.txt")
    ap.add_argument("--outdir", type=str, default="viz_out")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--keep", type=int, default=2048)
    ap.add_argument("--npoints", type=int, default=None, help="None = use ALL points (v6 default). Else resample to N.")
    ap.add_argument("--use_normals_if_available", action="store_true", help="feed Nx6 if file includes normals")
    ap.add_argument("--num_part", type=int, default=50)
    ap.add_argument("--amp", action="store_true", help="use autocast fp16 for forwards")
    ap.add_argument("--seed", type=int, default=0, help="random seed for resampling/repro")
    ap.add_argument("--no_view", action="store_true", help="skip the Open3D interactive viewer")
    ap.add_argument("--category", type=str, default=None,
                    help="override category by NAME (e.g., Airplane, Chair, ...)")

    # model paths (PointCVaR baseline)
    ap.add_argument("--model_cvar", type=str, default="pointnet2.models.pointnet2_part_seg_msg")
    ap.add_argument("--ckpt_cvar", type=str, default="pointnet2/log/without_normal/checkpoints/best_model.pth")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset meta & label inference ----
    ds, inferred_label_idx, inferred_cat_name = dataset_meta_from_file(args.file)

    # Optional override via --category
    if args.category is not None:
        cand = args.category.strip()
        if cand in ds.classes:
            label_idx = ds.classes[cand]; cat_name = cand
        else:
            cand2 = cand.capitalize()
            if cand2 in ds.classes:
                label_idx = ds.classes[cand2]; cat_name = cand2
            else:
                raise ValueError(f"Unknown category '{args.category}'. Available: {list(ds.classes.keys())}")
        print(f"[INFO] Overriding category: {cat_name} (idx={label_idx})")
    else:
        label_idx, cat_name = inferred_label_idx, inferred_cat_name
        print(f"[INFO] Inferred category from path: {cat_name} (idx={label_idx})")

    # ---- Load & preprocess points like dataset/eval ----
    raw = load_any_points(args.file)
    has_normals_flag = (raw.shape[1] >= 6) and args.use_normals_if_available
    pts, has_normals = normalize_and_optionally_resample(raw, use_normals=has_normals_flag, npoints=args.npoints)

    if has_normals:
        xyz = pts[:, :3]
        feats = pts.T
        normal_channel = True
    else:
        xyz = pts[:, :3]
        feats = xyz.T
        normal_channel = False

    M = xyz.shape[0]
    base = os.path.basename(args.file)
    print(f"[INFO] Loaded {args.file} | M={M} | normals={has_normals} | use_all_points={args.npoints is None}")

    # ---- Build model & inputs ----
    model = smart_instantiate(args.model_cvar, args.num_part, normal_channel=normal_channel).to(device).eval()

    ckpt_obj = torch.load(args.ckpt_cvar, map_location=device,weights_only=False)
    v6_load_state(ckpt_obj, model)  # prints: [INFO] ckpt loaded (strict). missing=0 unexpected=0


    xyz_bcn = torch.from_numpy(feats).unsqueeze(0).float().to(device)    # (1,C,M)
    label_B = torch.tensor([label_idx], dtype=torch.long, device=device) # (1,)
    amp_dtype = torch.float16 if args.amp else None

    # ---- First forward (no grad) ----
    with torch.no_grad():
        logits_bnc, _ = run_model_logits(
            model, xyz_bcn, label_B,
            amp_enabled=args.amp, amp_dtype=amp_dtype,
            num_part=args.num_part, prefer_sngp=False,
            forward_kwargs=None,
        )

    # ---- Risk (with grad) ----
    xyz_req = xyz_bcn.detach().clone().requires_grad_(True)
    risk_bn = grad_risk_norm(
        model, xyz_req, label_B,
        amp_dtype=amp_dtype,
        mask_allowed=True,
        num_part=args.num_part, prefer_sngp=False
    )  # (1, M)
    risk = risk_bn[0].detach().cpu().numpy()

    # ---- Keep K lowest-risk ----
    K = min(int(args.keep), M)
    idx_kept = np.argsort(risk)[:K]
    xyz_kept = xyz[idx_kept]

    # ---- Second forward on kept K (no grad) ----
    with torch.no_grad():
        def _gather_bn3_to_bn(x_b3n, idx_bk):
            B, C, N = x_b3n.shape
            idx_exp = idx_bk.unsqueeze(1).expand(-1, C, -1)
            return torch.gather(x_b3n, dim=2, index=idx_exp)
        xyz_kept_b3k = _gather_bn3_to_bn(xyz_bcn, torch.from_numpy(idx_kept).to(device).view(1, -1))
        logits_kept_bnc, _ = run_model_logits(
            model, xyz_kept_b3k, label_B,
            amp_enabled=args.amp, amp_dtype=amp_dtype,
            num_part=args.num_part, prefer_sngp=False,
        )

    # Predictions (mask-aware if possible)
    valid_parts_bc = build_valid_parts(label_B)
    if valid_parts_bc is None:
        pred_full = logits_bnc.argmax(dim=-1)[0].detach().cpu().numpy()
        pred_kept = logits_kept_bnc.argmax(dim=-1)[0].detach().cpu().numpy()
    else:
        pred_full_t = predict_labels_with_gt(
            logits_bnc, label_B,
            gt_n=torch.full((M,), -1, dtype=torch.long, device=device),
            mask_allowed=True, num_part=args.num_part
        )[0]
        pred_full = pred_full_t.detach().cpu().numpy()
        pred_kept_t = predict_labels_with_gt(
            logits_kept_bnc, label_B,
            gt_n=torch.full((K,), -1, dtype=torch.long, device=device),
            mask_allowed=True, num_part=args.num_part
        )[0]
        pred_kept = pred_kept_t.detach().cpu().numpy()

    # # ---- Save TXT ----
    # norm_r, rgb = to_heat_rgb(risk)
    # risk_table = np.concatenate(
    #     [xyz, risk.reshape(-1, 1).astype(np.float32), norm_r.reshape(-1, 1), rgb.astype(np.float32)],
    #     axis=1
    # )
    # f_risk_txt = os.path.join(args.outdir, f"{base}_risk.txt")
    # np.savetxt(
    #     f_risk_txt, risk_table,
    #     fmt="%.6f %.6f %.6f %.6f %.6f %.0f %.0f %.0f",
    #     header="x y z risk norm_r R G B",
    #     comments=""
    # )

    # kept_table = np.concatenate([xyz_kept, pred_kept.reshape(-1, 1).astype(np.float32)], axis=1)
    # f_kept_txt = os.path.join(args.outdir, f"{base}_kept{K}_pred.txt")
    # np.savetxt(
    #     f_kept_txt, kept_table,
    #     fmt="%.6f %.6f %.6f %.0f",
    #     header="x y z pred",
    #     comments=""
    # )
    # print(f"[OK] TXT saved:\\n  {f_risk_txt}\\n  {f_kept_txt}")

    # # ---- Save PNGs ----
    # fig = plt.figure(figsize=(7, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # scatter_heat(ax, xyz, norm_r, label="PointCVaR Risk (normalized)")
    # ax.set_title("PointCVaR Risk Heatmap")
    # plt.tight_layout()
    # f_risk_png = os.path.join(args.outdir, f"{base}_risk.png")
    # plt.savefig(f_risk_png, dpi=300)
    # plt.close(fig)

    # fig = plt.figure(figsize=(7, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # scatter_pred(ax, xyz_kept, pred_kept, num_part=args.num_part, title=f"Kept {K} Points Prediction (PointCVaR)")
    # plt.tight_layout()
    # f_kept_png = os.path.join(args.outdir, f"{base}_kept{K}_pred_cvar.png")
    # plt.savefig(f_kept_png, dpi=300)
    # plt.close(fig)

    # # ---- NPZ bundle ----
    # f_npz = os.path.join(args.outdir, f"{base}_vizdata_cvar.npz")
    # np.savez_compressed(f_npz, xyz=xyz, risk=risk, pred_full=pred_full, idx_kept=idx_kept)
    # print(f"[OK] Saved:\\n  {f_risk_png}\\n  {f_kept_png}\\n  {f_npz}")
    # print("[INFO] Opening interactive viewer... (press 'Q' or ESC to close)")

    # Show interactively (kept predictions)
    if not args.no_view:
        show_interactive(xyz_kept, pred=pred_kept, title="PointCVaR: Kept K Predictions")


if __name__ == "__main__":
    main()
