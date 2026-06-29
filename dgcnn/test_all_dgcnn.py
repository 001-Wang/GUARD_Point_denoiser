#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified noisy-eval script (same outputs/logic as test_all_v5) but using **DGCNN** as the
segmentation backbone for all methods. For **SNGP**, we still use the PN-front
(geo_transformer + SNGP head) ONLY to compute uncertainty (σ²) and pick points,
then run DGCNN on the selected points.

Methods:
  - pointcvar : gradient-risk (PointCVaR). Keep K lowest-risk points, then second forward on kept K (DGCNN).
  - entropy   : predictive entropy on DGCNN logits; keep K lowest-entropy points (single forward).
  - sngp      : PN-front for σ² selection (keep K lowest σ²), then DGCNN forward on kept K.

Notes
- Batch size is forced to 1 (variable N per shape).
- Labels use -1 to mark noisy points.
- Dataset is PartNormalDataset from dgcnn/data_utils/ShapeNetDataLoader_test.py.
- Output format matches test_all_v5 (per-sample logs + summary: latency, noise, mIoU stats).

Default paths:
  * entropy/pointcvar backbone:  dgcnn.model  +  dgcnn/pretrained/model.partseg.t7
  * sngp PN-front:               dgcnn.models.sngp_s2_6layers  +  dgcnn/log/sngp/checkpoints/best_model.ckpt
  * clean precision P:           dgcnn/log/sngp/checkpoints/P_clean.pt
"""

from __future__ import annotations
import os
import time
import math
import importlib
import argparse
from statistics import mean, median, pstdev
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------
# Utilities
# ------------------------------

def autocast_cuda(enabled: bool, dtype: Optional[torch.dtype]):
    if not enabled:
        class Dummy:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        return Dummy()
    return torch.autocast(device_type='cuda', dtype=dtype if dtype is not None else torch.float16)


def _one_hot_labels(label: torch.Tensor, num_classes: int = 16) -> torch.Tensor:
    label = torch.as_tensor(label).to(torch.int64).view(-1)
    b = label.shape[0]
    oh = torch.zeros(b, num_classes, device=label.device, dtype=torch.float32)
    oh.scatter_(1, label.view(-1,1), 1.0)
    return oh

# Global seg maps (filled from dataset)
SEG_CLASSES: Optional[Dict[str, list]] = None
CLASS_CHOICE: Optional[list] = None

# ------------------------------
# Robust model forward normalizer (for DGCNN backbone)
# ------------------------------

def run_model_logits(model: nn.Module,
                     xyz_bcn: torch.Tensor,
                     label_B: torch.Tensor,
                     amp_enabled: bool,
                     amp_dtype: Optional[torch.dtype],
                     num_part: int = 50) -> torch.Tensor:
    """Return logits as (B, N, C) for DGCNN.
    The DGCNN forward signature is (xyz_bcn, one_hot[B,16,1]).
    """
    B, C, N = xyz_bcn.shape
    oh = _one_hot_labels(label_B, num_classes=16).unsqueeze(-1)  # (B,16,1)
    with autocast_cuda(amp_enabled, amp_dtype):
        out = model(xyz_bcn, oh)
    # Normalize to (B,N,C)
    if out.dim() != 3:
        raise RuntimeError(f"Model did not return 3D logits: got {tuple(out.shape)}")
    if out.shape[1] == num_part and out.shape[2] == N:
        logits_bnc = out.transpose(1,2).contiguous()
    elif out.shape[1] == N and out.shape[2] == num_part:
        logits_bnc = out
    else:
        if out.shape[-1] == num_part:
            logits_bnc = out
        elif out.shape[1] == num_part:
            logits_bnc = out.transpose(1,2).contiguous()
        else:
            raise RuntimeError(f"Unexpected logits shape {tuple(out.shape)}")
    return logits_bnc

# ------------------------------
# PN-front (for SNGP selection only)
# ------------------------------

def build_pnfront_and_load(pn_mod_path: str, ckpt_path: str, p_path: str, device: torch.device):
    mod = importlib.import_module(pn_mod_path)
    if not hasattr(mod, 'get_model'):
        raise RuntimeError('PN-front module must expose get_model(num_classes=50)')
    pn2 = mod.get_model(num_classes=50).eval().to(device)

    # Load front weights (geom_tf + early_rffgp)
    obj = torch.load(ckpt_path, map_location='cpu')
    if isinstance(obj, dict) and 'model_state_dict' in obj and isinstance(obj['model_state_dict'], dict):
        sd_raw = obj['model_state_dict']
    elif isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        sd_raw = obj['state_dict']
    elif isinstance(obj, dict):
        sd_raw = obj
    else:
        raise RuntimeError('Unrecognized PN2 checkpoint format')
    def _strip(k):
        return k.replace('module.', '') if k.startswith('module.') else k
    sd_front = { _strip(k): v for k,v in sd_raw.items()
                 if _strip(k).startswith('geom_tf') or _strip(k).startswith('early_rffgp') }
    pn2.load_state_dict(sd_front, strict=False)

    # Load precision P and make it safe
    Pobj = torch.load(p_path, map_location='cpu')
    P = Pobj['precision'] if isinstance(Pobj, dict) and 'precision' in Pobj else Pobj
    P = torch.as_tensor(P, dtype=torch.float32)
    P = 0.5*(P+P.T) + 5.0*torch.eye(P.shape[0], dtype=P.dtype)  # explicit sym + ridge
    pn2.early_rffgp.load_precision(P)
    return pn2

# ------------------------------
# Entropy / risk / PN σ² selection
# ------------------------------

def extract_entropy(logits_bnc: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(logits_bnc, dim=-1)
    ent = -(prob * (prob.clamp_min(1e-9).log())).sum(dim=-1)  # (B,N)
    return ent

def grad_risk_norm(model: nn.Module,
                   xyz_bcn_req: torch.Tensor,
                   label_B: torch.Tensor,
                   amp_dtype: Optional[torch.dtype],
                   mask_allowed: bool,
                   num_part: int = 50) -> torch.Tensor:
    """Per-point gradient risk ‖∂s/∂x‖, where s is logit of predicted class per point."""
    assert xyz_bcn_req.requires_grad
    with torch.no_grad():
        logits_bnc = run_model_logits(model, xyz_bcn_req, label_B, amp_enabled=False, amp_dtype=None, num_part=num_part)
        pred = logits_bnc.argmax(dim=-1)  # (B,N)
    logits2_bnc = run_model_logits(model, xyz_bcn_req, label_B, amp_enabled=False, amp_dtype=None, num_part=num_part)
    s = logits2_bnc.gather(dim=2, index=pred.unsqueeze(-1)).squeeze(-1)  # (B,N)
    grad = torch.autograd.grad(s.sum(), xyz_bcn_req, retain_graph=False, create_graph=False)[0]  # (B,3,N)
    return grad.norm(dim=1)  # (B,N)


def pnfront_sigma2(pn2, xyz_bcn: torch.Tensor) -> torch.Tensor:
    x_bnc = xyz_bcn.transpose(1,2).contiguous()  # (B,N,3)
    h_bnc = pn2.geom_tf(x_bnc)                   # (B,N,Dg)
    out = pn2.early_rffgp(h_bnc, update_precision=False, compute_var=True)
    if isinstance(out, tuple) and len(out) == 2:
        _, sigma2 = out
    else:
        sigma2 = out
    return sigma2  # (B,N)

# ------------------------------
# Predictions + IoU helpers (same as v5)
# ------------------------------

def get_cat_from_gt(gt_n: torch.Tensor) -> Optional[str]:
    # Build seg_label_to_cat once per process
    if not hasattr(get_cat_from_gt, "_map"):
        m = {}
        for cat, parts in SEG_CLASSES.items():
            for l in parts:
                m[l] = cat
        get_cat_from_gt._map = m
    idx = (gt_n != -1).nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return None
    lab = int(gt_n[idx[0]].item())
    return get_cat_from_gt._map.get(lab, None)


def predict_labels_with_gt(logits_bnc, label_B, gt_n, mask_allowed: bool, num_part=50):
    B, N, C = logits_bnc.shape
    if not (mask_allowed and (SEG_CLASSES is not None)):
        return logits_bnc.argmax(dim=-1)

    pred = torch.empty((B, N), dtype=torch.long, device=logits_bnc.device)
    seg_label_to_cat = {}
    for cat, parts in SEG_CLASSES.items():
        for l in parts:
            seg_label_to_cat[l] = cat

    for i in range(B):
        gt_i = gt_n
        idx = (gt_i != -1).nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            first_lab = int(gt_i[idx[0]].item())
            cat = seg_label_to_cat.get(first_lab, None)
        else:
            cat = None
        if cat is None:
            cat = CLASS_CHOICE[int(label_B[i].item())]
        allowed = torch.tensor(SEG_CLASSES[cat], device=logits_bnc.device, dtype=torch.long)
        idx_local = torch.argmax(logits_bnc[i, :, allowed], dim=-1)
        pred[i] = allowed[idx_local]
    return pred

# ------------------------------
# Collate & gather helpers
# ------------------------------
def pn2_forward_get_sigma2(pn2, xyz_bcn: torch.Tensor, label_B: torch.Tensor) -> torch.Tensor:
    """
    Robustly extract σ² from a PN-front model.
    1) Try full forward() like test_all_v5 (tuple with aux['sigma'|'var'|'log_var'] or second output)
    2) If not found, FALL BACK to manual path: geom_tf -> early_rffgp(..., compute_var=True)
    Returns (B,N) tensor.
    """
    B, C, N = xyz_bcn.shape
    # one-hot (B,16,1); if PN2 doesn't need it, it will ignore safely
    oh = torch.zeros(B, 16, device=xyz_bcn.device, dtype=torch.float32)
    oh.scatter_(1, label_B.to(torch.int64).view(-1, 1), 1.0)
    oh = oh.unsqueeze(-1)

    sigma2 = None
    # ---- Attempt 1: full forward (v5-style) ----
    try:
        out = pn2(xyz_bcn, oh)  # prefer identical call to v5; harmless if PN2 ignores 'oh'
        if isinstance(out, tuple):
            if len(out) == 3:
                base_logits, sngp_logits, aux = out
                # prefer aux dict
                if isinstance(aux, dict):
                    for key in ('sigma', 'var', 'log_var'):
                        if key in aux:
                            sigma2 = aux[key]
                            break
                if sigma2 is None:
                    # many implementations put var as 2nd tensor
                    if isinstance(sngp_logits, torch.Tensor):
                        sigma2 = sngp_logits
            elif len(out) == 2:
                logits, aux_or_var = out
                if isinstance(aux_or_var, dict):
                    for key in ('sigma', 'var', 'log_var'):
                        if key in aux_or_var:
                            sigma2 = aux_or_var[key]
                            break
                elif isinstance(aux_or_var, torch.Tensor):
                    sigma2 = aux_or_var
        # normalize shape if we found it
        if isinstance(sigma2, torch.Tensor):
            if sigma2.dim() == 3 and sigma2.shape[-1] == 1:
                sigma2 = sigma2.squeeze(-1)
            elif sigma2.dim() == 3 and sigma2.shape[1] == 1:
                sigma2 = sigma2.squeeze(1)
            elif sigma2.dim() != 2:
                sigma2 = sigma2[..., 0]
    except Exception:
        sigma2 = None  # force fallback

    # ---- Attempt 2: manual path (geom_tf -> early_rffgp) ----
    if not isinstance(sigma2, torch.Tensor):
        x_bnc = xyz_bcn.transpose(1, 2).contiguous()     # (B,N,3)
        h_bnc = pn2.geom_tf(x_bnc)                       # (B,N,Dg)
        out2 = pn2.early_rffgp(h_bnc, update_precision=False, compute_var=True)
        if isinstance(out2, tuple) and len(out2) == 2:
            _, sigma2 = out2
        else:
            sigma2 = out2
        # shape to (B,N)
        if sigma2.dim() == 3 and sigma2.shape[-1] == 1:
            sigma2 = sigma2.squeeze(-1)
        elif sigma2.dim() == 3 and sigma2.shape[1] == 1:
            sigma2 = sigma2.squeeze(1)
        elif sigma2.dim() != 2:
            sigma2 = sigma2[..., 0]

    # final guard
    if not isinstance(sigma2, torch.Tensor) or sigma2.dim() != 2 or sigma2.shape[1] != N:
        raise RuntimeError(f"Could not obtain sigma2 as (B,N). Got {None if sigma2 is None else tuple(sigma2.shape)}")
    return sigma2



def safe_collate_varlen(batch):
    assert len(batch) == 1
    pts, lbl, tgt = batch[0]
    if not isinstance(pts, torch.Tensor):
        pts = torch.tensor(pts)
    if not isinstance(tgt, torch.Tensor):
        tgt = torch.tensor(tgt, dtype=torch.long)
    else:
        tgt = tgt.long()
    if isinstance(lbl, torch.Tensor):
        lbl = lbl.view(1).long()
    else:
        lbl = torch.tensor([int(lbl)], dtype=torch.long)
    return pts, lbl, tgt


def gather_bn3_to_bn(x_b3n: torch.Tensor, idx_bk: torch.Tensor) -> torch.Tensor:
    B, C, N = x_b3n.shape
    idx_exp = idx_bk.unsqueeze(1).expand(-1, C, -1)
    return torch.gather(x_b3n, dim=2, index=idx_exp)

# ------------------------------
# Smart checkpoint loader for DGCNN
# ------------------------------

def clean_and_load_dgcnn_state(dgcnn: nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd.get('state_dict', sd)
    sd_clean = {}
    for k, v in sd.items():
        k2 = k.replace('module.', '')
        if any(s in k2 for s in ('weight_orig','weight_u','weight_v')):
            continue
        sd_clean[k2] = v
    missing, unexpected = dgcnn.load_state_dict(sd_clean, strict=False)
    print(f"[DGCNN] loaded. missing={missing} | unexpected={unexpected}")

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser('Noisy eval (DGCNN backbone) : pointcvar / entropy / sngp')
    parser.add_argument('--method', type=str, choices=['pointcvar','entropy','sngp'], required=True)
    parser.add_argument('--root', type=str, default=r'..\pointnet2\data\shapenet_c_add\add_local_s5', help='..\pointnet2\data\shapenet_c_add\add_local_s5')

    # Backbone & PN-front
    parser.add_argument('--dgcnn_module', type=str, default=r'dgcnn.model')
    parser.add_argument('--dgcnn_ckpt', type=str, default=r'dgcnn/pretrained/model.partseg.t7')
    parser.add_argument('--pnfront_module', type=str, default=r'../pointnet2/model/sngp_s2_6layers.py')
    parser.add_argument('--pnfront_ckpt', type=str, default=r'../pointnet2/log/sngp/checkpoints/best_model.ckpt')
    parser.add_argument('--precision_path', type=str, default=r'../pointnet2/log/sngp/checkpoints/P_clean.pt')

    parser.add_argument('--num_point', type=int, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--keep', type=int, default=2048)
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--mask_allowed', action='store_true', default=True)
    parser.add_argument('--num_part', type=int, default=50)
    parser.add_argument('--time_include_post', action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.float16 if args.amp else None

    # Data
    from pointnet2.data_utils.ShapeNetDataLoader_test import PartNormalDataset
    dataset = PartNormalDataset(root=args.root, split='test', normal_channel=args.normal, npoints=args.num_point)

    global SEG_CLASSES, CLASS_CHOICE
    if hasattr(dataset, 'seg_classes'):
        SEG_CLASSES = dataset.seg_classes
    if hasattr(dataset, 'classes'):
        CLASS_CHOICE = [None] * len(dataset.classes)
        for cat, idx in dataset.classes.items():
            CLASS_CHOICE[idx] = cat
    print(f"[INFO] mask_allowed={args.mask_allowed} | SEG_CLASSES loaded={SEG_CLASSES is not None} | CLASS_CHOICE loaded={CLASS_CHOICE is not None}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=safe_collate_varlen, pin_memory=True)

    # Build DGCNN backbone
    dg_mod = importlib.import_module(args.dgcnn_module)
    if hasattr(dg_mod, 'get_model'):
        try:
            dgcnn = dg_mod.get_model(num_classes=args.num_part, normal_channel=args.normal)
        except Exception:
            dgcnn = dg_mod.get_model(args.num_part)
    elif hasattr(dg_mod, 'DGCNN_partseg'):
        from types import SimpleNamespace
        dg_args = SimpleNamespace(k=20, emb_dims=1024, dropout=0.5)
        dgcnn = getattr(dg_mod, 'DGCNN_partseg')(dg_args, seg_num_all=args.num_part)
    else:
        raise RuntimeError('Could not instantiate DGCNN backbone: need get_model(...) or DGCNN_partseg class')

    dgcnn = dgcnn.to(device).eval()
    clean_and_load_dgcnn_state(dgcnn, args.dgcnn_ckpt)

    # Optional PN-front (for SNGP selection)
    pn2 = None
    if args.method == 'sngp':
        pn2 = build_pnfront_and_load(args.pnfront_module, args.pnfront_ckpt, args.precision_path, device)
        # quick RFF stats
        if hasattr(pn2.early_rffgp, 'W'):
            print('[RFF] W mean/std:', pn2.early_rffgp.W.mean().item(), pn2.early_rffgp.W.std().item())

    # Accumulators (match test_all_v5 output style)
    lat_ms = []
    remain_noise_counts = []
    kept_counts = []
    inst_mious = []
    per_cat_shape_ious = {cat: [] for cat in SEG_CLASSES.keys()}

    for it, batch in enumerate(tqdm(loader, total=len(dataset), desc=f"{args.method} eval", ncols=80)):
        pts_nc, label_B, target_n = batch
        pts = pts_nc.float()
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)
        B, N, C = pts.shape
        assert B == 1
        xyz_bcn = pts[:, :, :3].transpose(2,1).contiguous().to(device)  # (1,3,N)
        label_B = label_B.to(device)
        gt_n = target_n.view(-1).to(device)

        t0 = time.perf_counter()

        # First forward (DGCNN) to get logits for decoding/metrics
        with torch.no_grad():
            logits_full_bnc = run_model_logits(dgcnn, xyz_bcn, label_B, amp_enabled=args.amp, amp_dtype=amp_dtype, num_part=args.num_part)
        t_cut = time.perf_counter()

        # Selection
        if args.method == 'pointcvar':
            K = min(args.keep, N)
            if K == N:
                idx_sorted = torch.arange(N, device=xyz_bcn.device)
                kept_counts.append(int(K))
                noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
                remain_noise_counts.append(noise_kept)
                pred_kept = predict_labels_with_gt(logits_full_bnc, label_B, gt_n, mask_allowed=args.mask_allowed, num_part=args.num_part)[0]
                t1 = t_cut
            else:
                xyz_req = xyz_bcn.detach().clone().requires_grad_(True)
                risk_bn = grad_risk_norm(dgcnn, xyz_req, label_B, amp_dtype, mask_allowed=args.mask_allowed, num_part=args.num_part)
                idx_sorted = torch.argsort(risk_bn[0], descending=False)[:K]
                kept_counts.append(int(K))
                noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
                remain_noise_counts.append(noise_kept)
                with torch.no_grad():
                    xyz_kept_b3k = gather_bn3_to_bn(xyz_bcn, idx_sorted.view(1, -1))
                    logits_kept_bnc = run_model_logits(dgcnn, xyz_kept_b3k, label_B, amp_enabled=args.amp, amp_dtype=amp_dtype, num_part=args.num_part)
                pred_kept = predict_labels_with_gt(logits_kept_bnc, label_B, gt_n[idx_sorted], mask_allowed=args.mask_allowed, num_part=args.num_part)[0]
                t1 = time.perf_counter() if args.time_include_post else t_cut

        elif args.method == 'entropy':
            ent = extract_entropy(logits_full_bnc)
            K = min(args.keep, ent.shape[1])
            idx_sorted = torch.argsort(ent[0], descending=False)[:K]
            kept_counts.append(int(K))
            noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
            remain_noise_counts.append(noise_kept)
            pred_full = predict_labels_with_gt(logits_full_bnc, label_B, gt_n, mask_allowed=args.mask_allowed, num_part=args.num_part)[0]
            pred_kept = pred_full[idx_sorted]
            t1 = time.perf_counter() if args.time_include_post else t_cut

        else:  # sngp -> use PN-front for σ² selection
            # ---------- SNGP selection (fixed: PN2-first, then single DGCNN on kept) ----------
            assert pn2 is not None, "PN-front must be initialized for --method sngp"

            # 1) PN2 FRONT forward to get σ² on all N
            with torch.no_grad():
                sigma2 = pn2_forward_get_sigma2(pn2, xyz_bcn, label_B)  # (B,N), matches your helper

            # 2) Select K lowest σ² (ascending) – identical to v5
            K = min(args.keep, sigma2.shape[1])
            idx_sorted = torch.argsort(sigma2[0], descending=False)[:K]  # (K,)

            # 3) Count noise in kept indices (GT sliced by kept)
            noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
            remain_noise_counts.append(noise_kept)
            kept_counts.append(int(K))

            # 4) Single DGCNN forward on KEPT subset
            with torch.no_grad():
                xyz_kept_b3k = gather_bn3_to_bn(xyz_bcn, idx_sorted.view(1, -1))  # (1,3,K)
                logits_kept_bnc = run_model_logits(
                    dgcnn, xyz_kept_b3k, label_B, amp_enabled=args.amp, amp_dtype=amp_dtype, num_part=args.num_part
                )

            # 5) Category-masked decoding ON KEPT SET
            pred_kept = predict_labels_with_gt(
                logits_kept_bnc, label_B, gt_n[idx_sorted], mask_allowed=args.mask_allowed, num_part=args.num_part
            )[0]

            # timing (consistent with your --time_include_post flag)
            t1 = time.perf_counter() if args.time_include_post else t_cut



        # IoU on kept indices (noise counted wrong)
        gt_kept = gt_n[idx_sorted]
        cat_i = get_cat_from_gt(gt_kept)
        if cat_i is None:
            inst_mious.append(0.0)
        else:
            parts = SEG_CLASSES[cat_i]
            ious = []
            for l in parts:
                tp = ((gt_kept == l) & (pred_kept == l)).sum().item()
                fp = ((gt_kept != l) & (pred_kept == l)).sum().item()
                fn = ((gt_kept == l) & (pred_kept != l)).sum().item()
                denom = tp + fp + fn
                ious.append(1.0 if denom == 0 else (tp / float(denom)))
            shape_iou = sum(ious) / len(ious)
            inst_mious.append(shape_iou)
            per_cat_shape_ious[cat_i].append(shape_iou)

        # Latency + per-sample log (same format)
        lat_ms.append((t1 - t0) * 1000.0)
        print(f"[SAMPLE {it:05d}] N={N} | kept={K} | remain_noise_in_kept={noise_kept} | latency_ms={lat_ms[-1]:.2f}")

    # Summary (same as v5)
    def percentile(xs, p):
        if not xs: return 0.0
        xs_sorted = sorted(xs)
        k = (len(xs_sorted)-1) * (p/100.0)
        f = math.floor(k); c = math.ceil(k)
        if f == c: return xs_sorted[int(k)]
        return xs_sorted[f] * (c-k) + xs_sorted[c] * (k-f)

    n = len(lat_ms)
    lat_mean = mean(lat_ms) if n else 0.0
    lat_p50 = percentile(lat_ms, 50)
    lat_p90 = percentile(lat_ms, 90)
    lat_p95 = percentile(lat_ms, 95)
    lat_std = pstdev(lat_ms) if n > 1 else 0.0

    avg_noise_kept = (sum(remain_noise_counts)/len(remain_noise_counts)) if remain_noise_counts else 0.0

    print("[SUMMARY]")
    print(f"Samples evaluated: {n}")
    print(f"Latency ms per-sample | mean={lat_mean:.2f} | p50={lat_p50:.2f} | p90={lat_p90:.2f} | p95={lat_p95:.2f} | std={lat_std:.2f}")
    print(f"Avg noise count in kept-{args.keep}: {avg_noise_kept:.2f}")
    inst_miou_kept = sum(inst_mious)/len(inst_mious) if inst_mious else 0.0
    cat_means = [sum(v)/len(v) for v in per_cat_shape_ious.values() if len(v)]
    class_miou_kept_dataset_avg = (sum(cat_means)/len(cat_means)) if cat_means else 0.0

    print(f"Instance mIoU (kept-only, official-style): {inst_miou_kept:.4f}")
    print(f"Class mIoU (kept-only, dataset-avg): {class_miou_kept_dataset_avg:.4f}")


if __name__ == '__main__':
    main()
