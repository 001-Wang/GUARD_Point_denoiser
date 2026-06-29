#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified noisy-eval script for three methods:
  - pointcvar : gradient-risk (PointCVaR). Keep K lowest-risk points, then second forward on the kept K.
  - entropy   : predictive-entropy baseline on PointNet++ logits (no second forward).
  - sngp      : your SNGP model. Use SNGP head logits for selection (no second forward by default).

Requirements / assumptions
- Batch size is forced to 1 because point counts per shape vary.
- We ignore `num_point`; we use *all* points provided by the dataset.
- Targets use -1 to mark noisy/outlier points.
- Dataset is fixed to PartNormalDataset from data_utils.ShapeNetDataLoader_test (as requested).
- Model/ckpt defaults (can be overridden by flags):
    * pointcvar / entropy → models.pointnet2_part_seg_msg + log/part_seg/without_normal/checkpoints/best_model.pth
    * sngp                → models.sngp_s2_6layers       + log/part_seg/sngp/checkpoints/best_model.ckpt

Metrics collected
- Latency per-sample (ms): mean/p50/p90/p95/std
- For each sample, number of noise points in the kept K (default 2048)
- mIoU on the kept points, with noise (gt==-1) **always counted wrong** (into FP of predicted class)

Important no-grad policy
- We do *not* wrap the entire loop in no_grad so PointCVaR can compute gradients.
- We *do* wrap individual forwards that do not need gradients (first forward, entropy, SNGP, and CVaR second forward) in `torch.no_grad()` scopes.
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
    b = label.shape[0]
    oh = torch.zeros(b, num_classes, device=label.device)
    oh.scatter_(1, label.long().view(-1,1), 1.0)
    return oh


# Placeholders; if your env defines SEG_CLASSES / CLASS_CHOICE they will be imported.
SEG_CLASSES: Optional[Dict[str, list]] = None
CLASS_CHOICE: Optional[list] = None


def try_import_seg_maps():
    global SEG_CLASSES, CLASS_CHOICE
    for modname in [
        'partnet_seg_map',
        'data_utils.part_seg_utils',
        'utils.part_seg_utils',
        'provider',
    ]:
        try:
            m = importlib.import_module(modname)
            if hasattr(m, 'SEG_CLASSES') and hasattr(m, 'CLASS_CHOICE'):
                SEG_CLASSES = getattr(m, 'SEG_CLASSES')
                CLASS_CHOICE = list(getattr(m, 'CLASS_CHOICE'))
                return
        except Exception:
            pass
    # Fallback: leave as None, we will skip masking


# ------------------------------
# Robust model forward normalizer
# ------------------------------



def run_model_logits(model: nn.Module,
                     xyz_bcn: torch.Tensor,
                     label_B: torch.Tensor,
                     amp_enabled: bool,
                     amp_dtype: Optional[torch.dtype],
                     num_part: int = 50,
                     prefer_sngp: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns
      logits_bnc: (B, N, num_part)
      aux       : optional dict; we extract uncertainty from keys like 'sigma', 'var', 'log_var'

    Behavior
      - For SNGP models that return (base_logits, sngp_logits, aux), we use `sngp_logits` if `prefer_sngp=True`, else baseline.
      - For PointNet++ models that return (logits, l3_points), we use logits.
    """
    B, C, N = xyz_bcn.shape
    device = xyz_bcn.device
    oh = _one_hot_labels(label_B, num_classes=16)

    with autocast_cuda(amp_enabled, amp_dtype):
        # The two provided repos accept (xyz, cls_label)
        out = model(xyz_bcn, oh)

    aux: Dict[str, torch.Tensor] = {}
    # SNGP tuple (baseline, sngp, aux)
    if isinstance(out, tuple) and len(out) == 3:
        base_logits, sngp_logits, aux = out
        logits = sngp_logits if (prefer_sngp and sngp_logits is not None) else base_logits
    # PointNet++ tuple (logits, l3_points)
    elif isinstance(out, tuple) and len(out) == 2:
        logits, _ = out
    else:
        logits = out  # just logits

    if logits.dim() != 3:
        raise RuntimeError(f"Model did not return 3D logits: got {tuple(logits.shape)}")

    # Normalize to (B, N, C)
    if logits.shape[1] == num_part and logits.shape[2] == N:      # (B, C, N)
        logits_bnc = logits.transpose(1, 2).contiguous()
    elif logits.shape[1] == N and logits.shape[2] == num_part:    # (B, N, C)
        logits_bnc = logits
    else:
        if logits.shape[-1] == num_part:
            logits_bnc = logits
        elif logits.shape[1] == num_part:
            logits_bnc = logits.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")

    if not isinstance(aux, dict):
        aux = {}
    return logits_bnc, aux


# ------------------------------
# Uncertainty extraction for SNGP or fallback
# ------------------------------

def extract_uncertainty(aux: Dict[str, torch.Tensor], logits_bnc: torch.Tensor) -> torch.Tensor:
    """Return (B, N) uncertainty score. Prefer sigma/var from aux; else predictive entropy."""
    B, N, C = logits_bnc.shape
    for key in ['sigma', 'var', 'variance', 'log_var', 'logvar']:
        v = aux.get(key, None)
        if v is None:
            continue
        t = v
        if t.dim() == 3 and t.shape[:2] == (B, N):
            t = t.mean(dim=-1)
        elif t.dim() == 2 and t.shape == (B, N):
            pass
        elif t.dim() == 1 and t.numel() == B*N:
            t = t.view(B, N)
        else:
            continue
        if 'log' in key:
            t = t.exp()
        return t

    # Fallback: predictive entropy from softmax
    prob = F.softmax(logits_bnc, dim=-1)
    eps = 1e-9
    ent = -(prob * (prob.clamp_min(eps).log())).sum(dim=-1)  # (B,N)
    return ent


# ------------------------------
# Predictions + IoU helpers
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
    # Build seg_label_to_cat once from SEG_CLASSES
    seg_label_to_cat = {}
    for cat, parts in SEG_CLASSES.items():
        for l in parts:
            seg_label_to_cat[l] = cat

    for i in range(B):
        # Robust cat from GT (like baseline)
        gt_i = gt_n  # shape (N,) in your loop
        # find first non -1 label
        idx = (gt_i != -1).nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            first_lab = int(gt_i[idx[0]].item())
            cat = seg_label_to_cat.get(first_lab, None)
        else:
            cat = None

        if cat is None:
            # fallback to label_B if GT is all -1 (shouldn't happen on clean set)
            cat = CLASS_CHOICE[int(label_B[i].item())]

        allowed = torch.tensor(SEG_CLASSES[cat], device=logits_bnc.device, dtype=torch.long)
        idx_local = torch.argmax(logits_bnc[i, :, allowed], dim=-1)
        pred[i] = allowed[idx_local]
    return pred



def accumulate_iou_counts(pred: torch.Tensor, gt: torch.Tensor, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, num_part: int = 50):
    """Update TP/FP/FN for IoU. Noise (gt==-1) is counted as WRONG and contributes to FP of the predicted class."""
    for c in range(num_part):
        pc = (pred == c)
        gc = (gt == c)
        tp[c] += (pc & gc).sum().item()
        fp[c] += (pc & (~gc)).sum().item()  # includes gt==-1 as wrong
        fn[c] += ((~pc) & gc).sum().item()


def finalize_class_miou(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> float:
    ious = []
    for c in range(tp.numel()):
        denom = tp[c] + fp[c] + fn[c]
        if denom > 0:
            ious.append(tp[c] / denom)
    return float(sum(ious) / max(len(ious), 1))


# ------------------------------
# Risk via gradient norm (PointCVaR)
# ------------------------------

def grad_risk_norm(model: nn.Module,
                   xyz_bcn_req: torch.Tensor,
                   label_B: torch.Tensor,
                   amp_dtype: Optional[torch.dtype],
                   mask_allowed: bool,
                   num_part: int = 50,
                   prefer_sngp: bool = False) -> torch.Tensor:
    """Compute per-point gradient risk ‖∂s/∂x‖ where s is the logit of predicted class per point.
    Returns: (B, N) tensor of risk.

    Notes:
    - Always uses baseline head (prefer_sngp=False) for risk.
    - Manages its own no-grad/grad boundaries.
    """
    assert xyz_bcn_req.requires_grad, "xyz must require grad for risk"

    # First forward (no grad) to get per-point class
    with torch.no_grad():
        logits_bnc, _ = run_model_logits(
            model, xyz_bcn_req, label_B,
            amp_enabled=False, amp_dtype=None,
            num_part=num_part, prefer_sngp=False
        )
        if mask_allowed and (SEG_CLASSES is not None) and (CLASS_CHOICE is not None):
            pred = torch.empty(logits_bnc.shape[:2], dtype=torch.long, device=xyz_bcn_req.device)
            for i in range(logits_bnc.shape[0]):
                cat = CLASS_CHOICE[int(label_B[i].item())]
                allowed = torch.tensor(SEG_CLASSES[cat], device=xyz_bcn_req.device, dtype=torch.long)
                idx_local = torch.argmax(logits_bnc[i, :, allowed], dim=-1)
                pred[i] = allowed[idx_local]
        else:
            pred = logits_bnc.argmax(dim=-1)

    # Second forward WITH grad to take ∂s/∂x
    with autocast_cuda(True, amp_dtype):
        logits2_bnc, _ = run_model_logits(
            model, xyz_bcn_req, label_B,
            amp_enabled=False, amp_dtype=None,
            num_part=num_part, prefer_sngp=False
        )
        s = logits2_bnc.gather(dim=2, index=pred.unsqueeze(-1)).squeeze(-1)  # (B,N)

    grad = torch.autograd.grad(s.sum(), xyz_bcn_req, retain_graph=False, create_graph=False)[0]  # (B,3,N)
    risk = grad.norm(dim=1)  # (B,N)
    return risk


# ------------------------------
# Collate (batch=1) and gather helpers
# ------------------------------

def safe_collate_varlen(batch):
    # Expects items like (points[N,>=3], label_scalar, target[N])
    assert len(batch) == 1, "This script enforces batch_size=1."
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
    # x: (B,3,N), idx: (B,K) -> (B,3,K)
    B, C, N = x_b3n.shape
    idx_exp = idx_bk.unsqueeze(1).expand(-1, C, -1)
    return torch.gather(x_b3n, dim=2, index=idx_exp)


# ------------------------------
# Smart checkpoint loader (handles .pth and .ckpt)
# ------------------------------

def _strip_prefix(state_dict, prefix_list):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefix_list:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _smart_load_state_dict(ckpt_obj, classifier):
    if isinstance(ckpt_obj, dict) and 'state_dict' in ckpt_obj:
        sd = ckpt_obj['state_dict']
    elif isinstance(ckpt_obj, dict) and 'model_state_dict' in ckpt_obj:
        sd = ckpt_obj['model_state_dict']
    elif isinstance(ckpt_obj, dict):
        sd = ckpt_obj
    else:
        sd = {}

    sd = _strip_prefix(sd, ['module.', 'model.', 'net.', 'classifier.', 'backbone.'])

    try:
        missing, unexpected = classifier.load_state_dict(sd, strict=True)
        print(f"[INFO] ckpt loaded (strict). missing={len(missing)} unexpected={len(unexpected)}")
    except Exception:
        missing, unexpected = classifier.load_state_dict(sd, strict=False)
        print(f"[INFO] ckpt loaded (non-strict). missing={len(missing)} unexpected={len(unexpected)}")


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser("Noisy eval: pointcvar / entropy / sngp")
    parser.add_argument('--method', type=str, choices=['pointcvar','entropy','sngp'], required=True)
    parser.add_argument('--root', type=str, required=True, help='dataset root (noisy)')

    # Method-specific models (defaults set to user's paths)
    parser.add_argument('--model_cvar', type=str, default='models.pointnet2_part_seg_msg')
    parser.add_argument('--ckpt_cvar', type=str, default='log/part_seg/without_normal/checkpoints/best_model.pth')
    parser.add_argument('--model_entropy', type=str, default='models.pointnet2_part_seg_msg')
    parser.add_argument('--ckpt_entropy', type=str, default='log/part_seg/without_normal/checkpoints/best_model.pth')
    parser.add_argument('--model_sngp', type=str, default='models.sngp_s2_6layers')
    parser.add_argument('--ckpt_sngp', type=str, default='log/part_seg/sngp/checkpoints/best_model.ckpt')
    parser.add_argument('--precision_path', type=str, default=None)

    parser.add_argument('--num_point', type=int, default=None, help='point Number')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--keep', type=int, default=2048)
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--mask_allowed', action='store_true', default=True)
    parser.add_argument('--num_part', type=int, default=50)
    parser.add_argument('--time_include_post', action='store_true', help='include post-proc (selection/second-forward) time in latency')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try_import_seg_maps()

    # Choose model + ckpt based on method
    if args.method == 'pointcvar':
        model_path, ckpt_path = args.model_cvar, args.ckpt_cvar
        prefer_sngp = False
    elif args.method == 'entropy':
        model_path, ckpt_path = args.model_entropy, args.ckpt_entropy
        prefer_sngp = False
    else:  # sngp
        model_path, ckpt_path = args.model_sngp, args.ckpt_sngp
        prefer_sngp = True

    print(f"[INFO] method={args.method} | model_path={model_path} | ckpt_path={ckpt_path}")
    model_mod = importlib.import_module(model_path)

    # Instantiate via get_model(num_classes=..., normal_channel=...)
    def _instantiate_from_get_model(mod):
        if not hasattr(mod, "get_model"):
            raise AttributeError("get_model() not found")
        get_model = getattr(mod, "get_model")
        tried = []
        for sig in [
            lambda: get_model(num_classes=args.num_part, normal_channel=args.normal),
            lambda: get_model(num_classes=args.num_part),
            lambda: get_model(args.num_part),
        ]:
            try:
                m = sig()
                print("[INFO] instantiated via get_model(...)")
                return m
            except Exception as e:
                tried.append(repr(e))
        raise RuntimeError("get_model existed but all signatures failed: " + " | ".join(tried))

    def _instantiate_fallbacks(mod):
        tried = []
        for name in ("PointNet2", "DGCNN", "Model"):
            if hasattr(mod, name):
                ctor = getattr(mod, name)
                try:
                    m = ctor()
                    print(f"[INFO] instantiated via {name}()")
                    return m
                except Exception as e:
                    tried.append(f"{name}(): {repr(e)}")
        raise RuntimeError("No usable constructor. Tried: " + " | ".join(tried))

    try:
        classifier = _instantiate_from_get_model(model_mod)
    except Exception as e_get:
        print(f"[WARN] get_model path failed: {e_get}")
        classifier = _instantiate_fallbacks(model_mod)

    classifier = classifier.to(device)

    # Load ckpt (supports .pth/.ckpt)
    ckpt_obj = torch.load(ckpt_path, map_location=device,weights_only=False)
    _smart_load_state_dict(ckpt_obj, classifier)
    classifier.eval()

    # Data: PartNormalDataset (as requested)
    from data_utils.ShapeNetDataLoader_test import PartNormalDataset
    dataset = PartNormalDataset(root=args.root, split='test', normal_channel=args.normal, npoints=args.num_point)


    global SEG_CLASSES, CLASS_CHOICE
    if SEG_CLASSES is None and hasattr(dataset, "seg_classes"):
        SEG_CLASSES = dataset.seg_classes  # 映射: 类名 -> 该类允许的 part 索引列表
    if CLASS_CHOICE is None and hasattr(dataset, "classes"):
        # dataset.classes: 类名 -> 类别索引，转成 index -> 类名
        CLASS_CHOICE = [None] * len(dataset.classes)
        for cat, idx in dataset.classes.items():
            CLASS_CHOICE[idx] = cat
    print(f"[INFO] mask_allowed={args.mask_allowed} | "
        f"SEG_CLASSES loaded={SEG_CLASSES is not None} | "
        f"CLASS_CHOICE loaded={CLASS_CHOICE is not None}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=safe_collate_varlen, pin_memory=True)

    amp_dtype = torch.float16 if args.amp else None

    # Accumulators
    lat_ms = []
    remain_noise_counts = []
    kept_counts = []
    # IoU accumulators
    tp_c = torch.zeros(args.num_part, dtype=torch.long)
    fp_c = torch.zeros(args.num_part, dtype=torch.long)
    fn_c = torch.zeros(args.num_part, dtype=torch.long)
    inst_mious = []
    per_cat_shape_ious = {cat: [] for cat in SEG_CLASSES.keys()}


    # Loop (batch size = 1)
    for it, batch in enumerate(tqdm(loader, total=len(dataset), desc=f"{args.method} eval", ncols=80)):
        pts_nc, label_B, target_n = batch  # pts: (N, C>=3), label: (1,), target: (N,)
        # Ensure shapes to (B,3,N)
        pts = pts_nc.float()
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)  # (1,N,C)
        B, N, C = pts.shape
        assert B == 1, "Batch size forced to 1"
        xyz_bcn = pts[:, :, :3].transpose(2,1).contiguous().to(device)  # (1,3,N)
        label_B = label_B.to(device)
        gt_n = target_n.view(-1).to(device)    # shape = (N,)
        assert gt_n.dim() == 1, f"expected (N,) gt, got {gt_n.shape}"

        # Start timer
        t0 = time.perf_counter()

        # First forward (no grad) for selection/logits
        with torch.no_grad():
            logits_full_bnc, aux = run_model_logits(
                classifier, xyz_bcn, label_B,
                amp_enabled=args.amp, amp_dtype=amp_dtype,
                num_part=args.num_part, prefer_sngp=prefer_sngp
            )
        t_cut = time.perf_counter()  # time right after first forward

        # Selection + predictions
        if args.method == 'pointcvar':
            K = min(args.keep, N)

            if K == N:
                idx_sorted = torch.arange(N, device=xyz_bcn.device)
                kept_counts.append(int(K))

                noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
                remain_noise_counts.append(noise_kept)

                # Predictions from the first forward (baseline-equivalent)
                pred_kept = predict_labels_with_gt(
                    logits_full_bnc, label_B, gt_n,
                    mask_allowed=args.mask_allowed, num_part=args.num_part
                )[0]

                # Baseline-style per-category IoU for "dataset avg"
                cat = get_cat_from_gt(gt_n)
                if cat is None:
                    cat = CLASS_CHOICE[int(label_B[0].item())]
                parts = SEG_CLASSES[cat]
                part_ious = []
                for l in parts:
                    tp = ((gt_n == l) & (pred_kept == l)).sum().item()
                    fp = ((gt_n != l) & (pred_kept == l)).sum().item()
                    fn = ((gt_n == l) & (pred_kept != l)).sum().item()
                    denom = tp + fp + fn
                    iou_l = 1.0 if denom == 0 else (tp / float(denom))
                    part_ious.append(iou_l)
                per_cat_shape_ious[cat].append(float(sum(part_ious) / len(part_ious)))

                # Baseline timing (single forward)
                t1 = t_cut


            else:
                # === K < N：计算风险 + 二次前向 ===
                xyz_req = xyz_bcn.detach().clone().requires_grad_(True)
                risk_bn = grad_risk_norm(
                    classifier, xyz_req, label_B,
                    amp_dtype, mask_allowed=args.mask_allowed,
                    num_part=args.num_part, prefer_sngp=False
                )  # (1, N)

                # 取最低风险 K 个
                idx_sorted = torch.argsort(risk_bn[0], descending=False)[:K]
                kept_counts.append(int(K))

                # kept 内噪声数
                noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
                remain_noise_counts.append(noise_kept)

                # 二次前向只对 kept 点
                with torch.no_grad():
                    xyz_kept_b3k = gather_bn3_to_bn(xyz_bcn, idx_sorted.view(1, -1))
                    logits_kept_bnc, _ = run_model_logits(
                        classifier, xyz_kept_b3k, label_B,
                        amp_enabled=args.amp, amp_dtype=amp_dtype,
                        num_part=args.num_part, prefer_sngp=False
                    )

                # ⚠️ 这里一定要用 logits_kept_bnc，并且 GT 也要切到 kept
                pred_kept = predict_labels_with_gt(
                    logits_kept_bnc, label_B, gt_n[idx_sorted],
                    mask_allowed=args.mask_allowed, num_part=args.num_part
                )[0]
                gt_kept = gt_n[idx_sorted]

                # 计时边界
                t1 = time.perf_counter() if args.time_include_post else t_cut

        elif args.method == 'entropy':
            prob = torch.softmax(logits_full_bnc, dim=-1)
            ent = -(prob * (prob.clamp_min(1e-9).log())).sum(dim=-1)  # (1,N)
            K = min(args.keep, ent.shape[1])
            idx_sorted = torch.argsort(ent[0], descending=False)[:K]
            kept_counts.append(int(K))

            noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
            remain_noise_counts.append(noise_kept)

            pred_full = predict_labels_with_gt(
                logits_full_bnc, label_B, gt_n,
                mask_allowed=args.mask_allowed, num_part=args.num_part
            )[0]
            pred_kept = pred_full[idx_sorted]

            t1 = time.perf_counter() if args.time_include_post else t_cut

        else:  # sngp
            unc_bn = extract_uncertainty(aux, logits_full_bnc)  # (1,N)
            K = min(args.keep, unc_bn.shape[1])
            idx_sorted = torch.argsort(unc_bn[0], descending=False)[:K]
            kept_counts.append(int(K))

            noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
            remain_noise_counts.append(noise_kept)

            pred_full = predict_labels_with_gt(
                logits_full_bnc, label_B, gt_n,
                mask_allowed=args.mask_allowed, num_part=args.num_part
            )[0]
            pred_kept = pred_full[idx_sorted]

            t1 = time.perf_counter() if args.time_include_post else t_cut


        # IoU accumulation on kept indices (noise counted as wrong)
        gt_kept = gt_n[idx_sorted]
        cat_i = get_cat_from_gt(gt_kept)
        if cat_i is None:
            # all -1 in kept set → penalize as 0
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
            inst_mious.append(shape_iou)                  # instance mIoU (kept-only, official-style)
            per_cat_shape_ious[cat_i].append(shape_iou)   # class mIoU (kept-only, dataset-avg)


        # Latency accounting
        lat_ms.append((t1 - t0) * 1000.0)

        # Per-sample log
        print(f"[SAMPLE {it:05d}] N={N} | kept={K} | remain_noise_in_kept={noise_kept} | latency_ms={lat_ms[-1]:.2f}")

    # Summary stats
    def percentile(xs, p):
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        k = (len(xs_sorted)-1) * (p/100.0)
        f = math.floor(k); c = math.ceil(k)
        if f == c:
            return xs_sorted[int(k)]
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
    inst_miou_kept = sum(inst_mious)/len(inst_mious)
    cat_means = [sum(v)/len(v) for v in per_cat_shape_ious.values() if len(v)]
    class_miou_kept_dataset_avg = (sum(cat_means)/len(cat_means)) if cat_means else 0.0

    print(f"Instance mIoU (kept-only, official-style): {inst_miou_kept:.4f}")
    print(f"Class mIoU (kept-only, dataset-avg): {class_miou_kept_dataset_avg:.4f}")


if __name__ == '__main__':
    main()
