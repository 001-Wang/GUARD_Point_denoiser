#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# DGCNN noisy-eval (REFINED)
# - SNGP selection: PN2-FRONT (geom_tf + early_rffgp) sigma^2 ALWAYS, then DGCNN on kept.
# - Entropy selection: predictive entropy from DGCNN full forward.
# - PointCVaR: gradient-risk on DGCNN, re-forward on kept.
# This aligns selection with the PointNet2 pipeline that uses PN2-FRONT sigma^2, ensuring consistency.
from __future__ import annotations
import os, sys, math, time, importlib, argparse
from typing import Dict, Optional, Tuple
from statistics import mean, pstdev

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
# ------------------------------
# Utils
# ------------------------------

def autocast_cuda(enabled: bool, dtype: Optional[torch.dtype]):
    if not enabled:
        class Dummy:
            def __enter__(self): return None
            def __exit__(self, *args, **kwargs): return False
        return Dummy()
    return torch.autocast(device_type='cuda', dtype=dtype if dtype is not None else torch.float16)

def _one_hot_labels(label: torch.Tensor, num_classes: int = 16) -> torch.Tensor:
    label = torch.as_tensor(label).to(torch.int64).view(-1)
    B = label.shape[0]
    oh = torch.zeros(B, num_classes, device=label.device, dtype=torch.float32)
    oh.scatter_(1, label.view(-1,1), 1.0)
    return oh

SEG_CLASSES: Optional[Dict[str, list]] = None
CLASS_CHOICE: Optional[list] = None

def try_import_seg_maps():
    global SEG_CLASSES, CLASS_CHOICE
    for modname in ['partnet_seg_map','data_utils.part_seg_utils','utils.part_seg_utils','provider']:
        try:
            m = importlib.import_module(modname)
            if hasattr(m, 'SEG_CLASSES') and hasattr(m, 'CLASS_CHOICE'):
                SEG_CLASSES = getattr(m, 'SEG_CLASSES')
                CLASS_CHOICE = list(getattr(m, 'CLASS_CHOICE'))
                return
        except Exception:
            pass

# ------------------------------
# Forward normalizer (DGCNN) -> (B,N,C)
# ------------------------------
def dgcnn_logits_bnc(model: nn.Module,
                     xyz_bcn: torch.Tensor,
                     label_B: torch.Tensor,
                     amp_enabled: bool,
                     amp_dtype: Optional[torch.dtype],
                     num_part: int = 50) -> torch.Tensor:
    B, C, N = xyz_bcn.shape
    oh = _one_hot_labels(label_B, num_classes=16).unsqueeze(-1)
    with autocast_cuda(amp_enabled, amp_dtype):
        out = model(xyz_bcn, oh)
    if out.dim() != 3:
        raise RuntimeError(f'DGCNN forward must return (B,*,*). Got {tuple(out.shape)}')
    if out.shape[1] == num_part and out.shape[2] == N:
        out = out.transpose(1,2).contiguous()
    elif out.shape[1] == N and out.shape[2] == num_part:
        pass
    elif out.shape[-1] == num_part:
        pass
    elif out.shape[1] == num_part:
        out = out.transpose(1,2).contiguous()
    else:
        raise RuntimeError(f'Unexpected logits shape {tuple(out.shape)}')
    return out

# ------------------------------
# PN2-FRONT (geom_tf + early_rffgp) to get sigma^2
# ------------------------------
def build_pnfront_and_load(pn_mod_path: str, ckpt_path: str, p_path: str, device: torch.device):
    mod = importlib.import_module(pn_mod_path)
    if not hasattr(mod, 'get_model'):
        raise RuntimeError('PN-front module must expose get_model(num_classes=50)')
    pn2 = mod.get_model(num_classes=50).eval().to(device)

    obj = torch.load(ckpt_path, map_location='cpu')
    if isinstance(obj, dict) and 'model_state_dict' in obj and isinstance(obj['model_state_dict'], dict):
        sd_raw = obj['model_state_dict']
    elif isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        sd_raw = obj['state_dict']
    elif isinstance(obj, dict):
        sd_raw = obj
    else:
        raise RuntimeError('Unrecognized PN2 checkpoint format')

    def _strip(k): return k.replace('module.', '') if k.startswith('module.') else k
    sd_front = { _strip(k): v for k,v in sd_raw.items()
                 if _strip(k).startswith('geom_tf') or _strip(k).startswith('early_rffgp') }
    pn2.load_state_dict(sd_front, strict=False)

    Pobj = torch.load(p_path, map_location='cpu')
    P = Pobj['precision'] if isinstance(Pobj, dict) and 'precision' in Pobj else Pobj
    P = torch.as_tensor(P, dtype=torch.float32)
    P = 0.5*(P+P.T) + 10*torch.eye(P.shape[0], dtype=P.dtype)
    pn2.early_rffgp.load_precision(P)
    return pn2

@torch.no_grad()
def pn2_sigma2_from_front(pn2, xyz_bcn: torch.Tensor) -> torch.Tensor:
    x_bnc = xyz_bcn.transpose(1,2).contiguous()
    h_bnc = pn2.geom_tf(x_bnc)
    out = pn2.early_rffgp(h_bnc, update_precision=False, compute_var=True)
    sigma2 = out[1] if isinstance(out, tuple) and len(out) == 2 else out
    if sigma2.dim() == 3 and sigma2.shape[-1] == 1: sigma2 = sigma2.squeeze(-1)
    if sigma2.dim() == 3 and sigma2.shape[1] == 1: sigma2 = sigma2.squeeze(1)
    if sigma2.dim() != 2: sigma2 = sigma2[..., 0]
    return sigma2

# ------------------------------
# Selection metrics
# ------------------------------
def entropy_over_allowed(logits_bnc: torch.Tensor,
                         label_B: torch.Tensor,
                         mask_allowed: bool,
                         num_part: int = 50) -> torch.Tensor:
    """
    Entropy over the legal parts of the sample's category (B==1 here).
    Falls back to unmasked entropy if seg maps are not available or mask_allowed=False.
    Returns (B,N).
    """
    if not (mask_allowed and (SEG_CLASSES is not None) and (CLASS_CHOICE is not None)):
        return predictive_entropy(logits_bnc)

    B, N, C = logits_bnc.shape
    assert B == 1, "This helper assumes batch_size=1."
    cat = CLASS_CHOICE[int(label_B[0].item())]
    allowed = SEG_CLASSES.get(cat, None)
    if not allowed:
        return predictive_entropy(logits_bnc)

    idx = torch.tensor(allowed, device=logits_bnc.device, dtype=torch.long)
    logits_allowed = logits_bnc[:, :, idx]         # (1,N,|allowed|)
    prob = torch.softmax(logits_allowed, dim=-1).clamp_min(1e-12)
    ent = -(prob * prob.log()).sum(dim=-1)          # (1,N)
    return ent

def robust_minmax_per_sample(x_bn: torch.Tensor,
                             q_lo: float = 0.10,
                             q_hi: float = 0.90,
                             eps: float = 1e-6) -> torch.Tensor:
    """
    Robust [0,1] scaling using per-sample pseudo-quantiles.
    x_bn: (B,N)  ->  returns (B,N)
    """
    B, N = x_bn.shape
    if N == 1:
        return torch.zeros_like(x_bn)
    x_sorted, _ = torch.sort(x_bn, dim=1)
    lo_i = int(round(q_lo * (N - 1))); lo_i = max(0, min(lo_i, N-1))
    hi_i = int(round(q_hi * (N - 1))); hi_i = max(0, min(hi_i, N-1))
    arangeB = torch.arange(B, device=x_bn.device)
    q_lo_v = x_sorted[arangeB, lo_i][:, None]
    q_hi_v = x_sorted[arangeB, hi_i][:, None]
    scale = (q_hi_v - q_lo_v).abs().clamp_min(eps)
    return ((x_bn - q_lo_v) / scale).clamp(0.0, 1.0)


def predictive_entropy(logits_bnc: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits_bnc, dim=-1)
    return -(prob * (prob.clamp_min(1e-9).log())).sum(dim=-1)

def grad_risk_norm(model: nn.Module,
                   xyz_bcn_req: torch.Tensor,
                   label_B: torch.Tensor,
                   amp_dtype: Optional[torch.dtype],
                   mask_allowed: bool,
                   num_part: int = 50) -> torch.Tensor:
    assert xyz_bcn_req.requires_grad
    with torch.no_grad():
        logits_bnc = dgcnn_logits_bnc(model, xyz_bcn_req, label_B, amp_enabled=False, amp_dtype=None, num_part=num_part)
        pred = logits_bnc.argmax(dim=-1)
    with autocast_cuda(True, amp_dtype):
        logits2_bnc = dgcnn_logits_bnc(model, xyz_bcn_req, label_B, amp_enabled=False, amp_dtype=None, num_part=num_part)
        s = logits2_bnc.gather(dim=2, index=pred.unsqueeze(-1)).squeeze(-1)
    grad = torch.autograd.grad(s.sum(), xyz_bcn_req, retain_graph=False, create_graph=False)[0]
    return grad.norm(dim=1)

# ------------------------------
# Data helpers
# ------------------------------
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
# Part decoder / IoU helpers
# ------------------------------
def get_cat_from_gt(gt_n: torch.Tensor) -> Optional[str]:
    if not hasattr(get_cat_from_gt, '_map') and SEG_CLASSES is not None:
        m = {}
        for cat, parts in SEG_CLASSES.items():
            for l in parts: m[l] = cat
        get_cat_from_gt._map = m
    idx = (gt_n != -1).nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0 or not hasattr(get_cat_from_gt, '_map'): return None
    lab = int(gt_n[idx[0]].item())
    return get_cat_from_gt._map.get(lab, None)

def predict_labels_with_gt(logits_bnc: torch.Tensor,
                           label_B: torch.Tensor,
                           gt_n: torch.Tensor,
                           mask_allowed: bool,
                           num_part: int = 50) -> torch.Tensor:
    B, N, C = logits_bnc.shape
    if not (mask_allowed and (SEG_CLASSES is not None) and (CLASS_CHOICE is not None)):
        return logits_bnc.argmax(dim=-1)
    pred = torch.empty((B, N), dtype=torch.long, device=logits_bnc.device)
    seg_label_to_cat = {}
    for cat, parts in SEG_CLASSES.items():
        for l in parts: seg_label_to_cat[l] = cat
    for i in range(B):
        gt_i = gt_n
        idx = (gt_i != -1).nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            first_lab = int(gt_i[idx[0]].item())
            cat = seg_label_to_cat.get(first_lab, None)
        else:
            cat = CLASS_CHOICE[int(label_B[i].item())]
        allowed = torch.tensor(SEG_CLASSES[cat], device=logits_bnc.device, dtype=torch.long)
        idx_local = torch.argmax(logits_bnc[i, :, allowed], dim=-1)
        pred[i] = allowed[idx_local]
    return pred

# ------------------------------
# Checkpoint loader
# ------------------------------
def load_model_state(classifier: nn.Module, ckpt_path: str):
    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = obj.get('state_dict', obj.get('model_state_dict', obj if isinstance(obj, dict) else {}))
    clean = {}
    for k,v in sd.items():
        k2 = k
        for p in ('module.','model.','net.','classifier.','backbone.'):
            if k2.startswith(p): k2 = k2[len(p):]
        clean[k2] = v
    try:
        classifier.load_state_dict(clean, strict=True)
        print('[INFO] ckpt loaded (strict)')
    except Exception as e:
        classifier.load_state_dict(clean, strict=False)
        print('[INFO] ckpt loaded (non-strict):', e)

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser('Noisy eval (DGCNN backbone) with unified selection')
    parser.add_argument('--method', type=str, choices=['pointcvar','entropy','sngp'], required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dgcnn_module', type=str, default=r'dgcnn.model')
    parser.add_argument('--dgcnn_ckpt', type=str, default=r'dgcnn/pretrained/model.partseg.t7')
    parser.add_argument('--pnfront_module', type=str,default='pointnet2.models.sngp_s2_6layers')
    parser.add_argument('--pnfront_ckpt', type=str, default=r'pointnet2/log/sngp/checkpoints/best_model.ckpt')
    parser.add_argument('--precision_path', type=str, default=r'pointnet2/log/sngp/checkpoints/P_clean.pt')

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

    try_import_seg_maps()

    from pointnet2.data_utils.ShapeNetDataLoader_test import PartNormalDataset
    dataset = PartNormalDataset(root=args.root, split='test', normal_channel=args.normal, npoints=args.num_point)

    global SEG_CLASSES, CLASS_CHOICE
    if SEG_CLASSES is None and hasattr(dataset, 'seg_classes'):
        SEG_CLASSES = dataset.seg_classes
    if CLASS_CHOICE is None and hasattr(dataset, 'classes'):
        CLASS_CHOICE = [None] * len(dataset.classes)
        for cat, idx in dataset.classes.items():
            CLASS_CHOICE[idx] = cat
    print(f"[INFO] mask_allowed={args.mask_allowed} | seg_maps_loaded={SEG_CLASSES is not None}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=safe_collate_varlen, pin_memory=True)

    dg_mod = importlib.import_module(args.dgcnn_module)
    if hasattr(dg_mod, 'get_model'):
        try:
            classifier = dg_mod.get_model(num_classes=args.num_part, normal_channel=args.normal)
        except Exception:
            classifier = dg_mod.get_model(args.num_part)
    elif hasattr(dg_mod, 'DGCNN_partseg'):
        from types import SimpleNamespace
        dg_args = SimpleNamespace(k=20, emb_dims=1024, dropout=0.5)
        classifier =  getattr(dg_mod, 'DGCNN_partseg')(dg_args, seg_num_all=args.num_part)
    else:
        raise RuntimeError('DGCNN module must expose get_model(...) or DGCNN_partseg(...)')
    classifier = classifier.to(device).eval()
    load_model_state(classifier, args.dgcnn_ckpt)

    pn2 = None
    if args.method == 'sngp':
        pn2 = build_pnfront_and_load(args.pnfront_module, args.pnfront_ckpt, args.precision_path, device)

    lat_ms = []; remain_noise_counts = []; kept_counts = []; inst_mious = []
    per_cat_shape_ious = {cat: [] for cat in SEG_CLASSES.keys()} if SEG_CLASSES else {}

    for it, batch in enumerate(tqdm(loader, total=len(dataset), desc=f"{args.method} eval", ncols=80)):
        pts_nc, label_B, target_n = batch
        pts = pts_nc.float()
        if pts.dim() == 2: pts = pts.unsqueeze(0)
        B, N, C = pts.shape
        assert B == 1
        xyz_bcn = pts[:, :, :3].transpose(2,1).contiguous().to(device)
        label_B = label_B.to(device)
        gt_n = target_n.view(-1).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits_full_bnc = dgcnn_logits_bnc(classifier, xyz_bcn, label_B, amp_enabled=args.amp, amp_dtype=amp_dtype, num_part=args.num_part)
        t_cut = time.perf_counter()

        if args.method == 'pointcvar':
            K = min(args.keep, N)
            if K == N:
                idx_sorted = torch.arange(N, device=xyz_bcn.device)
                pred_kept = predict_labels_with_gt(logits_full_bnc, label_B, gt_n, mask_allowed=args.mask_allowed, num_part=args.num_part)[0]
                t1 = t_cut
            else:
                xyz_req = xyz_bcn.detach().clone().requires_grad_(True)
                risk_bn = grad_risk_norm(classifier, xyz_req, label_B, amp_dtype, mask_allowed=args.mask_allowed, num_part=args.num_part)
                idx_sorted = torch.argsort(risk_bn[0], descending=False)[:K]
                with torch.no_grad():
                    xyz_kept_b3k = gather_bn3_to_bn(xyz_bcn, idx_sorted.view(1,-1))
                    logits_kept_bnc = dgcnn_logits_bnc(classifier, xyz_kept_b3k, label_B, amp_enabled=args.amp, amp_dtype=amp_dtype, num_part=args.num_part)
                pred_kept = predict_labels_with_gt(logits_kept_bnc, label_B, gt_n[idx_sorted], mask_allowed=args.mask_allowed, num_part=args.num_part)[0]
                t1 = time.perf_counter() if args.time_include_post else t_cut

        elif args.method == 'entropy':
            ent = predictive_entropy(logits_full_bnc)
            K = min(args.keep, N)
            idx_sorted = torch.argsort(ent[0], descending=False)[:K]
            pred_full = predict_labels_with_gt(logits_full_bnc, label_B, gt_n, mask_allowed=args.mask_allowed, num_part=args.num_part)[0]
            pred_kept = pred_full[idx_sorted]
            t1 = time.perf_counter() if args.time_include_post else t_cut

        else:
            assert pn2 is not None, 'PN2 FRONT must be built for --method sngp'
            sigma2_bn = pn2_sigma2_from_front(pn2, xyz_bcn)        # (1,N)
            ent_bn = entropy_over_allowed(logits_full_bnc, label_B, mask_allowed=args.mask_allowed, num_part=args.num_part)  # (1,N)
            s_norm = robust_minmax_per_sample(sigma2_bn)   # (1,N)
            h_norm = robust_minmax_per_sample(ent_bn)      # (1,N)
            eps = 1e-6
            fused = 0.5 * s_norm + (1.0 - 0.5) * h_norm + eps * h_norm  # tiny tie-break
            K = min(args.keep, N)
            idx_sorted = torch.argsort(fused[0], descending=False)[:K]

            with torch.no_grad():
                rows = torch.arange(1, device=logits_full_bnc.device)[:, None]  # B==1
                logits_kept_bKC = logits_full_bnc[rows, idx_sorted.view(1, -1), :]
            pred_kept = predict_labels_with_gt(
                logits_kept_bKC, label_B, gt_n[idx_sorted],
                mask_allowed=args.mask_allowed, num_part=args.num_part
            )[0]

            t1 = time.perf_counter() if args.time_include_post else t_cut

        kept_counts.append(int(K))
        noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
        remain_noise_counts.append(noise_kept)

        gt_kept = gt_n[idx_sorted]
        cat_i = get_cat_from_gt(gt_kept) if SEG_CLASSES else None
        if (cat_i is None) or (SEG_CLASSES is None):
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

        lat_ms.append((t1 - t0) * 1000.0)
        print(f"[SAMPLE {it:05d}] N={N} | kept={K} | remain_noise_in_kept={noise_kept} | latency_ms={lat_ms[-1]:.2f}")

    def percentile(xs, p):
        if not xs: return 0.0
        xs_sorted = sorted(xs)
        k = (len(xs_sorted)-1) * (p/100.0)
        f = math.floor(k); c = math.ceil(k)
        if f == c: return xs_sorted[int(k)]
        return xs_sorted[f] * (c-k) + xs_sorted[c] * (k-f)
    lat_ms_mid = lat_ms[900:1900] if len(lat_ms) > 1900 else lat_ms[900:]

    save_path = os.path.join("results", f"{args.method}_latency_samples.csv")  # you can change the path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["latency_ms"])
        for v in lat_ms_mid:
            writer.writerow([v])
    print(f"[INFO] Saved {len(lat_ms_mid)} latency samples to {save_path}")
    # Compute latency statistics for that middle range
    lat_mean = mean(lat_ms_mid)
    lat_p50 = percentile(lat_ms_mid, 50)
    lat_p90 = percentile(lat_ms_mid, 90)
    lat_p95 = percentile(lat_ms_mid, 95)
    lat_std = pstdev(lat_ms_mid)
    avg_noise_kept = (sum(remain_noise_counts)/len(remain_noise_counts)) if remain_noise_counts else 0.0
    n = len(lat_ms)

    print("[SUMMARY]")
    print(f"latency Samples evaluated: 900-1900, noise eval set size={n}")
    print(f"Latency ms per-sample | mean={lat_mean:.2f} | p50={lat_p50:.2f} | p90={lat_p90:.2f} | p95={lat_p95:.2f} | std={lat_std:.2f}")
    print(f"Avg noise count in kept-{args.keep}: {avg_noise_kept:.2f}")
    inst_miou_kept = sum(inst_mious)/len(inst_mious) if inst_mious else 0.0
    print(f'Instance mIoU (kept-only): {inst_miou_kept:.4f}')
    if per_cat_shape_ious:
        cat_means = [sum(v)/len(v) for v in per_cat_shape_ious.values() if len(v)]
        class_miou_kept_dataset_avg = (sum(cat_means)/len(cat_means)) if cat_means else 0.0
        print(f'Class mIoU (kept-only, dataset-avg): {class_miou_kept_dataset_avg:.4f}')

if __name__ == '__main__':
    main()