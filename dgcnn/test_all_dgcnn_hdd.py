#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# DGCNN noisy-eval (SNGP vs PointCVaR) — single script
# - Both methods: load ALL points from dataset (npoints=None), then RANDOMLY SAMPLE to 2048 BEFORE forward.
# - SNGP: PN2-FRONT sigma^2 ranks points; we keep K for evaluation (single forward on sampled set).
# - PointCVaR: first forward -> gradient-risk per point; keep K; second forward on kept subset (reforward).

from __future__ import annotations
import os, sys, time, math, argparse, importlib
from typing import Optional, Dict
from statistics import mean, pstdev

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------
# Seg maps (optional)
# ------------------------------
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
# Util
# ------------------------------
def _maybe_square_tensor(x):
    return isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] == x.shape[1]

def restore_precision_from_any(path: str, classifier, device: torch.device) -> bool:
    import os
    if not path or not os.path.exists(path):
        return False
    obj = torch.load(path, map_location="cpu")

    # A) raw tensor = P
    if isinstance(obj, torch.Tensor):
        P = obj.to(device).float()
        P = 0.5 * (P + P.T)
        P = P + 10.0 * torch.eye(P.shape[0], device=P.device)
        if not hasattr(classifier, "early_rffgp"):
            raise RuntimeError("SNGP model has no early_rffgp head to load precision into.")
        classifier.early_rffgp.load_precision(P)
        print(f"[precision] loaded raw tensor from {path}")
        return True

    # B) dict
    if isinstance(obj, dict):
        # B1) state-dict style
        model_keys = set(classifier.state_dict().keys())
        overlap = len(model_keys & set(obj.keys()))
        if overlap > 0:
            missing, unexpected = classifier.load_state_dict(obj, strict=False)
            print(f"[precision] state-dict restored from {path} | missing={len(missing)} unexpected={len(unexpected)}")
            return True
        # B2) contains a P tensor
        for k in ["P","precision","precision_matrix","prec","P_clean","module.P"]:
            v = obj.get(k, None)
            if _maybe_square_tensor(v):
                P = v.to(device).float()
                P = 0.5 * (P + P.T)
                P = P + 10.0 * torch.eye(P.shape[0], device=P.device)
                if not hasattr(classifier, "early_rffgp"):
                    raise RuntimeError("SNGP model has no early_rffgp head to load precision into.")
                classifier.early_rffgp.load_precision(P)
                print(f"[precision] loaded P (‘{k}’) from {path}")
                return True
        # B3) any square 2D tensor
        cands = [(k, v) for k, v in obj.items() if _maybe_square_tensor(v)]
        if cands:
            cands.sort(key=lambda kv: (not any(t in kv[0].lower() for t in ["p","prec"]), kv[0]))
            _, P = cands[0]
            P = P.to(device).float()
            P = 0.5 * (P + P.T)
            P = P + 10.0 * torch.eye(P.shape[0], device=P.device)
            if not hasattr(classifier, "early_rffgp"):
                raise RuntimeError("SNGP model has no early_rffgp head to load precision into.")
            classifier.early_rffgp.load_precision(P)
            print(f"[precision] loaded square tensor (‘{cands[0][0]}’) from {path}")
            return True

    return False


def autocast_cuda(enabled: bool, dtype: Optional[torch.dtype]):
    if not enabled:
        class Dummy:
            def __enter__(self): return None
            def __exit__(self, *a, **k): return False
        return Dummy()
    return torch.autocast(device_type='cuda', dtype=dtype if dtype is not None else torch.float16)

def _one_hot_labels(label: torch.Tensor, num_classes: int = 16) -> torch.Tensor:
    label = torch.as_tensor(label).to(torch.int64).view(-1)
    B = label.shape[0]
    oh = torch.zeros(B, num_classes, device=label.device, dtype=torch.float32)
    oh.scatter_(1, label.view(-1,1), 1.0)
    return oh

def safe_collate_varlen(batch):
    assert len(batch) == 1
    pts, lbl, tgt = batch[0]
    if not isinstance(pts, torch.Tensor): pts = torch.tensor(pts)
    if not isinstance(tgt, torch.Tensor): tgt = torch.tensor(tgt, dtype=torch.long)
    else: tgt = tgt.long()
    if isinstance(lbl, torch.Tensor): lbl = lbl.view(1).long()
    else: lbl = torch.tensor([int(lbl)], dtype=torch.long)
    return pts, lbl, tgt

def gather_bn3_to_bn(x_b3n: torch.Tensor, idx_bk: torch.Tensor) -> torch.Tensor:
    B, C, N = x_b3n.shape
    idx_exp = idx_bk.unsqueeze(1).expand(-1, C, -1)
    return torch.gather(x_b3n, dim=2, index=idx_exp)

# ------------------------------
# DGCNN forward helper -> (B,N,C)
# ------------------------------
def dgcnn_logits_bnc(model: nn.Module,
                     xyz_bcn: torch.Tensor,
                     label_B: torch.Tensor,
                     amp_enabled: bool,
                     amp_dtype: Optional[torch.dtype],
                     num_part: int = 5) -> torch.Tensor:
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
# PN2-FRONT for sigma^2
# ------------------------------
def build_pnfront_and_load(pn_mod_path: str, ckpt_path: str, p_path: str, device: torch.device):
    # 1) build PN2-front
    mod = importlib.import_module(pn_mod_path)
    if not hasattr(mod, 'get_model'):
        raise RuntimeError('PN-front module must expose get_model(num_classes=5)')
    pn2 = mod.get_model(num_classes=5).eval().to(device)

    # 2) load PN2-front weights (geom_tf + early_rffgp only)
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
    sd_front = { _strip(k): v for k, v in sd_raw.items()
                 if _strip(k).startswith('geom_tf') or _strip(k).startswith('early_rffgp') }
    pn2.load_state_dict(sd_front, strict=False)

    # 3) load precision into early_rffgp (robust to tensor OR state-dict)
    if not p_path:
        raise RuntimeError("Provide --precision_path (tensor or ckpt containing P).")
    ok = restore_precision_from_any(p_path, pn2, device)   # << pass pn2, not classifier
    if not ok:
        raise RuntimeError(f"Could not restore precision info from {p_path}.")

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

def predictive_entropy(logits_bnc: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits_bnc, dim=-1)
    return -(prob * (prob.clamp_min(1e-9).log())).sum(dim=-1)


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
# ------------------------------
# PointCVaR gradient-risk
# ------------------------------
def grad_risk_norm(model: nn.Module,
                   xyz_bcn_req: torch.Tensor,
                   label_B: torch.Tensor,
                   amp_dtype: Optional[torch.dtype],
                   num_part: int = 5) -> torch.Tensor:
    """per-point risk = ||∂ s_pred / ∂ xyz||, where s_pred is the score of argmax class (stop-grad on class)"""
    assert xyz_bcn_req.requires_grad
    with torch.no_grad():
        logits_bnc = dgcnn_logits_bnc(model, xyz_bcn_req, label_B, amp_enabled=False, amp_dtype=None, num_part=num_part)
        pred = logits_bnc.argmax(dim=-1)
    with autocast_cuda(True, amp_dtype):
        logits2_bnc = dgcnn_logits_bnc(model, xyz_bcn_req, label_B, amp_enabled=False, amp_dtype=None, num_part=num_part)
        s = logits2_bnc.gather(dim=2, index=pred.unsqueeze(-1)).squeeze(-1)  # (B,N)
    grad = torch.autograd.grad(s.sum(), xyz_bcn_req, retain_graph=False, create_graph=False)[0]  # (B,3,N)
    return grad.norm(dim=1)  # (B,N)

# ------------------------------
# Kept-only penalized IoU (noise-penalized, baseline style)
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
def penalized_iou_kept(gt_kept: torch.Tensor, pr_kept: torch.Tensor, parts_full: list[int]) -> float:
    """
    Kept-only, baseline-style: IoU_c = TP / (TP + FP + FN + FP_noise),
    computed on the KEPT indices. parts_full = all GT parts in the full shape (ignoring -1).
    """
    if not parts_full: return 0.0
    ious = []
    for c in parts_full:
        gt_c = (gt_kept == c)
        pr_c = (pr_kept == c)
        tp = (gt_c & pr_c).sum().item()
        fp = ((~gt_c) & (gt_kept != -1) & pr_c).sum().item()
        fn = (gt_c & (~pr_c)).sum().item()
        fp_noise = ((gt_kept == -1) & pr_c).sum().item()
        denom = tp + fp + fn + fp_noise
        ious.append(1.0 if denom == 0 else (tp / float(denom)))
    return float(sum(ious) / len(ious))

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
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser("Noisy eval (DGCNN) — SNGP vs PointCVaR")
    parser.add_argument('--method', type=str, choices=['sngp','pointcvar','baseline'], required=True)

    # paths
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dgcnn_module', type=str, default='dgcnn.model_hdd')
    parser.add_argument('--dgcnn_ckpt', type=str, default='dgcnn/pretrained/model.t7')
    parser.add_argument('--pnfront_module', type=str, default='pointnet2.models.sngp_hdd')
    parser.add_argument('--pnfront_ckpt', type=str, default='pointnet2/log/sngp_hdd/checkpoints/best_model.pth')
    parser.add_argument('--precision_path', type=str, default='pointnet2/log/sngp_hdd/checkpoints/P_clean.pth')

    # eval config
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--mask_allowed', action='store_true', default=False)
    parser.add_argument('--num_part', type=int, default=5)

    # sampling & keeping
    parser.add_argument('--sample_to', type=int, default=None, help='RANDOM sample to this size BEFORE forward; None/0 to disable')
    parser.add_argument('--keep', type=int, default=2048, help='kept size for evaluation / re-forward in PointCVaR')

    # dataloader
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--time_include_post', action='store_true', help='include post-processing time in latency measurement')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.float16 if args.amp else None

    try_import_seg_maps()

    # load ALL points from dataset (npoints=None)
    from pointnet2.data_utils.ShapeNetDataLoader_test import PartNormalDataset
    dataset = PartNormalDataset(root=args.root, split='test', normal_channel=args.normal, npoints=None)

    global SEG_CLASSES, CLASS_CHOICE
    if SEG_CLASSES is None and hasattr(dataset, 'seg_classes'):
        SEG_CLASSES = dataset.seg_classes
    if CLASS_CHOICE is None and hasattr(dataset, 'classes'):
        CLASS_CHOICE = [None] * len(dataset.classes)
        for cat, idx in dataset.classes.items():
            CLASS_CHOICE[idx] = cat
    print(f"[INFO] seg_maps_loaded={SEG_CLASSES is not None}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=safe_collate_varlen, pin_memory=True)

    # DGCNN
    dg_mod = importlib.import_module(args.dgcnn_module)
    if hasattr(dg_mod, 'get_model'):
        try:
            classifier = dg_mod.get_model(num_classes=args.num_part, normal_channel=args.normal)
        except Exception:
            classifier = dg_mod.get_model(args.num_part)
    elif hasattr(dg_mod, 'DGCNN_partseg'):
        from types import SimpleNamespace
        dg_args = SimpleNamespace(k=20, emb_dims=1024, dropout=0.5)
        classifier = getattr(dg_mod, 'DGCNN_partseg')(dg_args, seg_num_all=args.num_part)
    else:
        raise RuntimeError('DGCNN module must expose get_model(...) or DGCNN_partseg(...)')
    classifier = classifier.to(device).eval()

    # load DGCNN weights
    obj = torch.load(args.dgcnn_ckpt, map_location='cpu', weights_only=False)
    sd = obj.get('state_dict', obj.get('model_state_dict', obj if isinstance(obj, dict) else {}))
    clean = {}
    for k,v in sd.items():
        k2 = k
        for p in ('module.','model.','net.','classifier.','backbone.'):
            if k2.startswith(p): k2 = k2[len(p):]
        clean[k2] = v
    try:
        classifier.load_state_dict(clean, strict=True)
        print('[INFO] DGCNN ckpt loaded (strict)')
    except Exception as e:
        classifier.load_state_dict(clean, strict=False)
        print('[INFO] DGCNN ckpt loaded (non-strict):', e)

    # PN2 FRONT (only if SNGP)
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
                pred_kept = predict_labels_with_gt(logits_full_bnc, label_B, gt_n, num_part=args.num_part)[0]
                t1 = t_cut
            else:
                xyz_req = xyz_bcn.detach().clone().requires_grad_(True)
                risk_bn = grad_risk_norm(classifier, xyz_req, label_B, amp_dtype, num_part=args.num_part)
                idx_sorted = torch.argsort(risk_bn[0], descending=False)[:K]
                with torch.no_grad():
                    xyz_kept_b3k = gather_bn3_to_bn(xyz_bcn, idx_sorted.view(1,-1))
                    logits_kept_bnc = dgcnn_logits_bnc(classifier, xyz_kept_b3k, label_B, amp_enabled=args.amp, amp_dtype=amp_dtype, num_part=args.num_part)
                pred_kept = predict_labels_with_gt(logits_kept_bnc, label_B, gt_n[idx_sorted], mask_allowed=args.mask_allowed, num_part=args.num_part)[0]
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
    lat_ms_mid = lat_ms[100:450]

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

    # avg_noise = (sum(kept_noise_counts)/len(kept_noise_counts)) if kept_noise_counts else 0.0
    inst_miou_kept = (sum(inst_mious)/len(inst_mious)) if inst_mious else 0.0
    print("\n[SUMMARY]")
    print(f"Avg noise count in kept-{K}: {avg_noise_kept:.2f}")
    print(f"Instance mIoU (kept-only | {args.method}): {inst_miou_kept:.4f}")

if __name__ == '__main__':
    main()
