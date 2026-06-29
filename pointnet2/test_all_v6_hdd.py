#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# ------------------------------
# Utilities
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


def build_valid_parts(label_B: torch.Tensor):
    if (SEG_CLASSES is None) or (CLASS_CHOICE is None):
        return None
    B = label_B.shape[0]
    C = max(max(v) for v in SEG_CLASSES.values()) + 1
    mask = torch.zeros(B, C, dtype=torch.bool, device=label_B.device)
    for i in range(B):
        cat = CLASS_CHOICE[int(label_B[i].item())]
        parts = SEG_CLASSES.get(cat, [])
        if parts:
            mask[i, torch.tensor(parts, device=label_B.device)] = True
    return mask

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

def run_model_logits(model: nn.Module,
                     xyz_bcn: torch.Tensor,
                     label_B: torch.Tensor,
                     amp_enabled: bool,
                     amp_dtype: Optional[torch.dtype],
                     num_part: int = 50,
                     prefer_sngp: bool = False,
                     forward_kwargs: Optional[dict] = None
                     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns
      logits_bnc: (B, N, num_part)
      aux       : optional dict; we extract uncertainty from keys like:
                  'sigma2_geo', 'sigma2_sem', 'sigma', 'sigma2', 'var', 'variance', 'log_var', 'logvar'

    Behavior
      - If model returns (base_logits, sngp_logits, aux), choose sngp_logits when prefer_sngp=True.
      - If model returns (logits, aux) with aux a dict, use that dict.
      - If model returns (logits, l3_points), ignore l3_points.
      - Else assume it returned logits directly.
    """
    B, C, N = xyz_bcn.shape
    device = xyz_bcn.device

    # one-hot to same device
    oh = _one_hot_labels(label_B, num_classes=16).to(device)

    forward_kwargs = forward_kwargs or {}
    with autocast_cuda(amp_enabled, amp_dtype):
        out = model(xyz_bcn, oh, **forward_kwargs)

    aux: Dict[str, torch.Tensor] = {}
    # (base_logits, sngp_logits, aux)
    if isinstance(out, tuple) and len(out) == 3:
        base_logits, sngp_logits, maybe_aux = out
        if isinstance(maybe_aux, dict):
            aux = maybe_aux
        logits = sngp_logits if (prefer_sngp and sngp_logits is not None) else base_logits

    # (logits, something) where something may be aux dict or l3_points tensor
    elif isinstance(out, tuple) and len(out) == 2:
        first, second = out
        if isinstance(second, dict):
            logits, aux = first, second
        else:
            logits, _ = first, second  # treat as (logits, l3_points)

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
        # fallback using last dim as classes
        if logits.shape[-1] == num_part:
            logits_bnc = logits
        elif logits.shape[1] == num_part:
            logits_bnc = logits.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)} for num_part={num_part}, N={N}")

    if not isinstance(aux, dict):
        aux = {}
    return logits_bnc, aux



# ------------------------------
# Uncertainty extraction for SNGP or fallback
# ------------------------------

def _masked_logits(logits_bnc: torch.Tensor, valid_parts_bc: Optional[torch.Tensor]):
    if valid_parts_bc is None:
        return logits_bnc
    C_logits = logits_bnc.shape[-1]
    C_mask   = valid_parts_bc.shape[-1]
    if C_logits != C_mask:
        print(f"[warn] class-mask size {C_mask} != logits size {C_logits}; skip masking.")
        return logits_bnc
    neg_inf = torch.finfo(logits_bnc.dtype).min
    mask = valid_parts_bc[:, None, :].to(torch.bool)  # (B,1,C)
    return torch.where(mask, logits_bnc, torch.full_like(logits_bnc, neg_inf))


def _entropy_from_logits(logits_bnc: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(logits_bnc, dim=-1).clamp_min(1e-12)
    return -(prob * prob.log()).sum(dim=-1)  # (B,N)

def _robust_minmax(x_bn: torch.Tensor, lo=0.10, hi=0.90, eps=1e-6) -> torch.Tensor:
    """
    Per-sample robust min-max via quantile-like indices.
    x_bn: (B, N)
    Returns: (B, N) scaled to [0,1].
    """
    B, N = x_bn.shape
    if N == 1:
        return torch.zeros_like(x_bn)  # avoid divide-by-zero when only one point

    # sort per-sample
    x_sorted, _ = torch.sort(x_bn, dim=1)

    # compute scalar indices as Python ints (avoid tensor.float.round on Python floats)
    lo_i = int(round(lo * (N - 1)))
    hi_i = int(round(hi * (N - 1)))
    lo_i = max(0, min(lo_i, N - 1))
    hi_i = max(0, min(hi_i, N - 1))

    arangeB = torch.arange(B, device=x_bn.device)
    q_lo = x_sorted[arangeB, lo_i][:, None]  # (B,1)
    q_hi = x_sorted[arangeB, hi_i][:, None]  # (B,1)

    scale = (q_hi - q_lo).abs().clamp_min(eps)
    return ((x_bn - q_lo) / scale).clamp(0.0, 1.0)


def extract_uncertainty(
    aux: Dict[str, torch.Tensor],
    logits_bnc: torch.Tensor,
    valid_parts_bc: Optional[torch.Tensor] = None,  # pass if you use class-mask at eval
    alpha_hybrid: float = 0.5,                      # weight on SNGP σ² (0~1)
    q_lo: float = 0.10, q_hi: float = 0.90,         # robust normalization quantiles
) -> torch.Tensor:
    """
    Return (B,N) hybrid uncertainty: alpha*norm(sigma2) + (1-alpha)*norm(entropy).
    Lower = more certain. No entropy-only fallback (asserts σ² exists).
    """
    B, N, C = logits_bnc.shape

    # ---- masked entropy (aligns with your eval argmax policy) ----
    logits_eff = _masked_logits(logits_bnc, valid_parts_bc)
    ent_bn = _entropy_from_logits(logits_eff)  # (B,N)

    # ---- fetch SNGP variance-like key (assert required) ----
    sigma2_bn = None
    for key in ['sigma2_geo','sigma2_sem','sigma2','sigma','var','variance','log_var','logvar']:
        v = aux.get(key)
        if v is None:
            continue
        t = v
        if t.dim() == 3 and t.shape[:2] == (B, N):
            t = t.mean(dim=-1)
        elif t.dim() == 2 and t.shape == (B, N):
            pass
        elif t.dim() == 1 and t.numel() == B * N:
            t = t.view(B, N)
        else:
            continue
        if 'log' in key:
            t = t.exp()
        sigma2_bn = t
        # print(f"[SNGP] using '{key}' from aux for uncertainty")
        break

    if sigma2_bn is None:
        raise RuntimeError("[SNGP] Expected sigma2-like key in aux but none found.")

    # ---- robust per-sample normalization & fusion ----
    s_norm = _robust_minmax(sigma2_bn, lo=q_lo, hi=q_hi)
    h_norm = _robust_minmax(ent_bn,   lo=q_lo, hi=q_hi)
    eps = 1e-6
    key_bn = alpha_hybrid * s_norm + (1.0 - alpha_hybrid) * h_norm + eps * h_norm  # tiny tie-break
    return key_bn


# ------------------------------
# Predictions + IoU helpers
# ------------------------------
def miou_local(pred: torch.Tensor, gt: torch.Tensor, C: int) -> float:
    ious = []
    for l in range(C):
        pc = (pred == l)
        gc = (gt == l)
        tp = (pc & gc).sum().item()
        fp = (pc & (~gc)).sum().item()   
        fn = ((~pc) & gc).sum().item()
        denom = tp + fp + fn
        ious.append(1.0 if denom == 0 else (tp / float(denom)))
    return float(sum(ious) / len(ious))


def _predict_local_argmax(logits_bnc: torch.Tensor) -> torch.Tensor:
    # logits_bnc: (B,N,C_local)
    return logits_bnc.argmax(dim=-1)

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
                try:
                    allowed = torch.tensor(SEG_CLASSES[cat], device=xyz_bcn_req.device, dtype=torch.long)
                except KeyError:
                    # HDD/simple path: allow the whole local head space [0..C-1]
                    C = logits_bnc.shape[-1]
                    allowed = torch.arange(C, device=xyz_bcn_req.device, dtype=torch.long)
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
    parser.add_argument('--hdd_simple', action='store_true',
                    help='HDD fast path: ignore SEG_CLASSES/category logic; evaluate in local 0..num_part-1.')
    parser.add_argument('--method', type=str, choices=['pointcvar','entropy','sngp'], required=True)
    parser.add_argument('--root', type=str, required=True, help='dataset root (noisy)')

    # Method-specific models (defaults set to user's paths)
    parser.add_argument('--model_cvar', type=str, default='pointnet2.models.pointnet2_hdd')
    parser.add_argument('--ckpt_cvar', type=str, default='pointnet2/log/without_normal_hdd/checkpoints/best_model.pth')
    # parser.add_argument('--model_entropy', type=str, default='pointnet2.models.pointnet2_part_seg_msg')
    # parser.add_argument('--ckpt_entropy', type=str, default='pointnet2/log/without_normal/checkpoints/best_model.pth')
    parser.add_argument('--model_entropy', type=str, default='pointnet2.models.sngp_hdd')
    parser.add_argument('--ckpt_entropy', type=str, default='pointnet2/log/sngp_hdd/checkpoints/best_model.ckpt')
    parser.add_argument('--model_sngp', type=str, default='pointnet2.models.sngp_hdd')
    parser.add_argument('--ckpt_sngp', type=str, default='pointnet2/log/sngp_hdd/checkpoints/best_model.pth')
    parser.add_argument('--precision_path', type=str, default='pointnet2/log/sngp_hdd/checkpoints/p_clean.pth')

    parser.add_argument('--num_point', type=int, default=None, help='point Number')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--keep', type=int, default=2048)
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--mask_allowed', action='store_true', default=True)
    parser.add_argument('--num_part', type=int, default=5)
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
    if args.method == 'sngp':
        src = args.precision_path
        if not src and not args.no_calib:
            raise RuntimeError("Provide --precision_cache (state-dict) or --precision_path (tensor/ckpt with P), or pass --no_calib.")
        if src:
            ok = restore_precision_from_any(src, classifier, device)
            if not ok and not args.no_calib:
                raise RuntimeError(f"Could not restore precision info from {src}. Pass --no_calib to skip.")
    classifier.eval()


    from data_utils.ShapeNetDataLoader_test import PartNormalDataset
    dataset = PartNormalDataset(root=args.root, split='test', normal_channel=args.normal, npoints=args.num_point)


    global SEG_CLASSES, CLASS_CHOICE
    if SEG_CLASSES is None and hasattr(dataset, "seg_classes"):
        SEG_CLASSES = dataset.seg_classes  
    if CLASS_CHOICE is None and hasattr(dataset, "classes"):
        CLASS_CHOICE = [None] * len(dataset.classes)
        for cat, idx in dataset.classes.items():
            CLASS_CHOICE[idx] = cat
    print(f"[INFO] mask_allowed={args.mask_allowed} | "
        f"SEG_CLASSES loaded={SEG_CLASSES is not None} | "
        f"CLASS_CHOICE loaded={CLASS_CHOICE is not None}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=safe_collate_varlen, pin_memory=True)
    
    SIMPLE_HDD = args.hdd_simple
    if not SIMPLE_HDD:
        if SEG_CLASSES is not None:
            global_C = max(max(v) for v in SEG_CLASSES.values()) + 1
            if args.num_part < global_C:
                SIMPLE_HDD = True
                print(f"[info] Enabling SIMPLE_HDD (num_part={args.num_part} < global seg space={global_C}).")
        else:
            SIMPLE_HDD = True
            print("[info] Enabling SIMPLE_HDD (no SEG_CLASSES provided).")

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
    inst_mious_all_penal = []
    inst_mious_kept_penal= [] 


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
            fw_kwargs = {"compute_sigma": True} if args.method == "sngp" else None
            logits_full_bnc, aux = run_model_logits(
                classifier, xyz_bcn, label_B,
                amp_enabled=args.amp, amp_dtype=amp_dtype,
                num_part=args.num_part, prefer_sngp=prefer_sngp,
                forward_kwargs=fw_kwargs
            )
        t_cut = time.perf_counter()  # time right after first forward
        if SIMPLE_HDD:
            pred_full = _predict_local_argmax(logits_full_bnc)[0]   # (N,)
        else:
            pred_full = predict_labels_with_gt(
                logits_full_bnc, label_B, gt_n,
                mask_allowed=args.mask_allowed, num_part=args.num_part
            )[0]

        # Selection + predictions
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
                xyz_req = xyz_bcn.detach().clone().requires_grad_(True)
                risk_bn = grad_risk_norm(
                    classifier, xyz_req, label_B,
                    amp_dtype, mask_allowed=args.mask_allowed,
                    num_part=args.num_part, prefer_sngp=False
                )  # (1, N)

                idx_sorted = torch.argsort(risk_bn[0], descending=False)[:K]
                kept_counts.append(int(K))

                noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
                remain_noise_counts.append(noise_kept)

                with torch.no_grad():
                    xyz_kept_b3k = gather_bn3_to_bn(xyz_bcn, idx_sorted.view(1, -1))
                    logits_kept_bnc, _ = run_model_logits(
                        classifier, xyz_kept_b3k, label_B,
                        amp_enabled=args.amp, amp_dtype=amp_dtype,
                        num_part=args.num_part, prefer_sngp=False
                    )

                pred_kept = predict_labels_with_gt(
                    logits_kept_bnc, label_B, gt_n[idx_sorted],
                    mask_allowed=args.mask_allowed, num_part=args.num_part
                )[0]
                gt_kept = gt_n[idx_sorted]

                t1 = time.perf_counter() if args.time_include_post else t_cut

        elif args.method == 'entropy':
            prob = torch.softmax(logits_full_bnc, dim=-1)
            ent = -(prob * (prob.clamp_min(1e-9).log())).sum(dim=-1)  # (1,N)
            K = min(args.keep, ent.shape[1])
            idx_sorted = torch.argsort(ent[0], descending=False)[:K]
            kept_counts.append(int(K))

            noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
            remain_noise_counts.append(noise_kept)

            if SIMPLE_HDD:
                pred_full = _predict_local_argmax(logits_full_bnc)[0]   # or logits_kept_bnc for the kept one
                pred_kept = pred_full[idx_sorted]                       # when needed
            else:
                # existing call
                pred_full = predict_labels_with_gt(
                    logits_full_bnc, label_B, gt_n,
                    mask_allowed=args.mask_allowed, num_part=args.num_part
                )[0]
                pred_kept = pred_full[idx_sorted]



            t1 = time.perf_counter() if args.time_include_post else t_cut

        else:  # sngp
            valid_parts_bc = None if SIMPLE_HDD else (build_valid_parts(label_B) if args.mask_allowed else None)
            unc_bn = extract_uncertainty(
                aux, logits_full_bnc,
                valid_parts_bc=valid_parts_bc,
                alpha_hybrid=getattr(args, 'alpha_hybrid', 1),
            )
            K = min(args.keep, unc_bn.shape[1])
            idx_sorted = torch.argsort(unc_bn[0], descending=False)[:K]
            kept_counts.append(int(K))

            noise_kept = int((gt_n[idx_sorted] == -1).sum().item())
            remain_noise_counts.append(noise_kept)

            if SIMPLE_HDD:
                pred_full = _predict_local_argmax(logits_full_bnc)[0]   # or logits_kept_bnc for the kept one
                pred_kept = pred_full[idx_sorted]                       # when needed
            else:
                # existing call
                pred_full = predict_labels_with_gt(
                    logits_full_bnc, label_B, gt_n,
                    mask_allowed=args.mask_allowed, num_part=args.num_part
                )[0]
                pred_kept = pred_full[idx_sorted]


            t1 = time.perf_counter() if args.time_include_post else t_cut
            C = args.num_part  # 例如 5
            inst_mious_all_penal.append(miou_local(pred_full, gt_n, C))
            # inst_mious_kept_penal.append(miou_local(pred_kept, gt_kept, C))


        # IoU accumulation on kept indices (noise counted as wrong)
        gt_kept = gt_n[idx_sorted]

        if SIMPLE_HDD:
            # Evaluate in local label space 0..num_part-1
            ious = []
            for l in range(args.num_part):
                pc = (pred_kept == l)
                gc = (gt_kept == l)
                tp = (pc & gc).sum().item()
                fp = (pc & (~gc)).sum().item()  # includes gt==-1 as wrong
                fn = ((~pc) & gc).sum().item()
                denom = tp + fp + fn
                ious.append(1.0 if denom == 0 else (tp / float(denom)))
            shape_iou = sum(ious) / len(ious)
            inst_mious.append(shape_iou)
            # If you want "class mIoU dataset-avg" like v6, just collect under a single key:
            if 'HDD' not in per_cat_shape_ious:
                per_cat_shape_ious['HDD'] = []
            per_cat_shape_ious['HDD'].append(shape_iou)
        else:
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


    # lat_ms_mid = lat_ms[150:350] 
    # # Compute latency statistics for that middle range
    # lat_mean = mean(lat_ms_mid)
    # lat_p50 = percentile(lat_ms_mid, 50)
    # lat_p90 = percentile(lat_ms_mid, 90)
    # lat_p95 = percentile(lat_ms_mid, 95)
    # lat_std = pstdev(lat_ms_mid)

    avg_noise_kept = (sum(remain_noise_counts)/len(remain_noise_counts)) if remain_noise_counts else 0.0
    n = len(lat_ms)

    # print("[SUMMARY]")
    # print(f"latency Samples evaluated: 150-350, noise eval set size={n}")
    # print(f"Latency ms per-sample | mean={lat_mean:.2f} | p50={lat_p50:.2f} | p90={lat_p90:.2f} | p95={lat_p95:.2f} | std={lat_std:.2f}")
    print(f"Avg noise count in kept-{args.keep}: {avg_noise_kept:.2f}")
    inst_miou_kept = sum(inst_mious)/len(inst_mious)
    cat_means = [sum(v)/len(v) for v in per_cat_shape_ious.values() if len(v)]
    class_miou_kept_dataset_avg = (sum(cat_means)/len(cat_means)) if cat_means else 0.0

    print(f"Instance mIoU (kept-only, official-style): {inst_miou_kept:.4f}")
    print(f"Class mIoU (kept-only, dataset-avg): {class_miou_kept_dataset_avg:.4f}")
    def avg(xs): return sum(xs)/len(xs) if xs else 0.0
    print(f"Instance mIoU (ALL points, penalize -1): {avg(inst_mious_all_penal):.4f}")
    # print(f"Instance mIoU (kept-only, penalize -1): {avg(inst_mious_kept_penal):.4f}")


if __name__ == '__main__':
    main()
