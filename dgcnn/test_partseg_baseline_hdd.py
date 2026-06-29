#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from tqdm import tqdm

from pointnet2.data_utils.ShapeNetDataLoader import PartNormalDataset
from dgcnn.model_hdd import DGCNN_partseg
from dgcnn.util import cal_loss

import numpy as np
import torch

def calc_shape_iou_penalized(pred, gt, parts=None):
    """
    pred, gt: 1D arrays (N,)
      - gt uses -1 to mark noise points
    parts: iterable of valid part IDs for this shape/category.
      If None, we derive a conservative set from labeled points.

    For each class c in `parts`, we compute:
      IoU_c = TP_c / (TP_c + FP_c + FN_c + FP_noise_c)
    where FP_noise_c counts predictions of c on noise points (gt == -1).
    """
    pr = np.asarray(pred, dtype=np.int64).copy()
    gt = np.asarray(gt,   dtype=np.int64).copy()

    # Build a reasonable parts set if caller didn't pass one.
    if parts is None:
        # use classes that appear in labeled gt and labeled pred;
        # this mimics the per-shape eval while avoiding counting
        # totally irrelevant global classes.
        labeled = (gt != -1)
        parts = sorted(set(gt[labeled].tolist()) | set(pr[labeled].tolist()))

    ious = []
    for c in parts:
        gt_c = (gt == c)
        pr_c = (pr == c)

        tp = (gt_c & pr_c).sum()
        fp = (~gt_c & (gt != -1) & pr_c).sum()   # FP on labeled (non-noise) points
        fn = (gt_c & ~pr_c).sum()

        # extra penalty: predictions of c on noise points
        fp_noise = ((gt == -1) & pr_c).sum()

        denom = tp + fp + fn + fp_noise
        if denom == 0:
            # Neither GT nor PR (and no noise predicted as c): perfect by convention
            iou_c = 1.0
        else:
            iou_c = float(tp) / float(denom)
        ious.append(iou_c)

    return float(np.mean(ious)) if len(ious) else 1.0

# seg_classes = {
#     'Airplane':  [0,1,2,3], 'Bag':[4,5], 'Cap':[6,7], 'Car':[8,9,10,11],
#     'Chair':[12,13,14,15], 'Earphone':[16,17,18], 'Guitar':[19,20,21],
#     'Knife':[22,23], 'Lamp':[24,25,26,27], 'Laptop':[28,29],
#     'Motorbike':[30,31,32,33,34,35], 'Mug':[36,37], 'Pistol':[38,39,40],
#     'Rocket':[41,42,43], 'Skateboard':[44,45,46], 'Table':[47,48,49]
# }
seg_classes = {
    'hdd':  [0,1,2,3,4]
}
classes_str = list(seg_classes.keys())


def parse_args():
    p = argparse.ArgumentParser("Noisy PartSeg Eval")
    p.add_argument('--root', type=str, default=r'data_prepare\hdd_data' )
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--num_points', type=int, default=2048)
    p.add_argument('--normal', action='store_true')
    # p.add_argument('--ckpt', type=str, default=r'dgcnn\pretrained\model.partseg.t7')
    p.add_argument('--ckpt', type=str, default=r'dgcnn\pretrained\model.t7')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=str, default='0')
    return p.parse_args()


def calc_shape_iou(pred, gt, ignore_index=-1):
    """Compute IoU for a single shape, ignoring -1."""
    mask = (gt != ignore_index)
    gt, pr = gt[mask], pred[mask]
    if gt.size == 0:
        return 0.0
    parts = np.unique(gt)
    part_ious = []
    for c in parts:
        gt_c, pr_c = (gt == c), (pr == c)
        I, U = np.sum(gt_c & pr_c), np.sum(gt_c | pr_c)
        part_ious.append(1.0 if U == 0 else I/float(U))
    return float(np.mean(part_ious)) if part_ious else 0.0


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PartNormalDataset(root=args.root, npoints=args.num_points,
                                split=args.split, normal_channel=args.normal)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # seg_num_all = 50
    seg_num_all = 5
    class ModelArgs: pass
    margs = ModelArgs()
    margs.emb_dims, margs.k, margs.dropout = 1024, 40, 0.5

    model = DGCNN_partseg(margs, seg_num_all).to(device)
    model = nn.DataParallel(model)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()

    all_true_cls, all_pred_cls = [], []
    inst_ious = []                                  # collects per-shape IoU
    class_ious = {cls: [] for cls in classes_str}   # collects per-class IoUs
    total_pts, total_noise = 0, 0                            # per-shape IoU (ignore -1)
    inst_ious_penalized = []                        # per-shape IoU (noise-penalized)

    print(f"[INFO] Evaluating {len(dataset)} shapes ...")

    with torch.no_grad():
        for data, label, seg in tqdm(loader, desc="Testing", ncols=100):
            B, N = seg.shape
            one_hot = np.zeros((B, 16), dtype=np.float32)
            for i in range(B):
                one_hot[i, int(label[i, 0])] = 1
            one_hot = torch.from_numpy(one_hot)

            data = data.to(device).permute(0, 2, 1)
            one_hot = one_hot.to(device)
            seg = seg.to(device)

            seg_pred = model(data, one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            pred = seg_pred.argmax(dim=2)

            noise_mask = (seg == -1)
            total_pts += B * N
            total_noise += int(noise_mask.sum().item())

            seg_np = seg.cpu().numpy()
            pred_np = pred.cpu().numpy()

            seg_overall = seg_np.copy()
            seg_overall[seg_overall == -1] = 999
            all_true_cls.append(seg_overall.reshape(-1))
            all_pred_cls.append(pred_np.reshape(-1))

            # per-shape IoU
            for b in range(B):
                gt_b   = seg_np[b]
                pred_b = pred_np[b]
                class_name = classes_str[int(label[b, 0])]

                # Standard IoU (ignores -1)
                iou_clean = calc_shape_iou(pred_b, gt_b, ignore_index=-1)

                # New noise-penalized IoU
                iou_pen = calc_shape_iou_penalized(pred_b, gt_b)

                inst_ious.append(iou_clean)
                class_ious[class_name].append(iou_clean)

                # Collect penalized IoU separately for reporting
                inst_ious_penalized.append(iou_pen)

    # OA
    all_true_cls = np.concatenate(all_true_cls)
    all_pred_cls = np.concatenate(all_pred_cls)
    overall_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)

    # Clean Acc
    mask_clean = (all_true_cls != 999)
    clean_acc = metrics.accuracy_score(all_true_cls[mask_clean], all_pred_cls[mask_clean])

    # Instance mIoU
    instance_miou = np.mean(inst_ious)
    instance_miou_penalized = np.mean(inst_ious_penalized)

    # Class mIoU
    class_miou = {cls: (np.mean(v) if v else None) for cls, v in class_ious.items()}
    avg_class_miou = np.mean([v for v in class_miou.values() if v is not None])

    print("=================================================")
    print(f"Overall Acc (noise wrong): {overall_acc:.4f}")
    print(f"Instance mIoU (noise-penalized): {instance_miou_penalized:.4f}")
    print(f"Class mIoU (avg):          {avg_class_miou:.4f}")
    print(f"Noise ratio: {total_noise}/{total_pts} ({100*total_noise/total_pts:.2f}%)")
    print("=================================================")


if __name__ == "__main__":
    main()
