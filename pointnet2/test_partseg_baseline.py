import argparse
import os
from pointnet2.data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, default='pointnet2\log\without_normal', help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--root', type=str, default='data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', help='dataset root')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'pointnet2/log/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)


    TEST_DATASET = PartNormalDataset(root=args.root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])

# --- BEGIN PATCH: robust eval that penalizes -1 and avoids KeyError ---
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        num_part = 50  # keep your original
        shape_ious = {cat: [] for cat in seg_classes.keys()}

        # seg_label_to_cat already defined globally; rebuild here if needed
        seg_label_to_cat = {}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier.eval()

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader),
                                                    total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            # Forward
            seg_pred, _ = classifier(points, to_categorical(label, num_classes))
            # Expect seg_pred shape [B, N, num_part]; keep consistent with your original script
            cur_pred_val_logits = seg_pred.detach().cpu().numpy()  # [B, N, 50]
            target_np = target.detach().cpu().numpy()              # [B, N]
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT), dtype=np.int32)

            for i in range(cur_batch_size):
                # ---- Safely determine category; avoid KeyError when first label is -1 ----
                first_label = int(target_np[i, 0])
                cat = None
                if first_label != -1:
                    cat = seg_label_to_cat[first_label]
                else:
                    # Try to find the first non -1 label in this shape
                    valid_idxs = np.where(target_np[i] != -1)[0]
                    if valid_idxs.size > 0:
                        cat = seg_label_to_cat[int(target_np[i, valid_idxs[0]])]
                    else:
                        cat = None  # all labels are -1; no valid category

                logits_i = cur_pred_val_logits[i]  # [N, 50]

                if cat is not None:
                    # ---- Official class-mask: restrict logits to this category's parts ----
                    allowed = np.array(seg_classes[cat], dtype=np.int64)          # e.g., [34,35,36,37]
                    logits_allowed = logits_i[:, allowed]                         # [N, num_parts_cat]
                    pred_allowed = np.argmax(logits_allowed, axis=1)              # [N] in [0..num_parts_cat-1]
                    cur_pred_val[i, :] = allowed[pred_allowed]                    # map back to global part ids
                else:
                    # No valid category (all -1). Argmax over all parts (still penalized later).
                    cur_pred_val[i, :] = np.argmax(logits_i, axis=1)

            # ---- Overall accuracy: count -1 as wrong (pred ∈ [0..49], never equals -1) ----
            correct = np.sum(cur_pred_val == target_np)  # -1 never matches any pred, so they are counted as wrong
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            # ---- Per-shape IoU (penalize -1 shapes with IoU=0.0) ----
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]    # predictions [N]
                segl = target_np[i, :]       # targets [N] (may contain -1)

                # determine category for IoU the same safe way
                first_label = int(segl[0])
                cat = None
                if first_label != -1:
                    cat = seg_label_to_cat[first_label]
                else:
                    valid_idxs = np.where(segl != -1)[0]
                    if valid_idxs.size > 0:
                        cat = seg_label_to_cat[int(segl[valid_idxs[0]])]
                    else:
                        cat = None

                if cat is None:
                    # all points are -1; penalize this shape IoU as 0.0
                    shape_ious.setdefault('__ALL_NEG__', [])
                    shape_ious['__ALL_NEG__'].append(0.0)
                    continue

                parts = seg_classes[cat]
                part_ious = []
                for l in parts:
                    # standard IoU over valid part id l; -1 contributes to neither side
                    tp = np.sum((segl == l) & (segp == l))
                    fp = np.sum((segl != l) & (segp == l))
                    fn = np.sum((segl == l) & (segp != l))
                    denom = tp + fp + fn
                    iou_l = 1.0 if denom == 0 else (tp / float(denom))
                    part_ious.append(iou_l)
                shape_ious[cat].append(float(np.mean(part_ious)))

        # ---- Aggregate metrics ----
        # Flatten all per-shape IoUs
        all_shape_ious = []
        for cat, vals in shape_ious.items():
            if len(vals) > 0 and cat in seg_classes:  # only real categories for category mIoU
                shape_ious[cat] = float(np.mean(vals))
            all_shape_ious.extend(vals)

        class_avg_iou = float(np.mean([v for k, v in shape_ious.items() if k in seg_classes]))
        instance_avg_iou = float(np.mean(all_shape_ious)) if len(all_shape_ious) > 0 else 0.0
        accuracy = total_correct / float(total_seen)

        # --- END PATCH ---
    test_metrics = {
    'accuracy': accuracy,
    'class_avg_iou': class_avg_iou,
    'instance_avg_iou': instance_avg_iou,}


    log_string('========================================')
    log_string(f'Overall Acc : {test_metrics["accuracy"]:.4f}')
    log_string(f'Instance mIoU :  {test_metrics["instance_avg_iou"]:.4f}')
    log_string(f'Class mIoU :  {test_metrics["class_avg_iou"]:.4f}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
