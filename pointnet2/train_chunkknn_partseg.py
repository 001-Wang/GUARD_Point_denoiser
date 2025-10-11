import argparse, os, sys, shutil, datetime, logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import importlib

# ==== Import-safe paths ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If this file is placed under .../models/, ROOT_DIR becomes the project root
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'models' else BASE_DIR
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'models'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'data_utils'))

# ==== Data ====
from data_utils.ShapeNetDataLoader import PartNormalDataset
import provider

# ======= Category setup (default Airplane 5 parts) =======
SEG_CLASSES = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

SEG_LABEL_TO_CAT = {}
for cat, labels in SEG_CLASSES.items():
    for lb in labels:
        SEG_LABEL_TO_CAT[lb] = cat

# ----------------- Helpers -----------------
def inplace_relu(m):
    if isinstance(m, nn.ReLU):
        m.inplace = True

def bn_momentum_adjust(m, momentum):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.momentum = momentum

def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        if getattr(m, "weight", None) is not None:
            nn.init.xavier_normal_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0.0)

class SegLoss(nn.Module):
    def __init__(self, num_part):
        super().__init__()
        self.num_part = num_part
    def forward(self, logits, target, weight=None, is_log_softmax=False):
        """
        logits: (B*N, C). If is_log_softmax=True, use NLL; else CE.
        """
        if is_log_softmax:
            return F.nll_loss(logits, target, weight=weight)
        else:
            return F.cross_entropy(logits, target, weight=weight)

# ---------- Arg parsing ----------
def parse_comma_list_int(s):
    return tuple(int(x) for x in s.split(',') if x.strip()!='')

def parse_comma_list_float(s):
    return tuple(float(x) for x in s.split(',') if x.strip()!='')

def parse_args():
    p = argparse.ArgumentParser("Train PartSeg PlanA-FIX with Multi-Scale RFF + Chunked-KNN")
    # I/O & runtime
    p.add_argument('--model', type=str, default='sngp_s2', help='model module name under models/')
    p.add_argument('--data_root', type=str, default=r'data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                   help='Path to ShapeNetPart root (portable; override Windows path)')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epoch', type=int, default=120)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--optimizer', type=str, default='Adam', choices=['Adam','SGD'])
    p.add_argument('--log_dir', type=str, default=None)
    p.add_argument('--decay_rate', type=float, default=1e-4)
    p.add_argument('--npoint', type=int, default=2048)
    p.add_argument('--normal', action='store_true', default=False)
    p.add_argument('--step_size', type=int, default=20)
    p.add_argument('--lr_decay', type=float, default=0.5)
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--amp', action='store_true', help='use mixed precision (fp16/bf16)')
    p.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume')
    p.add_argument('--save_best', action='store_true', help='keep best.ckpt when Test Acc improves')

    # Geometric transformer & RFF-GP config
    p.add_argument('--ct', type=int, default=128, help='Transformer feature dim C_t')
    p.add_argument('--tf_layers', type=int, default=1)
    p.add_argument('--tf_heads', type=int, default=4)
    p.add_argument('--tf_dropout', type=float, default=0.1)
    p.add_argument('--sn_first', action='store_true', default=True)

    # Multi-scale neighborhood (Chunked-KNN)
    p.add_argument('--ms_k', type=str, default='16,32,64', help='multi-scale k, comma separated')
    p.add_argument('--per_scale_width', type=int, default=32)
    p.add_argument('--chunk_size', type=int, default=256, help='knn chunk size for memory saving')

    # Multi-band RFF (early geo GP)
    p.add_argument('--rff_sigmas', type=str, default='0.5,1.0,2.0', help='RFF sigma list')
    p.add_argument('--rff_m_each', type=int, default=128)
    p.add_argument('--rff_ridge', type=float, default=5.0)

    # Final semantic SNGP
    p.add_argument('--final_num_rff', type=int, default=1024)
    p.add_argument('--final_ridge', type=float, default=1.0)
    return p.parse_args()



def evaluate_iou(logits_bnc, target_bn, seg_label_to_cat, seg_classes):
    """
    logits_bnc: (B,N,C) raw logits or log-probs
    target_bn:  (B,N)
    returns: test_acc, class_mIoU, inst_mIoU
    """
    with torch.no_grad():
        B, N, C = logits_bnc.shape
        pred = logits_bnc.argmax(dim=-1)  # (B,N)
        correct = (pred == target_bn).sum().item()
        total = B * N

        # IoU per instance and class averaging (like planA_FIX)
        pred_np = pred.cpu().numpy()
        target_np = target_bn.cpu().numpy()

        shape_ious = {cat: [] for cat in seg_classes.keys()}
        for i in range(B):
            segp = pred_np[i, :]
            segl = target_np[i, :]
            cat = seg_label_to_cat[segl[0]]  # assume single-category per shape
            part_ious = []
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                    part_ious.append(1.0)
                else:
                    inter = np.sum((segl == l) & (segp == l))
                    union = np.sum((segl == l) | (segp == l))
                    part_ious.append(float(inter) / float(union + 1e-8))
            shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = [iou for cat in shape_ious for iou in shape_ious[cat]]
        class_avg_iou = float(np.mean([np.mean(shape_ious[cat]) for cat in shape_ious])) if all_shape_ious else 0.0
        inst_avg_iou  = float(np.mean(all_shape_ious)) if all_shape_ious else 0.0

        return correct / (total + 1e-8), class_avg_iou, inst_avg_iou


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- logging dirs -----
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    exp_dir = Path(ROOT_DIR).joinpath('log', f'part_seg_sngp'); exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_dir.joinpath(args.log_dir if args.log_dir is not None else timestr); exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints'); checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs'); log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Train-PartSeg-PlanA(mscale+chunkknn)"); logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_dir / 'train.txt')); fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')); logger.addHandler(fh)
    def log_string(s): logger.info(s); print(s)

    # ----- data -----
    # ----- data -----
    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    TEST_DATASET  = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                               num_workers=8, drop_last=True, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(TEST_DATASET,  batch_size=args.batch_size, shuffle=False,
                                               num_workers=8, pin_memory=True)

    num_part = 50


    # ----- model -----
    try:
        MODEL = importlib.import_module(args.model)
    except ModuleNotFoundError:
        MODEL = importlib.import_module(f'models.{args.model}')

    # backup model sources (like planA_FIX)
    src_model_path = Path('models')/f'{args.model}.py'
    if src_model_path.exists():
        shutil.copy(str(src_model_path), str(exp_dir))
    for extra in ['pointnet2_utils.py']:
        pth = Path('models')/extra
        if pth.exists():
            shutil.copy(str(pth), str(exp_dir))

    # Build geom_cfg for the model
    geom_cfg = dict(
        embed_dim=args.ct,
        num_layers=args.tf_layers,
        num_heads=args.tf_heads,
        dropout=args.tf_dropout,
        sn_first=args.sn_first,
        ms_k=parse_comma_list_int(args.ms_k),
        per_scale_width=args.per_scale_width,
        chunk_size=args.chunk_size,
        rff_sigmas=parse_comma_list_float(args.rff_sigmas),
        rff_m_each=args.rff_m_each,
        rff_ridge=args.rff_ridge,
    )
    final_sngp_cfg = dict(
        enabled=True,
        num_rff=args.final_num_rff,
        ridge=args.final_ridge,
        spectral_beta=True,
    )

    # Build on CPU, init, then move to device (keeps SN buffers consistent)
    model = MODEL.get_model(num_classes=num_part, geom_cfg=geom_cfg, final_sngp_cfg=final_sngp_cfg)
    model.apply(inplace_relu); model.apply(weights_init); model = model.to(device)
    log_string(f"Model: {args.model} | Device: {device} ")

    # loss & opt
    criterion = SegLoss(num_part=num_part).to(device)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate, weight_decay=args.decay_rate, betas=(0.9, 0.999), eps=1e-8
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)

    scaler = GradScaler(enabled=args.amp)

    # resume
    start_epoch, best_acc = 0, 0.0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state.get('model_state_dict', state))
        if 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = int(state.get('epoch', 0))
        best_acc = float(state.get('best_acc', 0.0))
        log_string(f"[Resume] epoch={start_epoch}, best_acc={best_acc:.4f} from {args.resume}")

    # schedulers (PlanA-FIX style)
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_cls_iou = 0.0
    best_inst_iou = 0.0

    for epoch in range(start_epoch, args.epoch):
        # adjust lr & bn momentum
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for pg in optimizer.param_groups: pg['lr'] = lr
        momentum = max(MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP)), 0.01)
        model.apply(lambda m: bn_momentum_adjust(m, momentum))
        log_string(f'Epoch {epoch+1}/{args.epoch} | lr={lr:.6f} | bn_mom={momentum:.4f} | amp={args.amp}')

        # ---- Train ----
        model.train()
        mean_correct = []

        for i, (points, label, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            optimizer.zero_grad(set_to_none=True)

            # augment xyz (planA_FIX)
            points_np = points.numpy()
            points_np[:, :, 0:3] = provider.random_scale_point_cloud(points_np[:, :, 0:3])
            points_np[:, :, 0:3] = provider.shift_point_cloud(points_np[:, :, 0:3])
            points = torch.tensor(points_np).float().to(device)
            target = target.long().to(device)

            # to (B,3,N)
            points = points.transpose(2, 1).contiguous()

            with autocast(enabled=args.amp):
                baseline_ll, sngp_logits, aux = model(points, return_point_feats=True, compute_sigma=False)
                if sngp_logits is not None:  # raw logits
                    seg_pred = sngp_logits.reshape(-1, sngp_logits.size(-1))
                    loss = criterion(seg_pred, target.view(-1), is_log_softmax=False)
                else:  # log-softmax
                    seg_pred = baseline_ll.reshape(-1, baseline_ll.size(-1))
                    loss = criterion(seg_pred, target.view(-1), is_log_softmax=True)
                pred_choice = seg_pred.argmax(dim=1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                correct = (pred_choice == target.view(-1)).float().mean().item()
                mean_correct.append(correct)

        train_acc = float(np.mean(mean_correct)) if mean_correct else 0.0
        log_string(f'Train acc: {train_acc:.5f}')

        # ---- Eval ----
        model.eval()
        total_correct = 0
        total_seen = 0
        class_miou_meter = []
        inst_miou_meter = []

        with torch.no_grad():
            for batch_id, (points, label, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                points = points.float().to(device)
                target = target.long().to(device)
                B, NUM_POINT, _ = points.shape

                points = points.transpose(2, 1).contiguous()
                with autocast(enabled=args.amp):
                    baseline_ll, sngp_logits, aux = model(points, return_point_feats=True, compute_sigma=True)
                    logits_eval = sngp_logits if sngp_logits is not None else baseline_ll

                # accuracy
                pred = logits_eval.argmax(dim=-1)
                total_correct += (pred == target).sum().item()
                total_seen += (B * NUM_POINT)

                # IoU metrics (planA_FIX style)
                acc, class_miou, inst_miou = evaluate_iou(logits_eval, target, SEG_LABEL_TO_CAT, SEG_CLASSES)
                class_miou_meter.append(class_miou)
                inst_miou_meter.append(inst_miou)

        test_acc = float(total_correct) / float(total_seen + 1e-8)
        class_miou = float(np.mean(class_miou_meter)) if class_miou_meter else 0.0
        inst_miou  = float(np.mean(inst_miou_meter))  if inst_miou_meter else 0.0
        log_string(f'Epoch {epoch+1} | Test Acc: {test_acc:.4f} | Class mIoU: {class_miou:.4f} | Inst mIoU: {inst_miou:.4f} | Best Acc: {max(best_acc, test_acc):.4f}')

        # update bests
        improved = test_acc > best_acc
        best_acc = max(best_acc, test_acc)
        best_cls_iou = max(best_cls_iou, class_miou)
        best_inst_iou = max(best_inst_iou, inst_miou)

        # save periodic
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epoch or improved:
            savepath = checkpoints_dir / f'{args.model}_epoch{epoch+1:03d}.pth'
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_cls_iou': best_cls_iou,
                'best_inst_iou': best_inst_iou,
                'args': vars(args)
            }
            torch.save(state, savepath)
            log_string(f"Saved checkpoint: {savepath}")
            if args.save_best and improved:
                best_path = checkpoints_dir / 'best.ckpt'
                shutil.copy(str(savepath), str(best_path))
                log_string(f"[Best] Updated {best_path} (acc={best_acc:.4f})")

if __name__ == '__main__':
    main()
