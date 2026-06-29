# train_scannet_semseg.py
import argparse, os, sys, datetime, logging, shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "models"))
sys.path.insert(0, os.path.join(ROOT_DIR, "data_utils"))

from data_utils.ScanNetDataLoader_block import ScanNetPTHBlockDataset




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


def parse_comma_list_int(s):
    return tuple(int(x) for x in s.split(",") if x.strip() != "")

def parse_comma_list_float(s):
    return tuple(float(x) for x in s.split(",") if x.strip() != "")


def compute_miou(pred_bn, target_bn, num_classes, ignore_index=-1):
    """
    pred_bn:   (B,N) predicted class ids
    target_bn: (B,N) ground truth class ids
    returns: (miou, per_class_iou np.array)
    """
    pred = pred_bn.reshape(-1).detach().cpu().numpy()
    tgt  = target_bn.reshape(-1).detach().cpu().numpy()

    mask = (tgt != ignore_index)
    pred = pred[mask]
    tgt  = tgt[mask]

    ious = np.zeros((num_classes,), dtype=np.float64)
    valid = np.zeros((num_classes,), dtype=np.float64)

    for c in range(num_classes):
        p = (pred == c)
        g = (tgt == c)
        inter = np.sum(p & g)
        union = np.sum(p | g)
        if union > 0:
            ious[c] = inter / (union + 1e-8)
            valid[c] = 1.0
        else:
            ious[c] = 0.0

    miou = float((ious * valid).sum() / (valid.sum() + 1e-8))
    return miou, ious


def build_class_weights(dataset, num_classes, ignore_index=-1, max_scenes_scan=9999):
    """
    Compute inverse-frequency weights by scanning some scenes quickly.
    """
    counts = torch.zeros(num_classes, dtype=torch.long)
    n = min(len(dataset.files), max_scenes_scan)
    for i in range(n):
        scene = dataset._load_scene(dataset.files[i])
        y = scene["labels"].view(-1)
        if ignore_index is not None:
            y = y[y != ignore_index]
        y = y[(y >= 0) & (y < num_classes)]
        if y.numel() > 0:
            counts += torch.bincount(y, minlength=num_classes)

    w = 1.0 / (counts.float() + 1.0)  # +1 to avoid inf
    w = w / w.mean().clamp_min(1e-6)
    return w


def parse_args():
    p = argparse.ArgumentParser("Train ScanNet semantic seg with PointNet++ + SNGP")
    p.add_argument("--model", type=str, default="sngp_s2_6layers", help="models/<model>.py module name")
    p.add_argument("--data_root", type=str, required=True, help="ScanNet .pth root (contains train/val/test or flat)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--decay_rate", type=float, default=1e-4)
    p.add_argument("--npoint", type=int, default=8192)
    p.add_argument("--step_size", type=int, default=20)
    p.add_argument("--lr_decay", type=float, default=0.5)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--ignore_index", type=int, default=-1)
    p.add_argument("--use_rgb", action="store_true", help="use xyzrgb as input; model still uses xyz only by default")
    p.add_argument("--normalize_xyz", action="store_true")
    p.add_argument("--repeat", type=int, default=50, help="samples per scene per epoch")

    # your geom + sngp configs (same as your ShapeNet script)
    p.add_argument("--ct", type=int, default=128)
    p.add_argument("--tf_layers", type=int, default=1)
    p.add_argument("--tf_heads", type=int, default=4)
    p.add_argument("--tf_dropout", type=float, default=0.1)
    p.add_argument("--sn_first", action="store_true", default=True)

    p.add_argument("--ms_k", type=str, default="16,32,64")
    p.add_argument("--per_scale_width", type=int, default=32)
    p.add_argument("--chunk_size", type=int, default=256)

    p.add_argument("--rff_sigmas", type=str, default="0.3,0.7,1.4")
    p.add_argument("--rff_m_each", type=int, default=128)
    p.add_argument("--rff_ridge", type=float, default=5.0)

    p.add_argument("--final_num_rff", type=int, default=1024)
    p.add_argument("--final_ridge", type=float, default=1.0)
    
    p.add_argument("--block_size", type=float, default=1.5)
    p.add_argument("--min_points_in_block", type=int, default=1024)
    p.add_argument("--blocks_per_scene", type=int, default=1, help="K blocks per scene per __getitem__")

    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = Path(ROOT_DIR).joinpath("log", "scannet_semseg_sngp")
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_dir.joinpath(args.log_dir if args.log_dir is not None else timestr)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir.joinpath("checkpoints"); checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath("logs"); log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Train-ScanNet-SemSeg"); logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_dir / "train.txt")); fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    def log_string(s): logger.info(s); print(s)

    # dataset
    train_set = ScanNetPTHBlockDataset(
        root=args.data_root, split="train",
        npoints=args.npoint,
        block_size=args.block_size,
        min_points_in_block=args.min_points_in_block,
        use_rgb=args.use_rgb,
        normalize_xyz=args.normalize_xyz,
        ignore_index=args.ignore_index,
        repeat=args.repeat,
        blocks_per_scene=args.blocks_per_scene,
        seed=0
    )
    val_set = ScanNetPTHBlockDataset(
        root=args.data_root, split="val",
        npoints=args.npoint,
        block_size=args.block_size,
        min_points_in_block=args.min_points_in_block,
        use_rgb=args.use_rgb,
        normalize_xyz=args.normalize_xyz,
        ignore_index=args.ignore_index,
        repeat=max(1, args.repeat // 5),
        blocks_per_scene=args.blocks_per_scene,  # you can set to 1 for val if you prefer
        seed=1
    )


    # build class weights (optional but helps ScanNet imbalance)
    class_w = build_class_weights(train_set, args.num_classes, ignore_index=args.ignore_index).to(device)
    log_string(f"class weights (mean=1): {class_w.detach().cpu().numpy()}")

    # model import
    try:
        MODEL = importlib.import_module(args.model)
    except ModuleNotFoundError:
        MODEL = importlib.import_module(f"models.{args.model}")

    # backup sources
    src_model_path = Path("models") / f"{args.model}.py"
    if src_model_path.exists():
        shutil.copy(str(src_model_path), str(exp_dir))
    for extra in ["pointnet2_utils.py"]:
        pth = Path("models") / extra
        if pth.exists():
            shutil.copy(str(pth), str(exp_dir))

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

    model = MODEL.get_model(num_classes=args.num_classes, geom_cfg=geom_cfg, final_sngp_cfg=final_sngp_cfg)
    model.apply(inplace_relu); model.apply(weights_init); model = model.to(device)
    log_string(f"Model: {args.model} | device={device} | num_classes={args.num_classes}")

    # optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate, weight_decay=args.decay_rate, betas=(0.9, 0.999), eps=1e-8
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate
        )

    scaler = GradScaler(enabled=args.amp)

    # resume
    start_epoch, best_miou = 0, 0.0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state.get("epoch", 0))
        best_miou = float(state.get("best_miou", 0.0))
        log_string(f"[Resume] epoch={start_epoch}, best_miou={best_miou:.4f} from {args.resume}")

    # schedule constants
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    for epoch in range(start_epoch, args.epoch):
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        momentum = max(MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP)), 0.01)
        model.apply(lambda m: bn_momentum_adjust(m, momentum))
        log_string(f"Epoch {epoch+1}/{args.epoch} | lr={lr:.6f} | bn_mom={momentum:.4f} | amp={args.amp}")

        # ---- train ----
        model.train()
        train_loss_meter = []
        train_acc_meter = []

        for _, (pts, y, scene_name) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            optimizer.zero_grad(set_to_none=True)

            # pts: (B, N, C). Your model expects xyz as (B,3,N).
            pts = pts.float().to(device)
            y = y.long().to(device)

            xyz = pts[:, :, 0:3].transpose(2, 1).contiguous()  # (B,3,N)

            with autocast(enabled=args.amp):
                baseline_ll, sngp_logits, aux = model(xyz, return_point_feats=True, compute_sigma=False)

                logits = sngp_logits if sngp_logits is not None else baseline_ll  # (B,N,C) or log-softmax
                # make (B*N, C)
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_flat = y.reshape(-1)

                # ignore index
                if args.ignore_index is not None:
                    valid = (y_flat != args.ignore_index)
                    logits_flat = logits_flat[valid]
                    y_flat = y_flat[valid]

                # If using baseline_ll (log-softmax), use NLL; else CE
                if sngp_logits is None:
                    loss = F.nll_loss(logits_flat, y_flat, weight=class_w)
                else:
                    loss = F.cross_entropy(logits_flat, y_flat, weight=class_w)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = logits_flat.argmax(dim=1)
                acc = (pred == y_flat).float().mean().item() if y_flat.numel() > 0 else 0.0
                train_loss_meter.append(float(loss.item()))
                train_acc_meter.append(float(acc))

        log_string(f"Train | loss={np.mean(train_loss_meter):.4f} | acc={np.mean(train_acc_meter):.4f}")

        # ---- val ----
        model.eval()
        val_loss_meter = []
        val_acc_meter = []
        val_miou_meter = []

        with torch.no_grad():
            for _, (pts, y, scene_name) in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
                pts = pts.float().to(device)
                y = y.long().to(device)
                xyz = pts[:, :, 0:3].transpose(2, 1).contiguous()

                with autocast(enabled=args.amp):
                    baseline_ll, sngp_logits, aux = model(xyz, return_point_feats=True, compute_sigma=True)
                    logits = sngp_logits if sngp_logits is not None else baseline_ll

                # loss
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_flat = y.reshape(-1)

                if args.ignore_index is not None:
                    valid = (y_flat != args.ignore_index)
                    logits_flat_v = logits_flat[valid]
                    y_flat_v = y_flat[valid]
                else:
                    logits_flat_v = logits_flat
                    y_flat_v = y_flat

                if sngp_logits is None:
                    loss = F.nll_loss(logits_flat_v, y_flat_v, weight=class_w)
                else:
                    loss = F.cross_entropy(logits_flat_v, y_flat_v, weight=class_w)

                pred_bn = logits.argmax(dim=-1)  # (B,N)
                miou, _ = compute_miou(pred_bn, y, args.num_classes, ignore_index=args.ignore_index)

                # acc
                pred_flat = logits_flat_v.argmax(dim=1)
                acc = (pred_flat == y_flat_v).float().mean().item() if y_flat_v.numel() > 0 else 0.0

                val_loss_meter.append(float(loss.item()))
                val_acc_meter.append(float(acc))
                val_miou_meter.append(float(miou))

        val_loss = float(np.mean(val_loss_meter)) if val_loss_meter else 0.0
        val_acc = float(np.mean(val_acc_meter)) if val_acc_meter else 0.0
        val_miou = float(np.mean(val_miou_meter)) if val_miou_meter else 0.0
        log_string(f"Val   | loss={val_loss:.4f} | acc={val_acc:.4f} | mIoU={val_miou:.4f} | best_mIoU={max(best_miou,val_miou):.4f}")

        improved = val_miou > best_miou
        best_miou = max(best_miou, val_miou)

        # save
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epoch or improved:
            savepath = checkpoints_dir / f"{args.model}_epoch{epoch+1:03d}.pth"
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "args": vars(args),
            }
            torch.save(state, savepath)
            log_string(f"Saved checkpoint: {savepath}")
            if args.save_best and improved:
                best_path = checkpoints_dir / "best.ckpt"
                shutil.copy(str(savepath), str(best_path))
                log_string(f"[Best] Updated {best_path} (mIoU={best_miou:.4f})")


if __name__ == "__main__":
    main()
