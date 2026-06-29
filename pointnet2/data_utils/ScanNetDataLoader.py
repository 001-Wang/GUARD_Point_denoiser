# data_utils/ScanNetPTHDataset.py
import os
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset


def _find_first(d, keys):
    for k in keys:
        if k in d:
            return d[k], k
    return None, None


class ScanNetPTHSceneDataset(Dataset):
    """
    ScanNet-style scene dataset stored as .pth files (one file per scene).
    Each .pth can be:
      - dict with keys like: points/xyz/rgb/labels/semantic_labels
      - tuple/list: (points, labels) OR (xyz, rgb, labels)
      - directly: points tensor + labels elsewhere (rare)

    We return one random 'block' (subsample) per __getitem__:
      points: (npoints, C)  (we keep only xyz or xyzrgb depending on use_rgb)
      labels: (npoints,)
    """

    def __init__(
        self,
        root,
        split="train",
        npoints=8192,
        use_rgb=False,
        normalize_xyz=False,
        repeat=50,
        seed=0,
    ):
        """
        root: directory containing split folders or flat folder of .pth
              Supported layouts:
                root/train/*.pth , root/val/*.pth , root/test/*.pth
                OR root/*.pth (then split ignored)
        repeat: how many samples to draw per scene per epoch (controls epoch length)
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.npoints = int(npoints)
        self.use_rgb = bool(use_rgb)
        self.normalize_xyz = bool(normalize_xyz)
        self.repeat = int(repeat)
        self.rng = random.Random(seed)

        split_dir = self.root / split
        if split_dir.exists():
            self.files = sorted(split_dir.glob("*.pth"))
        else:
            self.files = sorted(self.root.glob("*.pth"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pth found under {split_dir} or {self.root}")

        # lazy cache: load each scene once
        self._cache = {}

    def __len__(self):
        # Each scene yields `repeat` random blocks per epoch
        return len(self.files) * self.repeat

    def _load_scene(self, path: Path):
        if path in self._cache:
            return self._cache[path]

        obj = torch.load(str(path), map_location="cpu")

        xyz = None
        rgb = None
        labels = None

        if isinstance(obj, dict):
            # points may be Nx3 or Nx6
            pts, _k = _find_first(obj, ["points", "pts", "point", "pc", "pointcloud"])
            xyz0, _k2 = _find_first(obj, ["xyz", "coords", "coord", "vertices", "v"])
            rgb0, _k3 = _find_first(obj, ["rgb", "color", "colors", "feat", "feats", "features"])
            lab, _k4 = _find_first(obj, [
                "label_train",          
                "labels", "label",
                "semantic_labels", "sem_labels",
                "y", "target", "targets",
                "label_nyu40"           
            ])



            if pts is not None:
                pts = torch.as_tensor(pts)
                if pts.ndim != 2:
                    raise ValueError(f"{path.name}: points tensor should be (N,C), got {pts.shape}")
                if pts.size(1) >= 3:
                    xyz = pts[:, 0:3]
                if pts.size(1) >= 6:
                    rgb = pts[:, 3:6]
            if xyz is None and xyz0 is not None:
                xyz = torch.as_tensor(xyz0)[:, 0:3]
            if rgb is None and rgb0 is not None:
                rgb = torch.as_tensor(rgb0)
                if rgb.ndim == 2 and rgb.size(1) > 3:
                    rgb = rgb[:, 0:3]
            if lab is not None:
                labels = torch.as_tensor(lab).long().view(-1)

        elif isinstance(obj, (list, tuple)):
            # common patterns:
            #   (points, labels)
            #   (xyz, rgb, labels)
            if len(obj) == 2:
                pts, lab = obj
                pts = torch.as_tensor(pts)
                xyz = pts[:, 0:3]
                if pts.size(1) >= 6:
                    rgb = pts[:, 3:6]
                labels = torch.as_tensor(lab).long().view(-1)
            elif len(obj) == 3:
                xyz = torch.as_tensor(obj[0])[:, 0:3]
                rgb = torch.as_tensor(obj[1])
                labels = torch.as_tensor(obj[2]).long().view(-1)
            else:
                raise ValueError(f"{path.name}: unsupported tuple/list len={len(obj)}")

        else:
            raise ValueError(f"{path.name}: unsupported .pth content type: {type(obj)}")

        if xyz is None or labels is None:
            raise ValueError(
                f"{path.name}: failed to parse. Need xyz + labels. "
                f"Got xyz={None if xyz is None else xyz.shape}, labels={None if labels is None else labels.shape}"
            )

        # sanitize
        xyz = xyz.float()
        labels = labels.long()
        if rgb is not None:
            rgb = rgb.float()

        # optional normalize xyz per scene (center + scale)
        if self.normalize_xyz:
            c = xyz.mean(dim=0, keepdim=True)
            xyz = xyz - c
            s = torch.sqrt((xyz ** 2).sum(dim=1)).max().clamp_min(1e-6)
            xyz = xyz / s

        scene = {"xyz": xyz, "rgb": rgb, "labels": labels}
        self._cache[path] = scene
        return scene

    def __getitem__(self, idx):
        scene_idx = idx // self.repeat
        path = self.files[scene_idx]
        scene = self._load_scene(path)

        xyz = scene["xyz"]              # (N,3)
        labels = scene["labels"]        # (N,)
        rgb = scene["rgb"]              # (N,3) or None

        N = xyz.size(0)
        n = self.npoints

        if N >= n:
            # random sample indices
            sel = torch.randint(0, N, (n,), dtype=torch.long)
        else:
            # pad by repeating
            pad = torch.randint(0, N, (n - N,), dtype=torch.long)
            sel = torch.cat([torch.arange(N, dtype=torch.long), pad], dim=0)

        xyz_s = xyz[sel, :]
        y_s = labels[sel]

        if self.use_rgb:
            if rgb is None:
                # if rgb missing, fill zeros
                rgb_s = torch.zeros((n, 3), dtype=torch.float32)
            else:
                rgb_s = rgb[sel, :]
            pts = torch.cat([xyz_s, rgb_s], dim=1)  # (n,6)
        else:
            pts = xyz_s  # (n,3)

        return pts, y_s, str(path.name)
