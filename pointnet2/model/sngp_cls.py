import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

# --- Try imports that match your repo layout ---
try:
    from pointnet2_utils import (
        PointNetSetAbstractionMsg, PointNetSetAbstraction
    )
except Exception:
    from models.pointnet2_utils import (
        PointNetSetAbstractionMsg, PointNetSetAbstraction
    )

# ====== Geometric encoder (MS + tiny Transformer) & Early RFF-GP ======
def chunked_knn_indices(xyz_bnc, k, chunk_size=256):
    B, N, _ = xyz_bnc.shape
    device = xyz_bnc.device
    idx_out = torch.empty(B, N, k, dtype=torch.long, device=device)
    for b in range(B):
        X = xyz_bnc[b]
        X_sq = (X ** 2).sum(dim=1)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            Q = X[start:end]
            Q_sq = (Q ** 2).sum(dim=1, keepdim=True)
            QX = Q @ X.t()
            dist2 = Q_sq + X_sq.unsqueeze(0) - 2.0 * QX
            # mask self
            row = torch.arange(start, end, device=device)
            dist2[torch.arange(end - start, device=device), row] = float('inf')
            idx_out[b, start:end] = torch.topk(dist2, k=k, largest=False, dim=1).indices
    return idx_out

def gather_neighbors(x, idx):
    B, N, C = x.shape
    k = idx.shape[-1]
    idx_exp = idx.unsqueeze(-1).expand(B, N, k, C)
    x_exp = x.unsqueeze(1).expand(B, N, N, C)
    return torch.take_along_dim(x_exp, idx_exp, dim=2)

class MultiScaleEncoder(nn.Module):
    def __init__(self, in_dim=3, embed_dim=128, ms_k=(16, 32, 64), per_scale_width=32, dropout=0.1, sn_first=True, chunk_size=256):
        super().__init__()
        self.ms_k = list(ms_k)
        self.per_scale_width = per_scale_width
        self.chunk_size = int(chunk_size)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4, per_scale_width),
                nn.ReLU(inplace=True),
                nn.Linear(per_scale_width, per_scale_width),
                nn.ReLU(inplace=True),
            ) for _ in self.ms_k
        ])
        fuse_in = 2 * per_scale_width * len(self.ms_k)
        proj = nn.Linear(fuse_in, embed_dim)
        self.proj = SN(proj) if sn_first else proj
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, xyz_bnc):
        feats = []
        for s, k in enumerate(self.ms_k):
            idx = chunked_knn_indices(xyz_bnc, k, chunk_size=self.chunk_size)   # (B,N,k)
            neigh = gather_neighbors(xyz_bnc, idx)                              # (B,N,k,3)
            center = xyz_bnc.unsqueeze(2).expand_as(neigh)                      # (B,N,k,3)
            rel = neigh - center                                                # (B,N,k,3)
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)                 # (B,N,k,1)
            f = torch.cat([rel, dist], dim=-1)                                  # (B,N,k,4)
            f = self.mlps[s](f)                                                 # (B,N,k,W)
            f_mean = f.mean(dim=2)
            f_max  = f.max(dim=2).values
            feats.append(torch.cat([f_mean, f_max], dim=-1))                    # (B,N,2W)
        h = self.proj(self.drop(torch.cat(feats, dim=-1)))                      # (B,N,Ct)
        return self.norm(h)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, sn=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ffn  = nn.Sequential(
            (SN(nn.Linear(dim, dim*4)) if sn else nn.Linear(dim, dim*4)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            (SN(nn.Linear(dim*4, dim)) if sn else nn.Linear(dim*4, dim)),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x

class GeomTransformerMS(nn.Module):
    def __init__(self, embed_dim=128, ms_k=(16,32,64), per_scale_width=32, num_layers=1, num_heads=4, dropout=0.1, sn_first=True, chunk_size=256):
        super().__init__()
        self.ms_enc = MultiScaleEncoder(
            embed_dim=embed_dim, ms_k=ms_k, per_scale_width=per_scale_width,
            dropout=dropout, sn_first=sn_first, chunk_size=chunk_size
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads, dropout=dropout, sn=(sn_first and i==0))
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, xyz_bnc):
        h = self.ms_enc(xyz_bnc)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)  # (B,N,Ct)

class MultiRFFHead(nn.Module):
    """Maintains Φ^TΦ precision matrix for early geo features; outputs σ²_geo per point."""
    def __init__(self, feat_dim, sigmas=(0.5, 1.0, 2.0), m_each=128, ridge=5.0):
        super().__init__()
        self.sigmas = list(sigmas)
        self.m_each = int(m_each)
        self.m_total = self.m_each * len(self.sigmas)
        self.feat_dim = feat_dim
        self.ridge = float(ridge)
        Ws, bs = [], []
        for s in self.sigmas:
            W = torch.randn(feat_dim, self.m_each) / max(s, 1e-6)
            b = torch.rand(self.m_each) * 2 * math.pi
            Ws.append(W); bs.append(b)
        self.register_buffer('W', torch.cat(Ws, dim=1))                   # (D, M)
        self.register_buffer('b', torch.cat(bs, dim=0))                   # (M,)
        self.register_buffer('P', torch.zeros(self.m_total, self.m_total))# (M, M)
        self.prefac = (2.0 / self.m_total) ** 0.5

    def _phi(self, h_bnc):
        proj = h_bnc @ self.W + self.b
        return self.prefac * torch.cos(proj)  # (B,N,M)

    def forward(self, h_bnc, update_precision=True, compute_var=True):
        phi = self._phi(h_bnc)
        B, N, M = phi.shape
        if update_precision:
            with torch.no_grad():
                Phi = phi.reshape(-1, M)         # (BN, M)
                self.P = self.P.to(Phi.device)
                self.P += Phi.T @ Phi            # accumulate precision
        sigma2 = None
        if compute_var:
            I = torch.eye(M, device=phi.device, dtype=phi.dtype)
            K = self.P.to(phi.device) + self.ridge * I
            L = torch.linalg.cholesky(K)
            rhs = phi.reshape(-1, M)
            sol = torch.cholesky_solve(rhs.T.contiguous(), L)  # (M, BN)
            v = (rhs * sol.T).sum(dim=1)                       # (BN,)
            sigma2 = v.reshape(B, N)                           # (B,N)
        return None, sigma2

# ====== Final SNGP classifier head (wraps per-sample features) ======
from sngp.sngp_headv2 import SNGPHead  # expects (B,N,D) -> (B,N,C) and σ²  (reuse with N=1)

class get_model(nn.Module):
    """
    PointNet++ MSG classifier (ModelNet40) + SNGP:
      - Early geometric RFF-GP for σ²_geo and precision matrix P (per-point)
      - Global SA -> FC -> logits
      - Optional final SNGP head on penultimate global features (treated as N=1 sequence)
    """
    def __init__(self, num_class, normal_channel=True,
                 geom_cfg=None, final_sngp_cfg=None):
        super().__init__()
        self.normal_channel = normal_channel

        # Geom transformer + early RFF-GP (used only for uncertainty/precision updates)
        if geom_cfg is None:
            geom_cfg = dict(
                embed_dim=128, num_layers=1, num_heads=4, dropout=0.1, sn_first=True,
                ms_k=(16,32,64), per_scale_width=32, chunk_size=256,
                rff_sigmas=(0.5,1.0,2.0), rff_m_each=128, rff_ridge=5.0
            )
        self.Ct = int(geom_cfg.get('embed_dim', 128))
        self.geom_tf = GeomTransformerMS(
            embed_dim=self.Ct,
            ms_k=tuple(geom_cfg.get('ms_k', (16,32,64))),
            per_scale_width=int(geom_cfg.get('per_scale_width', 32)),
            num_layers=int(geom_cfg.get('num_layers', 1)),
            num_heads=int(geom_cfg.get('num_heads', 4)),
            dropout=float(geom_cfg.get('dropout', 0.1)),
            sn_first=bool(geom_cfg.get('sn_first', True)),
            chunk_size=int(geom_cfg.get('chunk_size', 256)),
        )
        self.early_rffgp = MultiRFFHead(
            feat_dim=self.Ct,
            sigmas=tuple(geom_cfg.get('rff_sigmas', (0.5,1.0,2.0))),
            m_each=int(geom_cfg.get('rff_m_each', 128)),
            ridge=float(geom_cfg.get('rff_ridge', 5.0)),
        )

        # PointNet++ MSG backbone for classification (mirrors your cls baseline)
        in_channel = 3 if self.normal_channel else 0
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(
            None, None, None, 640 + 3, [256, 512, 1024], True
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

        # Final SNGP head on penultimate global feature (treat as seq len 1)
        if final_sngp_cfg is None:
            final_sngp_cfg = dict(enabled=True, num_rff=1024, ridge=1.0, spectral_beta=True)
        self.final_sngp = None
        if final_sngp_cfg.get('enabled', True):
            self.final_sngp = SNGPHead(
                feat_dim=256, num_part=num_class,
                num_rff=final_sngp_cfg.get('num_rff', 1024),
                ridge=final_sngp_cfg.get('ridge', 1.0),
                spectral_beta=final_sngp_cfg.get('spectral_beta', True)
            )

    def forward(self, xyz):
        """
        xyz: (B, 6/3, N) if normal_channel else (B, 3, N)
        Returns:
          logits_log: (B, num_class)  [log_softmax]
          l3_points: (B, 1024, 1)    [global feature before FCs, for compatibility]
        Precision matrices are kept internally at:
          - self.early_rffgp.P
          - self.final_sngp.P  (if enabled)
        """
        B, C, N = xyz.shape

        # --- Early geometry features for precision (doesn't alter backbone path)
        xyz_only = xyz[:, :3, :]                          # always just coordinates for geo encoder
        x_bnc = xyz_only.transpose(1, 2).contiguous()     # (B,N,3)
        h_bnc = self.geom_tf(x_bnc)                       # (B,N,Ct)
        _ , _ = self.early_rffgp(
            h_bnc, update_precision=self.training, compute_var=False
        )  # keep P updated; skip σ² to save time in training (toggle if needed)

        # --- PointNet++ MSG classifier path (as in your cls baseline)
        if self.normal_channel and C >= 6:
            norm = xyz[:, 3:, :]
            coords = xyz[:, :3, :]
        else:
            norm = None
            coords = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(coords, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)   # l3_points: (B,1024,1)

        # Global MLP head
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        penult = self.drop2(F.relu(self.bn2(self.fc2(x))))   # (B,256)
        logits = self.fc3(penult)                             # (B,C)
        logits_log = F.log_softmax(logits, dim=-1)

        # Optional final SNGP logits/precision update (treat penult as length-1 sequence)
        if self.final_sngp is not None and (self.training or torch.is_grad_enabled()):
            _sngp_logits, _sigma2 = self.final_sngp(
                penult.unsqueeze(1), update_precision=self.training, compute_var=False
            )  # update precision matrix; we keep baseline logits for loss to match cls codepath

        return logits_log, l3_points

class get_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, trans_feat=None):
        # match your cls baseline’s NLL loss on log_softmax logits
        return F.nll_loss(pred, target)
