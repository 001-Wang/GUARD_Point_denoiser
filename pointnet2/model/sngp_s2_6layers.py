
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

# --- Try imports that match your repo layout ---
try:
    from pointnet2_utils import (
        PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
    )
except Exception:
    from models.pointnet2_utils import (
        PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
    )

from sngp.sngp_headv2 import SNGPHead


def chunked_knn_indices(xyz_bnc, k, chunk_size=256):
    """
    xyz_bnc: (B,N,3)  -> idx_out: (B,N,k) with values in [0, N-1]
    """
    B, N, _ = xyz_bnc.shape
    device = xyz_bnc.device
    idx_out = torch.empty(B, N, k, dtype=torch.long, device=device)

    for b in range(B):
        X = xyz_bnc[b]                            # (N,3)
        X_sq = (X ** 2).sum(dim=1)                # (N,)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            C = end - start
            Q = X[start:end]                      # (C,3)
            Q_sq = (Q ** 2).sum(dim=1, keepdim=True)          # (C,1)
            QX = Q @ X.t()                                     # (C,N)
            dist2 = Q_sq + X_sq.unsqueeze(0) - 2.0 * QX        # (C,N)

            # Exclude self for this slice: row r (0..C-1) corresponds to global col (start+r)
            row = torch.arange(C, device=device)
            dist2[row, start + row] = float('inf')

            topk = torch.topk(dist2, k=min(k, N-1), largest=False, dim=1).indices  # (C,k)
            idx_out[b, start:end] = topk

    # Safety clamp (should be no-ops; keeps us bulletproof)
    return idx_out.clamp_(0, N-1).contiguous()



def gather_neighbors(x, idx):
    """
    x:   (B,N,C)
    idx: (B,N,k) in [0, N-1]
    ->   (B,N,k,C)
    """
    B, N, C = x.shape
    idx = idx.to(device=x.device, dtype=torch.long).contiguous()
    idx = idx.clamp_(0, N-1)  # extra safety

    # flatten-based gather avoids building (B,N,N,C)
    base = (torch.arange(B, device=x.device).view(B, 1, 1) * N)  # (B,1,1)
    idx_flat = (idx + base).reshape(-1)                          # (B*N*k,)
    x_flat = x.reshape(B * N, C)                                 # (B*N,C)
    neigh = x_flat.index_select(0, idx_flat).reshape(B, N, idx.size(-1), C)
    return neigh



# ----------------- Multi-Scale + Transformer -----------------
class MultiScaleEncoder(nn.Module):
    def __init__(self, in_dim=3, embed_dim=128, ms_k=(16, 32, 64), per_scale_width=48, dropout=0.1, sn_first=True, chunk_size=256):
        super().__init__()
        self.ms_k = list(ms_k)
        self.per_scale_width = per_scale_width
        self.chunk_size = int(chunk_size)

        mlps = []
        for _ in self.ms_k:
            mlp = nn.Sequential(
                nn.Linear(4, per_scale_width),
                nn.ReLU(inplace=True),
                nn.Linear(per_scale_width, per_scale_width),
                nn.ReLU(inplace=True),
            )
            mlps.append(mlp)
        self.mlps = nn.ModuleList(mlps)

        fuse_in = 2 * per_scale_width * len(self.ms_k)
        proj = nn.Linear(fuse_in, embed_dim)
        self.proj = SN(proj) if sn_first else proj

        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, xyz_bnc):
        # >>> ensure FP32 for KNN math (even if autocast is on)
        xyz32 = xyz_bnc.to(torch.float32)
        B, N, _ = xyz32.shape
        feats_scales = []

        for s, k in enumerate(self.ms_k):
            idx   = chunked_knn_indices(xyz32, k, chunk_size=self.chunk_size)   # (B,N,k) int64
            neigh = gather_neighbors(xyz32, idx)                                # (B,N,k,3)
            center = xyz32.unsqueeze(2).expand_as(neigh)                        # (B,N,k,3)
            rel   = neigh - center                                              # (B,N,k,3)
            dist  = torch.linalg.norm(rel, dim=-1, keepdim=True)                # (B,N,k,1)
            f     = torch.cat([rel, dist], dim=-1)                              # (B,N,k,4)
            f     = self.mlps[s](f)                                             # (B,N,k,W)
            f_mean = f.mean(dim=2)                                              # (B,N,W)
            f_max  = f.max(dim=2).values                                        # (B,N,W)
            fs     = torch.cat([f_mean, f_max], dim=-1)                         # (B,N,2W)
            feats_scales.append(fs)

        f_all = torch.cat(feats_scales, dim=-1)                                 # (B,N,2W*|S|)
        h = self.proj(self.drop(f_all))                                         # (B,N,Ct)
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
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class GeomTransformerMS(nn.Module):
    def __init__(self, embed_dim=128, ms_k=(16,32,64), per_scale_width=48, num_layers=1, num_heads=4, dropout=0.1, sn_first=True, chunk_size=256):
        super().__init__()
        self.ms_enc = MultiScaleEncoder(in_dim=3, embed_dim=embed_dim, ms_k=ms_k, per_scale_width=per_scale_width, dropout=dropout, sn_first=sn_first, chunk_size=chunk_size)
        blocks = []
        for i in range(num_layers):
            blocks.append(TransformerBlock(embed_dim, num_heads=num_heads, dropout=dropout, sn=(sn_first and i==0)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, xyz_bnc):
        h = self.ms_enc(xyz_bnc)   # (B,N,Ct)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)


# ----------------- Multi-bandwidth RFF GP (early geo) -----------------
class MultiRFFHead(nn.Module):
    def __init__(self, feat_dim, sigmas=(0.3,0.7,1.4), m_each=160, ridge=5.0):
        super().__init__()
        self.sigmas = list(sigmas)
        self.m_each = int(m_each)
        self.m_total = self.m_each * len(self.sigmas)
        self.feat_dim = feat_dim
        self.ridge = float(ridge)

        Ws = []; bs = []
        for s in self.sigmas:
            W = torch.randn(feat_dim, self.m_each) / max(s, 1e-6)  # gamma=1/sigma
            b = torch.rand(self.m_each) * 2 * math.pi
            Ws.append(W); bs.append(b)
        self.register_buffer('W', torch.cat(Ws, dim=1))     # (D, m_total)
        self.register_buffer('b', torch.cat(bs, dim=0))     # (m_total,)
        self.register_buffer('P', torch.zeros(self.m_total, self.m_total))  # Φ^TΦ

        self.prefac = (2.0 / self.m_total) ** 0.5

    @property
    def precision(self):
        # expose P via a standard name
        return self.P

    @torch.no_grad()
    def reset_precision(self):
        # IMPORTANT: keep P as ΦᵀΦ only; ridge is added later in variance computation
        self.P.zero_()

    @torch.no_grad()
    def load_precision(self, P: torch.Tensor):
        assert P.shape == self.P.shape
        self.P.copy_(P.to(self.P.device))

    def _phi(self, h_bnc):
        proj = h_bnc @ self.W + self.b          # (B,N,M)
        return self.prefac * torch.cos(proj)

    def forward(self, h_bnc, update_precision=True, compute_var=True, force_update=False):
        phi = self._phi(h_bnc)                   # (B,N,M)
        B, N, M = phi.shape


        if update_precision and (self.training or force_update):
            with torch.no_grad():
                Phi = phi.reshape(-1, M)
                self.P = self.P.to(Phi.device)
                # P += ΦᵀΦ
                self.P.addmm_(Phi.T, Phi, beta=1.0, alpha=1.0)

        sigma2 = None
        if compute_var:
            I = torch.eye(M, device=phi.device, dtype=phi.dtype)
            K = self.P.to(phi.device) + self.ridge * I
            L = torch.linalg.cholesky(K)
            rhs = phi.reshape(-1, M)            # (BN, M)
            sol = torch.cholesky_solve(rhs.T.contiguous(), L)  # (M, BN)
            v = (rhs * sol.T).sum(dim=1)        # (BN,)
            sigma2 = v.reshape(B, N)            # (B,N)
        return None, sigma2


# ----------------- Full Model -----------------
class get_model(nn.Module):
    def __init__(self, num_classes,
                 normal_channel=False,
                 geom_cfg=None,
                 final_sngp_cfg=None):
        super().__init__()

        if geom_cfg is None:
            geom_cfg = dict(
                embed_dim=128, num_layers=1, num_heads=4, dropout=0.1, sn_first=True,
                ms_k=(16,32,64), per_scale_width=48, chunk_size=256,
                rff_sigmas=(0.3,0.7,1.4), rff_m_each=160, rff_ridge=5.0
            )
        self.C_t = int(geom_cfg.get('embed_dim', 128))
        self.geom_tf = GeomTransformerMS(
            embed_dim=self.C_t,
            ms_k=tuple(geom_cfg.get('ms_k', (16,32,64))),
            per_scale_width=int(geom_cfg.get('per_scale_width', 48)),
            num_layers=int(geom_cfg.get('num_layers', 1)),
            num_heads=int(geom_cfg.get('num_heads', 4)),
            dropout=float(geom_cfg.get('dropout', 0.1)),
            sn_first=bool(geom_cfg.get('sn_first', True)),
            chunk_size=int(geom_cfg.get('chunk_size', 256)),
        )

        self.early_rffgp = MultiRFFHead(
            feat_dim=self.C_t,
            sigmas=tuple(geom_cfg.get('rff_sigmas', (0.3,0.7,1.4))),
            m_each=int(geom_cfg.get('rff_m_each', 160)),
            ridge=float(geom_cfg.get('rff_ridge', 5.0)),
        )

        # self.sa1 = PointNetSetAbstractionMsg(
        #     1024, [0.05, 0.1, 0.2], [16, 32, 64],
        #     self.C_t,
        #     [[16, 16, 32], [32, 32, 64], [32, 48, 64]]
        # )
        self.sa1 = PointNetSetAbstractionMsg(512,  [0.1, 0.2, 0.4], [32, 64, 128], self.C_t, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128,  [0.4, 0.8],       [64, 128],     128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(16, [0.8, 1.6], [64,128], in_channel=512+3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 320,  mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128])
        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv1d(128, 128, 1), eps=1e-12)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = torch.nn.utils.spectral_norm(nn.Conv1d(128, num_classes, 1), eps=1e-12)

        if final_sngp_cfg is None:
            final_sngp_cfg = dict(enabled=False, num_rff=1024, ridge=1.0, spectral_beta=True)
        self.final_sngp = None
        if final_sngp_cfg.get('enabled', True):
            self.final_sngp = SNGPHead(
                feat_dim=128, num_part=num_classes,
                num_rff=final_sngp_cfg.get('num_rff', 1024),
                ridge=final_sngp_cfg.get('ridge', 1.0),
                spectral_beta=final_sngp_cfg.get('spectral_beta', True)
            )

    def forward(self, xyz, cls_label=None, return_point_feats=False, compute_sigma=True, force_update_geo=False):
        B, _, N = xyz.shape
        x_bnc = xyz.transpose(1, 2).contiguous()

        h_bnc = self.geom_tf(x_bnc)                # (B,N,Ct)
        _, sigma2_geo = self.early_rffgp(
            h_bnc,
            update_precision=True,                 # keep True
            compute_var=compute_sigma,
            force_update=force_update_geo          # NEW: allow updates in eval()
        )

        l0_xyz = xyz
        l0_points_raw = xyz
        l0_points = h_bnc.permute(0, 2, 1).contiguous()

        with torch.amp.autocast('cuda', enabled=False):
            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)          # 512
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)          # 128
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)          # 1 (group_all)

            # FP：l3 → l2 → l1 → l0
            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(l0_xyz, l1_xyz,
                                torch.cat([l0_xyz, l0_points_raw], dim=1),  # 通道=6
                                l1_points)



            feat = F.relu(self.bn1(self.conv1(l0_points)))
            point_feats = feat.permute(0, 2, 1).contiguous()

            x = self.drop1(feat)
            x = self.conv2(x)
            baseline_logits = F.log_softmax(x, dim=1).permute(0, 2, 1)

        sngp_logits, sigma2_sem = None, None
        if self.final_sngp is not None:
            sngp_logits, sigma2_sem = self.final_sngp(
                point_feats, update_precision=self.training, compute_var=compute_sigma
            )

        aux = {'sigma2_geo': sigma2_geo, 'sigma2_sem': sigma2_sem, 'point_feats': point_feats}
        return baseline_logits, sngp_logits, aux


class get_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, trans_feat=None, weight=None):
        if weight is None:
            class_counts = torch.bincount(target.flatten(), minlength=pred.size(1))
            weight = 1.0 / (class_counts.float() + 1e-6)
            weight = weight / weight.sum()
            weight = weight.to(pred.device)
        total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss
