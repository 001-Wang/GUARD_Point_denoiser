# sngp_head.py
import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# ---------- utils ----------

def _chol_solve_quad(P, Z, base_jitter=1e-6, max_tries=6):
    """Return diag(Z P^{-1} Z^T).  P:(D,D), Z:(...,N,D) where leading dims optional."""
    P = 0.5 * (P + P.T)
    I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
    jitter = base_jitter
    L = None
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(P + jitter * I)
            break
        except RuntimeError:
            jitter *= 10.0
    if L is None:  # fallback
        P_pinv = torch.linalg.pinv(P)
        ZP = Z @ P_pinv
        return (ZP * Z).sum(-1), None

    # solve P X = Z^T
    Zt = Z.transpose(-1, -2)          # (..., D, N)
    X  = torch.cholesky_solve(Zt, L)  # (..., D, N)
    XP = X.transpose(-1, -2)          # (..., N, D)
    cov = (Z * XP).sum(-1)            # (..., N)
    return cov, jitter


# ---------- modules ----------

class RandomFourierFeatures(nn.Module):
    """RFF for shift-invariant kernels (RBF by default with length_scale absorbed into W init)."""
    def __init__(self, in_dim, out_dim, length_scale=1.0):
        super().__init__()
        self.register_buffer('W', torch.randn(in_dim, out_dim) / max(1e-6, length_scale))
        self.register_buffer('b', 2 * math.pi * torch.rand(out_dim))

    def forward(self, x):
        # x: (..., in_dim) -> (..., 2*out_dim)
        proj = x @ self.W + self.b
        return torch.cat([proj.cos(), proj.sin()], dim=-1) * math.sqrt(1.0 / self.W.shape[1])


class SNGPHead(nn.Module):
    """
    标准 SNGP 头：z = RFF(feats) -> logits = z @ beta；precision 累积 (ridge + Z^T Z)。
    - feats: (B, N, F)
    - 返回: logits (B,N,C), sigma2 (B,N) or None
    """
    def __init__(self, feat_dim, num_part, num_rff=1024, ridge=1.0, spectral_beta=True):
        super().__init__()
        self.rff  = RandomFourierFeatures(feat_dim, num_rff)
        self.beta = nn.Linear(2 * num_rff, num_part, bias=False)
        if spectral_beta:
            spectral_norm(self.beta, n_power_iterations=1)
        self.register_buffer("precision", torch.eye(2 * num_rff) * ridge)

    @torch.no_grad()
    def reset_precision(self, ridge=None):
        D = self.precision.size(0)
        if ridge is None:
            ridge = torch.diag(self.precision).mean().item()
        self.precision.copy_(torch.eye(D, device=self.precision.device) * ridge)

    def forward(self, feats, update_precision=True, compute_var=False, force_update=False):
        z = self.rff(feats)                    # (B,N,D)
        logits = self.beta(z)                  # (B,N,C)

        need_update = update_precision and (self.training or force_update)
        if need_update:
            with torch.no_grad():
                z_flat = z.reshape(-1, z.size(-1)).detach()
                self.precision.addmm_(z_flat.T, z_flat, beta=1.0, alpha=1.0)

        sigma2 = None
        if compute_var:
            P = self.precision.to(z.device)
            sigma2, _ = _chol_solve_quad(P, z, base_jitter=1e-6, max_tries=6)  # (B,N)
        return logits, sigma2


class ConcatFeatureSNGP(nn.Module):
    """
    拼特征单头 SNGP：先各自 RFF，再 concat -> 一个 beta + 一个 precision。
    - feats_g: (B,N,Fg)  几何（如 XYZ）
    - feats_s: (B,N,Fs)  语义（如 backbone 点特征）
    """
    def __init__(self,
                 geom_dim, sem_dim,
                 num_part,
                 geom_rff=256, sem_rff=1024,
                 ridge=1.5, spectral_beta=True):
        super().__init__()
        self.rff_g = RandomFourierFeatures(geom_dim, geom_rff)
        self.rff_s = RandomFourierFeatures(sem_dim,  sem_rff)
        D = 2 * (geom_rff + sem_rff)
        self.beta = nn.Linear(D, num_part, bias=False)
        if spectral_beta:
            spectral_norm(self.beta, n_power_iterations=1)
        self.register_buffer("precision", torch.eye(D) * ridge)

    @torch.no_grad()
    def reset_precision(self, ridge=None):
        D = self.precision.size(0)
        if ridge is None:
            ridge = torch.diag(self.precision).mean().item()
        self.precision.copy_(torch.eye(D, device=self.precision.device) * ridge)

    def forward(self, feats_g, feats_s, update_precision=True, compute_var=False, force_update=False):
        zg = self.rff_g(feats_g)   # (B,N,Dg)
        zs = self.rff_s(feats_s)   # (B,N,Ds)
        z  = torch.cat([zg, zs], dim=-1)  # (B,N,D)
        logits = self.beta(z)

        need_update = update_precision and (self.training or force_update)
        if need_update:
            z_flat = z.reshape(-1, z.size(-1)).detach()
            self.precision.addmm_(z_flat.T, z_flat, beta=1.0, alpha=1.0)

        sigma2 = None
        if compute_var:
            P = self.precision.to(z.device)
            sigma2, _ = _chol_solve_quad(P, z, base_jitter=1e-6, max_tries=6)
        return logits, sigma2
