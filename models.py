"""Transformer backbone (score / eps-prediction) and Transformer classifier (h).

Both tokenize the 10 stocks into 10 tokens and run cross-sectional self-attention.
"""
import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    """Sinusoidal embedding of diffusion time t in [0,1]. t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None, :] * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _Tokenizer(nn.Module):
    """x (B,n) -> tokens (B,n,d_model) with per-stock identity embedding + time cond."""
    def __init__(self, n_assets, d_model):
        super().__init__()
        self.n = n_assets
        self.d = d_model
        self.value_proj = nn.Linear(1, self.d)
        self.id_emb = nn.Embedding(self.n, self.d)
        self.t_mlp = nn.Sequential(
            nn.Linear(self.d, self.d), nn.SiLU(), nn.Linear(self.d, self.d))

    def forward(self, x, t):
        B = x.shape[0]
        tok = self.value_proj(x.unsqueeze(-1))                       # (B,n,d)
        ids = torch.arange(self.n, device=x.device)
        tok = tok + self.id_emb(ids).unsqueeze(0)                    # + identity
        t_emb = self.t_mlp(timestep_embedding(t, self.d))            # (B,d)
        tok = tok + t_emb.unsqueeze(1)                               # broadcast time cond
        return tok


def _make_encoder(d_model, n_heads, n_layers, dim_ff, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
        dropout=dropout, batch_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class TransformerScore(nn.Module):
    """eps-prediction backbone (score matching). Uses the pre_* arch params."""
    def __init__(self, cfg):
        super().__init__()
        self.tok = _Tokenizer(cfg.n_assets, cfg.pre_d_model)
        self.enc = _make_encoder(cfg.pre_d_model, cfg.pre_n_heads,
                                 cfg.pre_n_layers, cfg.pre_dim_ff, cfg.pre_dropout)
        self.head = nn.Linear(cfg.pre_d_model, 1)

    def forward(self, x, t):
        h = self.enc(self.tok(x, t))            # (B,n,d)
        return self.head(h).squeeze(-1)         # (B,n)


class TransformerClassifier(nn.Module):
    """h-function: single logit (B,). Uses the h_* arch params. Sigmoid at inference."""
    def __init__(self, cfg):
        super().__init__()
        self.tok = _Tokenizer(cfg.n_assets, cfg.h_d_model)
        self.enc = _make_encoder(cfg.h_d_model, cfg.h_n_heads,
                                 cfg.h_n_layers, cfg.h_dim_ff, cfg.h_dropout)
        self.head = nn.Sequential(
            nn.Linear(cfg.h_d_model, cfg.h_d_model), nn.SiLU(), nn.Linear(cfg.h_d_model, 1))

    def forward(self, x, t):
        h = self.enc(self.tok(x, t)).mean(dim=1)   # mean-pool over 10 tokens
        return self.head(h).squeeze(-1)            # (B,) logit


class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
