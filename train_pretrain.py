"""Pretrain the unconditional diffusion backbone via eps-prediction (DSM)."""
import os
import torch
import matplotlib.pyplot as plt

from models import TransformerScore, EMA
from sde import VPSDE


def train_pretrain(cfg, data):
    device = cfg.device
    X = data["X_train"]
    sde = VPSDE(cfg)

    model = TransformerScore(cfg).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.pre_lr)

    n = X.shape[0]
    losses = []
    print(f"[pretrain] {cfg.pre_epochs} epochs, {n} samples, batch {cfg.pre_batch_size}")
    for epoch in range(cfg.pre_epochs):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, cfg.pre_batch_size):
            x0 = X[perm[i:i + cfg.pre_batch_size]]
            t = torch.rand(x0.shape[0], device=device) * (1 - cfg.eps0) + cfg.eps0
            x_t, z = sde.forward_sample(x0, t)
            eps = model(x_t, t)
            loss = ((eps - z) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)
            ep_loss += loss.item()
            nb += 1
        losses.append(ep_loss / nb)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"[pretrain] epoch {epoch+1:4d}  loss {losses[-1]:.5f}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ema_model = TransformerScore(cfg).to(device)
    ema.copy_to(ema_model)
    arch = {k: getattr(cfg, k) for k in ("pre_d_model", "pre_n_heads", "pre_n_layers",
                                         "pre_dim_ff", "pre_dropout", "seq_len", "n_assets")}
    torch.save({"model": model.state_dict(), "ema": ema_model.state_dict(), "arch": arch},
               os.path.join(cfg.ckpt_dir, "pretrain.pt"))

    os.makedirs(cfg.fig_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.title("Pretrain (eps) loss")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.fig_dir, "loss_pretrain.png"), dpi=120)
    plt.close()
    print("[pretrain] saved ckpt/pretrain.pt and figures/loss_pretrain.png")
    return ema_model
