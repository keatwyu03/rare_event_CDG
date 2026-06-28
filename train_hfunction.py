"""Train the time-dependent binary classifier h (note eq 9), on the FULL train set.

Uses the SAME VP forward noising as the backbone. Handles the ~10/90 imbalance
with BCEWithLogits pos_weight = #neg/#pos.
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models import TransformerClassifier
from sde import VPSDE


def _auc(scores, labels):
    """Rank-based AUC (no sklearn). scores, labels: 1-D tensors on any device."""
    s = scores.detach().cpu()
    y = labels.detach().cpu()
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float)
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return ((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def train_hfunction(cfg, data):
    device = cfg.device
    X, B = data["X_train"], data["B_train"]
    sde = VPSDE(cfg)

    n_pos = B.sum().item()
    n_neg = len(B) - n_pos
    auto_w = n_neg / max(n_pos, 1.0)
    w = auto_w if cfg.h_pos_weight <= 0 else cfg.h_pos_weight
    pos_weight = torch.tensor(w, device=device)
    src = "auto #neg/#pos" if cfg.h_pos_weight <= 0 else "config override"
    print(f"[hfunc] pos_weight = {pos_weight.item():.2f} ({src})  (neg={int(n_neg)}, pos={int(n_pos)})")

    model = TransformerClassifier(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.h_lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = X.shape[0]
    losses = []
    for epoch in range(cfg.h_epochs):
        perm = torch.randperm(n, device=device)
        ep_loss, nb = 0.0, 0
        for i in range(0, n, cfg.h_batch_size):
            idx = perm[i:i + cfg.h_batch_size]
            x0, b = X[idx], B[idx]
            # only train on times near clean data: t ~ U(eps0, h_t_max)
            t = torch.rand(x0.shape[0], device=device) * (cfg.h_t_max - cfg.eps0) + cfg.eps0
            x_t, _ = sde.forward_sample(x0, t)
            logit = model(x_t, t)
            loss = bce(logit, b)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / nb)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            # quick AUC at a mid noise level on the train set
            with torch.no_grad():
                t = torch.full((n,), 0.1, device=device)
                xt, _ = sde.forward_sample(X, t)
                auc = _auc(model(xt, t), B)
            print(f"[hfunc] epoch {epoch+1:4d}  bce {losses[-1]:.4f}  train AUC(t=0.1) {auc:.3f}")

    # final AUC sanity check
    with torch.no_grad():
        t = torch.full((n,), 0.1, device=device)
        xt, _ = sde.forward_sample(X, t)
        auc = _auc(model(xt, t), B)
    if auc < 0.55:
        print(f"[hfunc] WARNING: AUC={auc:.3f} ~ 0.5 -> X nearly independent of the event; "
              f"Doob guidance will be weak.")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = cfg.hfunction_ckpt()
    torch.save({"model": model.state_dict(), "h_t_max": cfg.h_t_max}, ckpt_path)

    fig_path = os.path.join(cfg.fig_dir, cfg.htag(), "loss_hfunction.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.title(f"h-function loss ({cfg.htag()})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"[hfunc] saved {ckpt_path} and {fig_path}")
    return model
