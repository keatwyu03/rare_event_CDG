"""H-function calibration check — bypasses the sampling/guidance pipeline.

Forward-noises real windows at several fixed diffusion times tau and reports
sigmoid(h(y_tau, tau)) split by the TRUE surge label. If h learned real signal,
event and no-event groups separate cleanly at low tau (near clean data) and
converge toward the base rate as tau grows; past h_t_max the model was never
trained, so anything there is extrapolation. If the groups don't separate even
at low tau, h's fit is broken independent of guidance strength or gamma."""
import os

import numpy as np
import torch

from common import build_cfg, event_masks, load_data, out_dir, plt, savefig

from main import load_hfunction     # noqa: E402  (common puts ROOT on sys.path)
from sde import VPSDE               # noqa: E402

TAU_VALUES = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]


@torch.no_grad()
def h_probs(model, sde, X, tau_val, device, batch=512):
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        t = torch.full((len(xb),), tau_val, device=device)
        xt, _ = sde.forward_sample(xb, t)
        out.append(torch.sigmoid(model(xt, t)).cpu())
    return torch.cat(out).numpy()


def h_by_tau(model, sde, X, mask, device):
    """dict tau -> (pos_mean, neg_mean, pos_std, neg_std)."""
    res = {}
    for tau in TAU_VALUES:
        p = h_probs(model, sde, X, tau, device)
        pos, neg = p[mask], p[~mask]
        res[tau] = (pos.mean() if len(pos) else np.nan,
                    neg.mean() if len(neg) else np.nan,
                    pos.std() if len(pos) > 1 else np.nan,
                    neg.std() if len(neg) > 1 else np.nan)
    return res


def print_table(name, results):
    print(f"\n{name}")
    print(f"{'tau':>6} | {'pos mean':>10} {'pos std':>9} | {'neg mean':>10} {'neg std':>9} | {'separation':>10}")
    print("-" * 66)
    for tau in TAU_VALUES:
        pm, nm, ps, ns = results[tau]
        print(f"{tau:>6.2f} | {pm:>10.4f} {ps:>9.4f} | {nm:>10.4f} {ns:>9.4f} | {pm - nm:>10.4f}")


def main():
    cfg = build_cfg("h-function calibration by noise level")
    data = load_data(cfg)
    tm, te = event_masks(data)

    model = load_hfunction(cfg)
    model.eval()
    sde = VPSDE(cfg)

    train_res = h_by_tau(model, sde, data["X_train"], tm, cfg.device)
    test_res = h_by_tau(model, sde, data["X_test"], te, cfg.device)
    print_table(f"TRAIN (events {tm.sum()}/{len(tm)})", train_res)
    print_table(f"TEST  (events {te.sum()}/{len(te)})", test_res)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, res, title in [(axes[0], train_res, "Train"), (axes[1], test_res, "Test")]:
        for idx, color, label in [(0, "crimson", "true event (B=1)"),
                                  (1, "steelblue", "true no-event (B=0)")]:
            means = [res[t][idx] for t in TAU_VALUES]
            stds = [res[t][idx + 2] for t in TAU_VALUES]
            ax.errorbar(TAU_VALUES, means, yerr=stds, marker="o", color=color,
                        label=label, capsize=3)
        ax.axvline(cfg.h_t_max, color="gray", linestyle="--", linewidth=1,
                   label=f"h_t_max={cfg.h_t_max:g} (training cap)")
        ax.set_xlabel("tau (0 = clean data, 1 = pure noise)")
        ax.set_ylabel("sigmoid(h(y_tau, tau))")
        ax.set_title(f"{title} — h calibration by true label", fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"H-Function Calibration Check [{cfg.htag()}] (no sampling/guidance involved)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir(cfg), "h_function_eval.png"))


if __name__ == "__main__":
    main()
