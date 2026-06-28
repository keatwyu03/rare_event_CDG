"""Plots: generated-vs-actual marginals, correlation grid, and sample stats."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_raw_np(x_std, mu, sd):
    """Inverse-standardize and move to numpy. x_std: tensor (N,10)."""
    return (x_std * sd + mu).detach().cpu().numpy()


def hist_compare(cfg, X_gen_std, X_actual_std, mu, sd, tickers, filename, title,
                 gen_label="generated", actual_label="actual", bins=50):
    """2x5 per-stock marginal histograms: generated vs actual (overlaid, density)."""
    gen = _to_raw_np(X_gen_std, mu, sd)
    act = _to_raw_np(X_actual_std, mu, sd)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for j, ax in enumerate(axes.flat):
        ax.hist(act[:, j], bins=bins, density=True, alpha=0.5, label=actual_label, color="C0")
        ax.hist(gen[:, j], bins=bins, density=True, alpha=0.5, label=gen_label, color="C1")
        ax.set_title(tickers[j]); ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    p = os.path.join(cfg.fig_dir, filename)   # filename may contain a subdir, e.g. tmax0.6/...
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[viz] saved {p}  (gen={len(gen)}, actual={len(act)})")


def hist_pretrain_vs_actual(cfg, X_gen_std, X_actual_std, mu, sd, tickers):
    hist_compare(cfg, X_gen_std, X_actual_std, mu, sd, tickers,
                 "hist_pretrain_vs_actual.png",
                 "Pretrain (unconditional) generated vs actual logret",
                 gen_label="generated", actual_label="actual", bins=60)


def _corr(x_np):
    return np.corrcoef(x_np, rowvar=False)


def _avg_corr(M):
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu].mean()


def corr_grid(cfg, X_train_all, X_test_all, X_uncond,
              X_train_evt, X_test_evt, X_cond, mu, sd, tickers,
              filename="corr_grid.png", tag=""):
    """2x3 correlation heatmaps:
        row 1: train all   | test all   | pretrain uncond (generated)
        row 2: train event | test event | pretrain+Doob (generated, conditional)
    """
    panels = [
        ("train all",          X_train_all),
        ("test all",           X_test_all),
        ("pretrain uncond",    X_uncond),
        ("train event",        X_train_evt),
        ("test event",         X_test_evt),
        ("pretrain+Doob",      X_cond),
    ]
    mats = {name: _corr(_to_raw_np(X, mu, sd)) for name, X in panels}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (name, _) in zip(axes.flat, panels):
        im = ax.imshow(mats[name], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"{name}  (avg={_avg_corr(mats[name]):.2f})")
        ax.set_xticks(range(len(tickers))); ax.set_xticklabels(tickers, rotation=90, fontsize=7)
        ax.set_yticks(range(len(tickers))); ax.set_yticklabels(tickers, fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle(f"10x10 correlation [{tag}]: actual (train/test) vs generated (uncond/Doob)")
    p = os.path.join(cfg.fig_dir, filename)   # filename may contain a subdir
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[viz] saved {p}")

    # quantify: generated-conditional distance to actual event structure (in & out of sample)
    pre = f"[viz][{tag}]" if tag else "[viz]"
    d_cond_tr = np.linalg.norm(mats["pretrain+Doob"] - mats["train event"])
    d_unc_tr = np.linalg.norm(mats["pretrain uncond"] - mats["train event"])
    d_cond_te = np.linalg.norm(mats["pretrain+Doob"] - mats["test event"])
    d_unc_te = np.linalg.norm(mats["pretrain uncond"] - mats["test event"])
    print(f"{pre}[in-sample  / train event] ||Doob-event||_F={d_cond_tr:.4f}  "
          f"||uncond-event||_F={d_unc_tr:.4f}  "
          f"-> Doob {'IS' if d_cond_tr < d_unc_tr else 'is NOT'} closer")
    print(f"{pre}[out-sample / test event ] ||Doob-event||_F={d_cond_te:.4f}  "
          f"||uncond-event||_F={d_unc_te:.4f}  "
          f"-> Doob {'IS' if d_cond_te < d_unc_te else 'is NOT'} closer")


def print_marginal_stats(groups, mu, sd, tickers):
    """Print per-stock RAW logret mean ± std for each named group (standardized tensors)."""
    mu_np = mu.cpu().numpy()
    sd_np = sd.cpu().numpy()
    table = {}
    for name, X in groups.items():
        raw = X.detach().cpu().numpy() * sd_np + mu_np
        m, s = raw.mean(0), raw.std(0)
        table[name] = [f"{m[j]:+.4f}±{s[j]:.4f}" for j in range(len(tickers))]
    dfp = pd.DataFrame(table, index=tickers)
    print("\n[sample-stats] per-stock RAW logret (mean ± std). "
          "real: uncond=all days, cond=event days; gen=generated samples.")
    print(dfp.to_string())
