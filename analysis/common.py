"""Shared helpers for the analysis scripts in analysis/ (ported from the
cdg_finance / diffusion_stress_testing repo).

Conventions:
  - All analyses run in STANDARDIZED (z) space — the space the diffusion model
    is trained and sampled in — so real and generated windows are directly
    comparable with no de-standardization / entry-pairing confound.
  - All cross-sectional statistics (correlation, covariance, marginals,
    Wasserstein, tails) use the LAST DAY of each 10-day window, matching the
    cdg_finance convention. Only the ACF script uses the full day axis, since
    volatility clustering is a within-window temporal property.

Wiring to the pipeline outputs:
  run.sh pretrain/hfunction -> ckpt/*.pt + ckpt/*_losses*.csv   (loss curves)
  run.sh sample             -> results/samples/samples_{htag}_g{gamma}.pt
  bash run.sh analysis      -> runs every analysis script against those outputs

Each script is also standalone, e.g.
  python analysis/cov.py --event-quantile 0.9 --h-t-max 1 --gamma 2
The flags must match the run.sh values used to train/sample — checkpoints and
sample files are keyed by them.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)          # ckpt/, results/, figures/ are all ROOT-relative

from config import Config       # noqa: E402
from data import load_data      # noqa: E402


def build_cfg(description=""):
    """Config + the subset of CLI flags that key checkpoints / sample files."""
    cfg = Config()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--event-quantile", type=float, default=cfg.event_quantile)
    p.add_argument("--h-t-max", type=float, default=cfg.h_t_max)
    p.add_argument("--gamma", type=float, default=cfg.gamma)
    p.add_argument("--n-sample", type=int, default=cfg.n_sample)
    p.add_argument("--csv-path", default=cfg.csv_path)
    p.add_argument("--state-csv", default=cfg.state_csv)
    p.add_argument("--start-date", default=cfg.start_date)
    p.add_argument("--end-date", default=cfg.end_date)
    p.add_argument("--device", default=cfg.device)
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    cfg.start_date = cfg.start_date or None
    cfg.end_date = cfg.end_date or None
    return cfg


def gtag(cfg):
    return f"g{cfg.gamma:g}"


def samples_path(cfg):
    return os.path.join("results", "samples", f"samples_{cfg.htag()}_{gtag(cfg)}.pt")


def load_samples(cfg):
    """(X_uncond, X_cond) saved by main.py's sample stage, as cpu float tensors."""
    path = samples_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `bash run.sh sample` with the same "
            f"EVENT_QUANTILE/H_T_MAX/GAMMA first — it saves the generated windows there.")
    d = torch.load(path, map_location="cpu")
    return d["X_uncond"].float(), d["X_cond"].float()


def load_or_generate_uncond(cfg):
    """X_uncond from the samples file if present; else sample fresh from ckpt/pretrain.pt."""
    path = samples_path(cfg)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")["X_uncond"].float()
    from main import load_backbone
    from sample import sample_unconditional
    print(f"[analysis] {path} not found -> sampling {cfg.n_sample} unconditional windows fresh")
    model = load_backbone(cfg)
    return sample_unconditional(cfg, model).cpu().float()


def windows(X, seq, n):
    """(N, seq*n) flat day-major windows -> numpy (N, seq, n)."""
    A = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    return A.reshape(A.shape[0], seq, n).astype(np.float64)


def last_day(X, seq, n):
    """(N, seq*n) windows -> (N, n) last-day cross-sections."""
    return windows(X, seq, n)[:, -1, :]


def event_masks(data):
    """(train_mask, test_mask) boolean numpy arrays from the surge labels."""
    return (data["B_train"].cpu().numpy() > 0.5,
            data["B_test"].cpu().numpy() > 0.5)


def out_dir(cfg, *parts):
    d = os.path.join("results", "analysis", cfg.htag(), *parts)
    os.makedirs(d, exist_ok=True)
    return d


def savefig(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] saved {path}")


def kde_plot(ax, real_vals, gen_vals, real_label, gen_label, xlabel):
    """Overlaid KDE + faint histogram, real (orange) vs generated (blue)."""
    from scipy.stats import gaussian_kde
    x_min = min(real_vals.min(), gen_vals.min()) - 0.5
    x_max = max(real_vals.max(), gen_vals.max()) + 0.5
    x = np.linspace(x_min, x_max, 500)
    for vals, color, label in [(real_vals, "darkorange", real_label),
                               (gen_vals, "steelblue", gen_label)]:
        kde = gaussian_kde(vals, bw_method="silverman")
        ax.plot(x, kde(x), color=color, linewidth=2, label=label)
        ax.hist(vals, bins=40, density=True, alpha=0.2, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def stock_grid(n_stocks, panel_h=4.0):
    """Per-stock grid, rows of 5 (2x5 for the 10-stock universe)."""
    rows = int(np.ceil(n_stocks / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(20, panel_h * rows))
    return fig, np.atleast_1d(axes).ravel()


def marginal_kde_figure(real, gen, tickers, real_label, gen_label, xlabel, title, path):
    """Per-stock last-day KDE grid. real, gen: (N, n_stocks)."""
    fig, axes = stock_grid(len(tickers))
    for j, ax in enumerate(axes[:len(tickers)]):
        kde_plot(ax, real[:, j], gen[:, j],
                 f"{real_label} (n={len(real)})", f"{gen_label} (n={len(gen)})", xlabel)
        ax.set_title(tickers[j], fontsize=10, fontweight="bold")
    for ax in axes[len(tickers):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, path)


def save_table(rows, col_labels, title, path):
    fig, ax = plt.subplots(figsize=(min(30, 3 * len(col_labels) + 4),
                                    0.45 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, path)
