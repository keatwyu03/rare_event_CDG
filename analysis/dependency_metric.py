"""Temporal dependency check: ACF of SQUARED returns within each 10-day window
(volatility clustering), real vs generated, plus per-lag Welch t-test p-values.

This is the one analysis that uses the full day axis rather than only the last
day — the ACF is a within-window temporal property and cannot be computed from
a single day. Real windows are subsampled to non-overlapping ones (the sliding
stride makes neighbors share days); generated windows are independent draws.

Outputs (results/analysis/<htag>/):
  acf_sq_uncond_{train,test}.png     real all vs uncond gen, mean ACF +/- 95% CI
  acf_sq_cond_{train,test}_<g>.png   real event vs Doob cond gen
  acf_sq_pvalues_<g>.png             per-lag Welch t-test p-values
"""
import os

import numpy as np
from scipy.stats import norm, ttest_ind

from common import (build_cfg, event_masks, gtag, load_data, load_samples,
                    out_dir, plt, savefig, stock_grid, windows)

MAX_GEN_WINDOWS = 1500      # cap per-window ACF computations on the generated side


def _acf(x, nlags):
    """Biased sample ACF (statsmodels default), lags 0..nlags."""
    x = x - x.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return np.full(nlags + 1, np.nan)
    return np.array([1.0] + [(x[:-k] * x[k:]).sum() / denom for k in range(1, nlags + 1)])


def _acf_stack(W, nlags):
    """W: (N, seq, n) -> (N, nlags+1, n) ACF of squared returns per window/stock."""
    N, seq, n = W.shape
    out = np.empty((N, nlags + 1, n))
    sq = W ** 2
    for i in range(N):
        for j in range(n):
            out[i, :, j] = _acf(sq[i, :, j], nlags)
    return out


def _nonoverlap(W, seq, shift):
    """Subsample sliding windows to non-overlapping ones (stride seq/shift)."""
    step = max(1, int(np.ceil(seq / shift)))
    return W[::step]


def _subsample(W, cap, seed):
    if len(W) <= cap:
        return W
    idx = np.random.default_rng(seed).choice(len(W), cap, replace=False)
    return W[idx]


def _band_figure(acf_real, acf_gen, tickers, real_label, gen_label, title, path):
    """Mean ACF +/- 95% CI of the mean, real vs generated, per stock (lags >= 1)."""
    nlags = acf_real.shape[1] - 1
    lags = np.arange(1, nlags + 1)
    fig, axes = stock_grid(len(tickers), panel_h=3.5)
    for j, ax in enumerate(axes[:len(tickers)]):
        for A, color, label in [(acf_real, "darkorange", real_label),
                                (acf_gen, "steelblue", gen_label)]:
            m = np.nanmean(A[:, 1:, j], axis=0)
            se = np.nanstd(A[:, 1:, j], axis=0, ddof=1) / np.sqrt(len(A))
            ci = norm.ppf(0.975) * se
            ax.plot(lags, m, color=color, linewidth=1.5, label=label)
            ax.fill_between(lags, m - ci, m + ci, color=color, alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(tickers[j], fontsize=10, fontweight="bold")
        ax.set_xlabel("lag (days)"); ax.set_ylabel("ACF")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(tickers):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, path)


def main():
    cfg = build_cfg("ACF of squared returns: real vs generated")
    data = load_data(cfg)
    seq, n, tickers = data["seq_len"], data["n_assets"], data["tickers"]
    tm, te = event_masks(data)
    nlags = min(9, seq - 1)

    X_uncond, X_cond = load_samples(cfg)
    W = {"train": windows(data["X_train"], seq, n),
         "test": windows(data["X_test"], seq, n)}
    gu = _subsample(windows(X_uncond, seq, n), MAX_GEN_WINDOWS, cfg.seed)
    gc = _subsample(windows(X_cond, seq, n), MAX_GEN_WINDOWS, cfg.seed)

    print(f"[analysis] computing squared-return ACFs (nlags={nlags}, "
          f"gen capped at {MAX_GEN_WINDOWS} windows)")
    acf_gu, acf_gc = _acf_stack(gu, nlags), _acf_stack(gc, nlags)

    d, g = out_dir(cfg), gtag(cfg)
    tag = f"{cfg.htag()}_{g}"
    pval_panels = []        # (label, real_acfs, gen_acfs) for the p-value figure

    for split in ("train", "test"):
        mask = tm if split == "train" else te
        real_all = _nonoverlap(W[split], seq, cfg.window_shift)
        real_evt = W[split][mask]       # events are rare; keep all (may overlap)
        acf_all, acf_evt = _acf_stack(real_all, nlags), _acf_stack(real_evt, nlags)

        _band_figure(acf_all, acf_gu, tickers,
                     f"real {split} all (n={len(real_all)}, non-overlap)",
                     f"uncond gen (n={len(gu)})",
                     f"ACF of squared returns — real vs UNCOND ({split}) [{cfg.htag()}]",
                     os.path.join(d, f"acf_sq_uncond_{split}.png"))
        _band_figure(acf_evt, acf_gc, tickers,
                     f"real {split} EVENT (n={len(real_evt)})",
                     f"Doob cond (n={len(gc)})",
                     f"ACF of squared returns — real event vs COND ({split}) [{tag}]",
                     os.path.join(d, f"acf_sq_cond_{split}_{g}.png"))

        pval_panels.append((f"uncond vs {split} all", acf_all, acf_gu))
        pval_panels.append((f"cond vs {split} event", acf_evt, acf_gc))

    # ---- per-lag Welch t-test p-values, averaged view per stock ----
    lags = np.arange(1, nlags + 1)
    fig, axes = stock_grid(n, panel_h=3.5)
    colors = ["seagreen", "crimson", "steelblue", "mediumpurple"]
    for j, ax in enumerate(axes[:n]):
        for (label, A_r, A_g), color in zip(pval_panels, colors):
            _, p = ttest_ind(A_r[:, 1:, j], A_g[:, 1:, j], axis=0, equal_var=False)
            ax.plot(lags, p, marker="o", markersize=3, linewidth=1.2,
                    color=color, label=label)
        ax.axhline(0.05, color="black", linewidth=0.8, linestyle="--", label="p = 0.05")
        ax.set_title(tickers[j], fontsize=10, fontweight="bold")
        ax.set_xlabel("lag (days)"); ax.set_ylabel("p-value")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Welch t-test: real vs generated squared-return ACF per lag [{tag}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(d, f"acf_sq_pvalues_{g}.png"))


if __name__ == "__main__":
    main()
