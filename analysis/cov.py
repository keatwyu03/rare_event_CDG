"""Correlation & covariance matrices of LAST-DAY returns: real all / real event
vs unconditional / Doob-conditional generated, train and test, standardized
space. Also prints the matrices and Frobenius distances of each generated
panel to the real event-window structure.

Outputs (results/analysis/<htag>/):
  corr_lastday_{train,test}_<g>.png
  cov_lastday_{train,test}_<g>.png
"""
import os

import numpy as np
import pandas as pd

from common import (build_cfg, event_masks, gtag, last_day, load_data,
                    load_samples, out_dir, plt, savefig)


def _avg_offdiag(C):
    iu = np.triu_indices(C.shape[0], 1)
    return C[iu].mean()


def _plot_panels(panels, tickers, kind, title, path):
    """panels: list of (label, (N, n) last-day array). kind: 'corr' | 'cov'."""
    n = len(tickers)
    mats = [(lbl, np.corrcoef(a.T) if kind == "corr" else np.cov(a.T), len(a))
            for lbl, a in panels]
    if kind == "corr":
        vmin, vmax = -1.0, 1.0
    else:
        lim = max(np.abs(M).max() for _, M, _ in mats)
        vmin, vmax = -lim, lim

    fmt = "{:.2f}" if kind == "corr" else "{:.3f}"
    font_size = max(8, min(13, 40 // n))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    for ax, (lbl, M, cnt) in zip(axes.ravel(), mats):
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        extra = f", avg={_avg_offdiag(M):.2f}" if kind == "corr" else ""
        ax.set_title(f"{lbl}  (n={cnt}{extra})", fontsize=10, fontweight="bold")
        ax.set_xticks(range(n)); ax.set_xticklabels(tickers, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(tickers, fontsize=7)
        for r in range(n):
            for c in range(n):
                v = M[r, c]
                ax.text(c, r, fmt.format(v), ha="center", va="center",
                        fontsize=font_size, fontweight="bold",
                        color="white" if abs(v) > 0.6 * abs(vmax) else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, path)
    return {lbl: M for lbl, M, _ in mats}


def main():
    cfg = build_cfg("Last-day correlation/covariance comparison")
    data = load_data(cfg)
    seq, n, tickers = data["seq_len"], data["n_assets"], data["tickers"]
    tm, te = event_masks(data)

    X_uncond, X_cond = load_samples(cfg)
    gu, gc = last_day(X_uncond, seq, n), last_day(X_cond, seq, n)
    r_tr, r_te = last_day(data["X_train"], seq, n), last_day(data["X_test"], seq, n)

    d, g = out_dir(cfg), gtag(cfg)
    tag = f"{cfg.htag()}_{g}"
    splits = [("train", [("real train (all)", r_tr), ("real train (event)", r_tr[tm]),
                         ("uncond generated", gu), ("Doob conditional", gc)]),
              ("test", [("real test (all)", r_te), ("real test (event)", r_te[te]),
                        ("uncond generated", gu), ("Doob conditional", gc)])]

    for split, panels in splits:
        for kind in ("corr", "cov"):
            mats = _plot_panels(
                panels, tickers, kind,
                f"Last-day {kind} — {split} [{tag}] (standardized space)",
                os.path.join(d, f"{kind}_lastday_{split}_{g}.png"))

            print(f"\n==== {kind.upper()} matrices ({split}) ====")
            for lbl, M in mats.items():
                print(f"\n{lbl}:")
                print(pd.DataFrame(M, index=tickers, columns=tickers).round(3).to_string())

            if kind == "corr":
                ev = mats[f"real {split} (event)"]
                d_c = np.linalg.norm(mats["Doob conditional"] - ev)
                d_u = np.linalg.norm(mats["uncond generated"] - ev)
                print(f"\n[analysis][{tag}|{split}] ||Doob-event||_F={d_c:.4f}  "
                      f"||uncond-event||_F={d_u:.4f}  -> Doob "
                      f"{'IS' if d_c < d_u else 'is NOT'} closer to the event structure")


if __name__ == "__main__":
    main()
