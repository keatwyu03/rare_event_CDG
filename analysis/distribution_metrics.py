"""Distribution metrics on LAST-DAY marginals (standardized space):
  - per-stock Wasserstein-1 distances: uncond-gen vs real all, cond-gen vs
    real event windows, train and test (printed + table png)
  - log-log survival ("tail") plots of |last-day return|: real event vs
    conditional vs unconditional

Outputs (results/analysis/<htag>/):
  wasserstein_lastday_<g>.png
  tail_loglog_{train,test}_<g>.png
"""
import os

import numpy as np
from scipy.stats import wasserstein_distance

from common import (build_cfg, event_masks, gtag, last_day, load_data,
                    load_samples, out_dir, save_table, savefig, stock_grid)


def _survival(vals):
    s = np.sort(np.abs(vals))
    p = np.arange(len(s), 0, -1) / len(s)
    return s, p


def main():
    cfg = build_cfg("Wasserstein + tail plots (last-day marginals)")
    data = load_data(cfg)
    seq, n, tickers = data["seq_len"], data["n_assets"], data["tickers"]
    tm, te = event_masks(data)

    X_uncond, X_cond = load_samples(cfg)
    gu, gc = last_day(X_uncond, seq, n), last_day(X_cond, seq, n)
    r_tr, r_te = last_day(data["X_train"], seq, n), last_day(data["X_test"], seq, n)

    d, g = out_dir(cfg), gtag(cfg)
    tag = f"{cfg.htag()}_{g}"

    # ---- Wasserstein table: matched comparisons only ----
    print(f"\nWasserstein-1 — last-day marginals [{tag}]")
    print(f"{'Stock':<8} {'unc|tr all':>11} {'unc|te all':>11} {'cond|tr evt':>12} {'cond|te evt':>12}")
    print("-" * 58)
    rows = []
    for j, tk in enumerate(tickers):
        w = [wasserstein_distance(r_tr[:, j], gu[:, j]),
             wasserstein_distance(r_te[:, j], gu[:, j]),
             wasserstein_distance(r_tr[tm][:, j], gc[:, j]),
             wasserstein_distance(r_te[te][:, j], gc[:, j])]
        print(f"{tk:<8} {w[0]:>11.4f} {w[1]:>11.4f} {w[2]:>12.4f} {w[3]:>12.4f}")
        rows.append([tk] + [f"{v:.4f}" for v in w])
    save_table(rows,
               ["Stock", "uncond vs train all", "uncond vs test all",
                "cond vs train event", "cond vs test event"],
               f"Wasserstein-1 — Last-Day Marginals (standardized) [{tag}]",
               os.path.join(d, f"wasserstein_lastday_{g}.png"))

    # ---- tail plots ----
    for split, r_all, r_evt in [("train", r_tr, r_tr[tm]), ("test", r_te, r_te[te])]:
        fig, axes = stock_grid(n)
        for j, ax in enumerate(axes[:n]):
            for vals, color, label in [
                    (r_evt[:, j], "darkorange", f"real {split} event (n={len(r_evt)})"),
                    (gc[:, j], "steelblue", f"Doob cond (n={len(gc)})"),
                    (gu[:, j], "seagreen", f"uncond (n={len(gu)})")]:
                s, p = _survival(vals)
                ax.plot(s, p, color=color, linewidth=1.4, label=label)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(tickers[j], fontsize=10, fontweight="bold")
            ax.set_xlabel("|last-day return|"); ax.set_ylabel("P(|R| > x)")
            ax.legend(fontsize=6)
            ax.grid(True, which="both", alpha=0.3)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"Log-log tails — last-day |returns| ({split}, standardized) [{tag}]",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        savefig(fig, os.path.join(d, f"tail_loglog_{split}_{g}.png"))


if __name__ == "__main__":
    main()
