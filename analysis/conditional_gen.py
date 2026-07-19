"""Conditional generation diagnostics: real EVENT windows vs Doob-conditional
generated, standardized space, LAST-DAY cross-sections.

Outputs (results/analysis/<htag>/):
  conditional_lastday_{train,test}_<g>.png   per-stock last-day KDEs
  conditional_diagnostics_<g>.png            per-stock mean/std/quantile table
"""
import os

import numpy as np

from common import (build_cfg, event_masks, gtag, last_day, load_data,
                    load_samples, marginal_kde_figure, out_dir, save_table)


def main():
    cfg = build_cfg("Conditional generation diagnostics (last-day)")
    data = load_data(cfg)
    seq, n, tickers = data["seq_len"], data["n_assets"], data["tickers"]
    tm, te = event_masks(data)

    real = {"train": last_day(data["X_train"], seq, n)[tm],
            "test": last_day(data["X_test"], seq, n)[te]}
    _, X_cond = load_samples(cfg)
    gen = last_day(X_cond, seq, n)
    d, g = out_dir(cfg), gtag(cfg)
    tag = f"{cfg.htag()}_{g}"

    for split, R in real.items():
        marginal_kde_figure(
            R, gen, tickers,
            real_label=f"real {split} EVENT windows", gen_label="Doob conditional",
            xlabel="standardized return (last day of window)",
            title=f"Conditional generation — last-day marginals ({split} events, standardized) [{tag}]",
            path=os.path.join(d, f"conditional_lastday_{split}_{g}.png"))

    rows = []
    for j, tk in enumerate(tickers):
        for label, V in [("real train event", real["train"][:, j]),
                         ("real test event", real["test"][:, j]),
                         ("Doob cond gen", gen[:, j])]:
            qs = np.quantile(V, [.01, .05, .5, .95, .99]).round(3)
            rows.append([tk if label.startswith("real train") else "", label,
                         f"{V.mean():+.3f}", f"{V.std():.3f}", str(qs)])
    save_table(rows, ["Stock", "Split", "Mean", "Std", "q[1,5,50,95,99]"],
               f"Conditional Generation — Last-Day Diagnostics (standardized) [{tag}]",
               os.path.join(d, f"conditional_diagnostics_{g}.png"))


if __name__ == "__main__":
    main()
