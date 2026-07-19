"""Unconditional generation diagnostics: real (all windows) vs unconditionally
generated, standardized space, LAST-DAY cross-sections.

Outputs (results/analysis/<htag>/):
  unconditional_lastday_{train,test}.png   per-stock last-day KDEs
  unconditional_diagnostics.png            per-stock mean/std/quantile table
"""
import os

import numpy as np

from common import (build_cfg, last_day, load_data, load_or_generate_uncond,
                    marginal_kde_figure, out_dir, save_table)


def main():
    cfg = build_cfg("Unconditional generation diagnostics (last-day)")
    data = load_data(cfg)
    seq, n, tickers = data["seq_len"], data["n_assets"], data["tickers"]

    real = {"train": last_day(data["X_train"], seq, n),
            "test": last_day(data["X_test"], seq, n)}
    gen = last_day(load_or_generate_uncond(cfg), seq, n)
    d = out_dir(cfg)

    for split, R in real.items():
        marginal_kde_figure(
            R, gen, tickers,
            real_label=f"real {split} (all windows)", gen_label="uncond generated",
            xlabel="standardized return (last day of window)",
            title=f"Unconditional generation — last-day marginals ({split}, standardized) [{cfg.htag()}]",
            path=os.path.join(d, f"unconditional_lastday_{split}.png"))

    rows = []
    for j, tk in enumerate(tickers):
        for label, V in [("real train", real["train"][:, j]),
                         ("real test", real["test"][:, j]),
                         ("uncond gen", gen[:, j])]:
            qs = np.quantile(V, [.01, .05, .5, .95, .99]).round(3)
            rows.append([tk if label == "real train" else "", label,
                         f"{V.mean():+.3f}", f"{V.std():.3f}", str(qs)])
    save_table(rows, ["Stock", "Split", "Mean", "Std", "q[1,5,50,95,99]"],
               f"Unconditional Generation — Last-Day Diagnostics (standardized) [{cfg.htag()}]",
               os.path.join(d, "unconditional_diagnostics.png"))


if __name__ == "__main__":
    main()
