"""Training loss curves: pretrain (eps MSE) + h-function (BCE), from the CSVs
written by train_pretrain.py / train_hfunction.py. Falls back to the most
recent hfunction_losses_*.csv if the exact (quantile, t_max) tag is missing."""
import glob
import os

import pandas as pd

from common import build_cfg, out_dir, plt, savefig


def main():
    cfg = build_cfg("Training loss curves")

    score_path = os.path.join(cfg.ckpt_dir, "pretrain_losses.csv")
    h_path = os.path.join(cfg.ckpt_dir, f"hfunction_losses_{cfg.htag()}.csv")
    if not os.path.exists(h_path):
        hits = glob.glob(os.path.join(cfg.ckpt_dir, "hfunction_losses_*.csv"))
        h_path = max(hits, key=os.path.getmtime) if hits else None
        if h_path:
            print(f"[analysis] exact h-loss CSV for {cfg.htag()} missing -> using {h_path}")

    if not os.path.exists(score_path) and h_path is None:
        raise FileNotFoundError(
            f"no loss CSVs in {cfg.ckpt_dir}/ — run `bash run.sh pretrain` / "
            f"`bash run.sh hfunction` first (they now write *_losses.csv).")

    has_score = os.path.exists(score_path)
    has_h = h_path is not None and os.path.exists(h_path)
    ncols = int(has_score) + int(has_h)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5))
    axes = [axes] if ncols == 1 else list(axes)

    i = 0
    if has_score:
        df = pd.read_csv(score_path)
        axes[i].plot(df["epoch"], df["loss"], linewidth=1.5, color="steelblue")
        axes[i].set_xlabel("epoch"); axes[i].set_ylabel("MSE loss")
        axes[i].set_title("Pretrain (eps) loss", fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        i += 1
    else:
        print(f"[analysis] {score_path} not found — skipping pretrain panel")

    if has_h:
        df = pd.read_csv(h_path)
        axes[i].plot(df["epoch"], df["loss"], linewidth=1.5, color="darkorange")
        axes[i].set_xlabel("epoch"); axes[i].set_ylabel("BCE loss")
        axes[i].set_title(f"h-function loss ({os.path.basename(h_path)})", fontweight="bold")
        axes[i].grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir(cfg), "train_losses.png"))


if __name__ == "__main__":
    main()
