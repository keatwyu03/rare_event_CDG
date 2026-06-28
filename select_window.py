"""Sliding-window scan to find a data window where train and test have similar
CROSS-SECTIONAL CORRELATION LEVEL (the regime-drift we observed: train avg corr
0.52 vs test 0.37). For each candidate window we do the same 80/20 time split,
compute average pairwise correlation for {all, event} on each side, and score by
how close train and test are.

  score = |avgcorr_train_all - avgcorr_test_all|
        + |avgcorr_train_event - avgcorr_test_event|       (lower = better)

Prints a ranked table, saves results/window/window_scan.csv and a scatter plot,
and recommends START_DATE / END_DATE to paste into run.sh / config.py.

Usage:  python select_window.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data import build_clean_frame

# ── scan settings ───────────────────────────────────────────────────────────
MIN_LEN     = 1200        # min window length (trading days)
LEN_STEP    = 250         # window-length grid step
SLIDE_STEP  = 120         # start-index grid step
MIN_EVENTS  = 25          # need this many events in BOTH splits to score a window
RESULTS_DIR = "results/window"


def _avg_pairwise_corr(X):
    """Mean of upper-triangular pairwise correlations. X: (n, 10) raw logret."""
    if len(X) < 3:
        return np.nan
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(C.shape[0], 1)
    return float(C[iu].mean())


def _score_window(sub, logret_cols, cfg):
    """sub: clean DataFrame slice. Returns metrics dict or None if infeasible."""
    n_train = int(cfg.train_frac * len(sub))
    train, test = sub.iloc[:n_train], sub.iloc[n_train:]
    if len(train) < 50 or len(test) < 50:
        return None
    thr = train["dy"].quantile(cfg.event_quantile)
    tr_evt = train[train["dy"] >= thr]
    te_evt = test[test["dy"] >= thr]
    if len(tr_evt) < MIN_EVENTS or len(te_evt) < MIN_EVENTS:
        return None

    a_tr = _avg_pairwise_corr(train[logret_cols].values)
    a_te = _avg_pairwise_corr(test[logret_cols].values)
    e_tr = _avg_pairwise_corr(tr_evt[logret_cols].values)
    e_te = _avg_pairwise_corr(te_evt[logret_cols].values)
    diff_all = abs(a_tr - a_te)
    diff_evt = abs(e_tr - e_te)
    return {
        "start": sub.index[0].date(), "end": sub.index[-1].date(), "n": len(sub),
        "n_evt_train": len(tr_evt), "n_evt_test": len(te_evt),
        "corr_train_all": a_tr, "corr_test_all": a_te, "diff_all": diff_all,
        "corr_train_evt": e_tr, "corr_test_evt": e_te, "diff_evt": diff_evt,
        "score": diff_all + diff_evt,
    }


def main():
    cfg = Config()
    data, logret_cols, _, _ = build_clean_frame(cfg, verbose=False)
    N = len(data)
    print(f"[scan] full data: {N} rows, {data.index[0].date()} -> {data.index[-1].date()}")

    rows = []
    for L in range(MIN_LEN, N + 1, LEN_STEP):
        for s in range(0, N - L + 1, SLIDE_STEP):
            m = _score_window(data.iloc[s:s + L], logret_cols, cfg)
            if m is not None:
                rows.append(m)
    # always include the full-data baseline for reference
    base = _score_window(data, logret_cols, cfg)
    if base is not None:
        base["start_label"] = "FULL"
        rows.append(base)

    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "window_scan.csv"), index=False)

    show = ["start", "end", "n", "n_evt_train", "n_evt_test",
            "corr_train_all", "corr_test_all", "diff_all",
            "corr_train_evt", "corr_test_evt", "diff_evt", "score"]
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n[scan] TOP 15 windows (smallest train/test correlation gap):")
    print(df[show].head(15).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    if "start_label" in df.columns:
        base_row = df[df["start_label"] == "FULL"]
        if len(base_row):
            bscore = float(base_row["score"].iloc[0])
            rank = int((df["score"] < bscore).sum()) + 1
            print(f"\n[scan] full-data baseline score = {bscore:.3f}  (rank {rank} / {len(df)})")

    best = df.iloc[0]
    print("\n[scan] RECOMMENDED window:")
    print(f"    START_DATE=\"{best['start']}\"")
    print(f"    END_DATE=\"{best['end']}\"")
    print(f"    -> train/test all-corr {best['corr_train_all']:.3f}/{best['corr_test_all']:.3f}, "
          f"event-corr {best['corr_train_evt']:.3f}/{best['corr_test_evt']:.3f}, "
          f"score {best['score']:.3f}")
    print("    set these in run.sh (START_DATE/END_DATE) or config.py, then retrain.")

    # scatter: score vs window length, colored by start year
    fig, ax = plt.subplots(figsize=(9, 5))
    scan = df[df.get("start_label").ne("FULL")] if "start_label" in df else df
    sc = ax.scatter(scan["n"], scan["score"],
                    c=[pd.Timestamp(d).year for d in scan["start"]], cmap="viridis", s=18)
    ax.scatter(best["n"], best["score"], color="red", s=80, marker="*", label="recommended")
    ax.set_xlabel("window length (days)"); ax.set_ylabel("train/test corr gap (lower=better)")
    ax.set_title("Window scan: correlation-level stability"); ax.legend()
    fig.colorbar(sc, ax=ax, label="start year")
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, "window_scan.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"\n[scan] saved {RESULTS_DIR}/window_scan.csv and window_scan.png")


if __name__ == "__main__":
    main()
