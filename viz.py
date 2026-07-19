"""Plots: generated-vs-actual marginals, correlation grid, and sample stats.

All comparisons are in RAW logret units. Actual windows are de-standardized with
their own per-window entry EMA; generated windows are de-standardized (by the
caller) with entry (mu,sig) sampled from the set they are compared to (train vs
test). Here we only reshape (N, seq*n) windows to pooled daily cross-sections
(N*seq, n) and compute per-stock marginals / the 10x10 cross-sectional correlation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _cross(X_raw, n):
    """(N, seq*n) raw windows -> (N*seq, n) pooled daily cross-sections (with overlap)."""
    A = X_raw.detach().cpu().numpy() if hasattr(X_raw, "detach") else np.asarray(X_raw)
    return A.reshape(A.shape[0], -1, n).reshape(-1, n)


def _cross_dedup(X_raw, n, gidx, seq, shift):
    """Deduped-neighborhood pooling for ACTUAL windows: pool the seq days of every
    selected window but keep each unique CALENDAR day once (removes sliding-window
    oversampling). gidx = global window indices; calendar day of (window g, day d) =
    g*shift + d. De-standardization is exact inversion, so duplicate days are identical
    -> keeping the first occurrence is exact."""
    A = X_raw.detach().cpu().numpy() if hasattr(X_raw, "detach") else np.asarray(X_raw)
    rows = A.reshape(A.shape[0], -1, n).reshape(-1, n)                 # (N*seq, n) window-major
    gidx = np.asarray(gidx)
    day_ids = (gidx[:, None] * shift + np.arange(seq)[None, :]).ravel()  # aligned with rows
    _, keep = np.unique(day_ids, return_index=True)                   # first occ per unique day
    return rows[keep]


def hist_compare(cfg, X_gen_raw, X_actual_raw, tickers, filename, title,
                 gen_label="generated", actual_label="actual", bins=50,
                 actual_gidx=None, seq=None, shift=None):
    """2x5 per-stock marginal histograms: generated vs actual (raw logret, overlaid).
    If actual_gidx is given, the ACTUAL side is deduped by calendar day (no sliding-
    window oversampling); the generated side is always pooled (windows share no days)."""
    n = len(tickers)
    gen = _cross(X_gen_raw, n)
    act = (_cross_dedup(X_actual_raw, n, actual_gidx, seq, shift)
           if actual_gidx is not None else _cross(X_actual_raw, n))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for j, ax in enumerate(axes.flat):
        ax.hist(act[:, j], bins=bins, density=True, alpha=0.5, label=actual_label, color="C0")
        ax.hist(gen[:, j], bins=bins, density=True, alpha=0.5, label=gen_label, color="C1")
        ax.set_title(tickers[j]); ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    p = os.path.join(cfg.fig_dir, filename)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[viz] saved {p}  (gen days={len(gen)}, actual days={len(act)})")


def hist_pretrain_vs_actual(cfg, X_gen_raw, X_actual_raw, tickers,
                            actual_gidx=None, seq=None, shift=None):
    hist_compare(cfg, X_gen_raw, X_actual_raw, tickers,
                 "hist_pretrain_vs_actual.png",
                 "Pretrain (unconditional) generated vs actual raw logret (test-scaled)",
                 gen_label="generated", actual_label="actual", bins=60,
                 actual_gidx=actual_gidx, seq=seq, shift=shift)


def _corr(x_np):
    return np.corrcoef(x_np, rowvar=False)


def _avg_corr(M):
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu].mean()


def corr_grid(cfg, actual, gen_unc_tr, gen_cond_tr, gen_unc_te, gen_cond_te,
              tickers, seq, shift, filename="corr_grid.png", tag="", space="raw logret"):
    """2x3 10x10 correlation heatmaps (raw logret, daily cross-sections):
        row 1: train all   | test all   | pretrain uncond (train-scaled)
        row 2: train event | test event | pretrain+Doob   (train-scaled)
    ACTUAL panels use DEDUPED-neighborhood pooling (each calendar day once, no
    sliding-window oversampling); GENERATED panels pool all window-days (their
    windows don't share days). Frobenius uses MATCHED scaling (gen-vs-train uses
    train-scaled gen, gen-vs-test uses test-scaled gen).
    `actual` maps 'train all'/'test all'/'train event'/'test event' -> (X_raw, gidx).
    """
    n = len(tickers)
    Cg = lambda X: _corr(_cross(X, n))                                 # generated: pooled
    Ca = lambda X, gidx: _corr(_cross_dedup(X, n, gidx, seq, shift))   # actual: deduped
    display = [
        ("train all",   Ca(*actual["train all"])),
        ("test all",    Ca(*actual["test all"])),
        ("gen uncond",  Cg(gen_unc_tr)),
        ("train event", Ca(*actual["train event"])),
        ("test event",  Ca(*actual["test event"])),
        ("gen Doob",    Cg(gen_cond_tr)),
    ]
    mats = {name: M for name, M in display}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (name, _) in zip(axes.flat, display):
        im = ax.imshow(mats[name], vmin=-1, vmax=1, cmap="RdBu_r")
        extra = " (train-scaled)" if name.startswith("gen") and space.startswith("raw") else ""
        ax.set_title(f"{name}{extra}  (avg={_avg_corr(mats[name]):.2f})")
        ax.set_xticks(range(n)); ax.set_xticklabels(tickers, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(tickers, fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle(f"10x10 correlation [{tag}] ({space}): actual (train/test) vs generated")
    p = os.path.join(cfg.fig_dir, filename)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[viz] saved {p}")

    # matched-scaling Frobenius distances of the CONDITIONAL generator to the event structure
    ev_tr, ev_te = mats["train event"], mats["test event"]
    d_cond_tr = np.linalg.norm(Cg(gen_cond_tr) - ev_tr)
    d_unc_tr = np.linalg.norm(Cg(gen_unc_tr) - ev_tr)
    d_cond_te = np.linalg.norm(Cg(gen_cond_te) - ev_te)
    d_unc_te = np.linalg.norm(Cg(gen_unc_te) - ev_te)
    pre = f"[viz][{tag}|{space}]" if tag else f"[viz][{space}]"
    print(f"{pre}[in-sample  / train event] ||Doob-event||_F={d_cond_tr:.4f}  "
          f"||uncond-event||_F={d_unc_tr:.4f}  "
          f"-> Doob {'IS' if d_cond_tr < d_unc_tr else 'is NOT'} closer")
    print(f"{pre}[out-sample / test event ] ||Doob-event||_F={d_cond_te:.4f}  "
          f"||uncond-event||_F={d_unc_te:.4f}  "
          f"-> Doob {'IS' if d_cond_te < d_unc_te else 'is NOT'} closer")


def print_marginal_stats(groups, tickers):
    """Print per-stock RAW logret mean ± std for each named group (raw windows)."""
    n = len(tickers)
    table = {}
    for name, X in groups.items():
        raw = _cross(X, n)
        m, s = raw.mean(0), raw.std(0)
        table[name] = [f"{m[j]:+.4f}±{s[j]:.4f}" for j in range(n)]
    dfp = pd.DataFrame(table, index=tickers)
    print("\n[sample-stats] per-stock RAW logret (mean ± std). real: uncond=all windows, "
          "cond=event windows; gen=generated windows de-standardized with set entries.")
    print(dfp.to_string())