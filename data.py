"""Data: 10x10 windows (10 days x 10 stocks) of stock logret, per-window EMA
standardization, time-ordered split, and the inflation-surge event label.

Each sample is a 10-day window flattened to a 100-vector (day-major: flat index
d*n_assets + s = stock s on day d). Standardization is PER-WINDOW, causal:
    z = (r - EMA_mean_entry) / EMA_vol_entry
where EMA_mean/vol are the trailing EMA(span) of each stock's logret evaluated at
the window's ENTRY day (one 10-vector each), so the transform is invertible
(r = z*sig_entry + mu_entry) and uses no future information.

Event: a window qualifies if the largest single-day up-move of the inflation state
inside it, surge = max_{t in window} delta_s_t, is in the TRAIN top decile.
"""
import numpy as np
import pandas as pd
import torch

from config import Config, TICKERS


def build_clean_frame(cfg: Config, verbose: bool = True):
    """Read stocks + inflation state -> aligned (logret_cols..., delta_s) DataFrame."""
    df = pd.read_csv(cfg.resolve_csv(), index_col=0, parse_dates=True)
    price_cols = [c for c in df.columns if c in TICKERS]
    if len(price_cols) != cfg.n_assets:
        price_cols = list(df.columns[:cfg.n_assets])
    logret = np.log(df[price_cols]).diff()
    logret.columns = [f"{c}_logret" for c in price_cols]

    st = pd.read_csv(cfg.resolve_state(), index_col=0, parse_dates=True)
    if "delta_s" not in st.columns:
        raise KeyError(f"'delta_s' not in {cfg.resolve_state()}; columns={list(st.columns)}")
    ds = st["delta_s"].rename("delta_s")

    data = pd.concat([logret, ds], axis=1).dropna(how="any")
    if verbose:
        print(f"[data] stocks={cfg.resolve_csv()}")
        print(f"[data] state ={cfg.resolve_state()}")
        print(f"[data] aligned rows={len(data)}  {data.index[0].date()} -> {data.index[-1].date()}")
        print(f"[data] price_cols={price_cols}")
    return data, list(logret.columns), price_cols


def load_data(cfg: Config):
    """Returns standardized 10x10 window tensors (N, seq*n), surge-event labels, and stats."""
    data, logret_cols, price_cols = build_clean_frame(cfg)

    if cfg.start_date or cfg.end_date:
        before = len(data)
        data = data.loc[cfg.start_date:cfg.end_date]
        print(f"[data] window [{cfg.start_date or '...'} : {cfg.end_date or '...'}] "
              f"-> {len(data)}/{before} rows")

    seq, n, shift = cfg.seq_len, cfg.n_assets, cfg.window_shift
    R = data[logret_cols].to_numpy(np.float64)          # (T, n) logret
    ds = data["delta_s"].to_numpy(np.float64)           # (T,)  inflation increment

    # causal EMA mean & vol per stock (entry-day standardizer)
    rf = pd.DataFrame(R, columns=logret_cols)
    MU = rf.ewm(span=cfg.ema_span, min_periods=20).mean().shift(1).to_numpy()
    SG = rf.ewm(span=cfg.ema_span, min_periods=20).std().shift(1).to_numpy()
    valid = np.isfinite(MU).all(1) & np.isfinite(SG).all(1) & (SG > 0).all(1)
    first = int(np.argmax(valid))                       # EMA warm-up cut (contiguous thereafter)
    R, MU, SG, ds = R[first:], MU[first:], SG[first:], ds[first:]
    T = len(R)

    starts = np.arange(0, T - seq + 1, shift)
    Z = np.empty((len(starts), seq, n), np.float32)
    surge = np.empty(len(starts), np.float32)
    mu_entry = MU[starts].astype(np.float32)            # (Nw, n) for invertibility
    sig_entry = SG[starts].astype(np.float32)
    for i, s in enumerate(starts):
        Z[i] = ((R[s:s + seq] - MU[s]) / SG[s]).astype(np.float32)   # entry EMA, broadcast over days
        surge[i] = ds[s:s + seq].max()
    X = Z.reshape(len(starts), seq * n)                 # flatten day-major (d*n + s)

    # time-ordered split (NO shuffle), then TRAIN-derived surge threshold
    n_train = int(cfg.train_frac * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    thr = float(np.quantile(surge[:n_train], cfg.event_quantile))
    B_train = (surge[:n_train] >= thr).astype(np.float32)
    B_test = (surge[n_train:] >= thr).astype(np.float32)

    pos = int(B_train.sum())
    print(f"[data] windows={len(X)} (train={len(X_train)}, test={len(X_test)})  dim={seq*n}")
    print(f"[data] surge threshold (train q{cfg.event_quantile})={thr:.6f}")
    print(f"[data] train events={pos}/{len(B_train)} ({pos/len(B_train):.1%})  "
          f"test events={int(B_test.sum())}/{len(B_test)} ({B_test.mean():.1%})")

    dev = cfg.device
    t = lambda a: torch.tensor(a, device=dev)
    return {
        "X_train": t(X_train), "B_train": t(B_train),
        "X_test": t(X_test), "B_test": t(B_test),
        # data is already standardized per-window; identity mu/sd keeps viz inversion a no-op
        "mu": torch.zeros(n, dtype=torch.float32, device=dev),
        "sd": torch.ones(n, dtype=torch.float32, device=dev),
        "mu_entry_train": mu_entry[:n_train], "sig_entry_train": sig_entry[:n_train],
        "mu_entry_test": mu_entry[n_train:], "sig_entry_test": sig_entry[n_train:],
        "thr": thr, "logret_cols": logret_cols, "tickers": price_cols,
        "seq_len": seq, "n_assets": n,
    }


def destandardize(z_flat, mu_entry, sig_entry, seq, n):
    """Invert per-window EMA standardization back to raw logret.
    z_flat: (N, seq*n) standardized windows;  mu_entry, sig_entry: (N, n) entry EMA
    mean/vol (one 10-vector per window, broadcast over the window's days).
    Returns (N, seq*n) raw logret. For GENERATED samples pass entry stats drawn from
    the comparison set (see sample_entries) — the vol-regime knob."""
    z = z_flat.detach().cpu().numpy() if hasattr(z_flat, "detach") else np.asarray(z_flat)
    mu = np.asarray(mu_entry, np.float64)
    sg = np.asarray(sig_entry, np.float64)
    R = z.reshape(-1, seq, n).astype(np.float64) * sg[:, None, :] + mu[:, None, :]
    return R.reshape(z.shape[0], seq * n)


def sample_entries(mu_pool, sig_pool, n_draw, seed=0):
    """Draw n_draw entry (mu, sig) vectors (with replacement) from a set's entry pool
    (Nw, n_assets). Used to give generated windows the train/test regime's vol scale."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(mu_pool), size=n_draw)
    return np.asarray(mu_pool)[idx], np.asarray(sig_pool)[idx]


def _cross(A, n):
    """(N, seq*n) raw/std array -> (N*seq, n) pooled daily cross-sections."""
    A = A.detach().cpu().numpy() if hasattr(A, "detach") else np.asarray(A)
    return A.reshape(A.shape[0], -1, n).reshape(-1, n)


def print_data_stats(data):
    """Per-stock mean/std of RAW daily logret (de-standardized with each window's OWN
    entry EMA) for 4 groups: train(all), test(all), train(event), test(event)."""
    seq, n, tickers = data["seq_len"], data["n_assets"], data["tickers"]
    Xtr = destandardize(data["X_train"], data["mu_entry_train"], data["sig_entry_train"], seq, n)
    Xte = destandardize(data["X_test"], data["mu_entry_test"], data["sig_entry_test"], seq, n)
    Btr = data["B_train"].cpu().numpy() > 0.5
    Bte = data["B_test"].cpu().numpy() > 0.5
    etr = np.repeat(Btr, seq); ete = np.repeat(Bte, seq)   # window mask -> pooled day rows
    ctr, cte = _cross(Xtr, n), _cross(Xte, n)
    groups = {"train_all": ctr, "test_all": cte, "train_event": ctr[etr], "test_event": cte[ete]}
    means = pd.DataFrame({g: X.mean(0) for g, X in groups.items()}, index=tickers)
    stds = pd.DataFrame({g: X.std(0) for g, X in groups.items()}, index=tickers)
    print("\n[stats] RAW daily logret (own-entry de-standardized) — MEAN per stock:")
    print(means.to_string(float_format=lambda v: f"{v:+.5f}"))
    print("\n[stats] RAW daily logret — STD per stock:")
    print(stds.to_string(float_format=lambda v: f"{v:.5f}"))
    ntr, nte = len(data["B_train"]), len(data["B_test"])
    print(f"\n[stats] windows: train={ntr} (event {int(Btr.sum())}, {Btr.mean():.1%}), "
          f"test={nte} (event {int(Bte.sum())}, {Bte.mean():.1%})")
    print("[stats] generated windows are compared after de-standardizing with entry "
          "(mu,sig) sampled from the SET they are compared to (train vs test).\n")