"""Data loading, logret/Δy, train/test split, standardization and event labels.

No sklearn: means/stds are computed by hand (train set only) so they can be
saved and used to invert the standardization for plotting.
"""
import numpy as np
import pandas as pd
import torch

from config import Config, TICKERS


def build_clean_frame(cfg: Config, verbose: bool = True):
    """Read CSV -> logret + Δy clean DataFrame (date-indexed). No window/split/standardize.
    Returns (data, logret_cols, price_cols, y_col)."""
    path = cfg.resolve_csv()
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if verbose:
        print(f"[data] loaded {path}")
        print(f"[data] columns: {list(df.columns)}")
        print(df.head())

    # Identify price columns (tickers) and the single rate level column.
    price_cols = [c for c in df.columns if c in TICKERS]
    if len(price_cols) != cfg.n_assets:
        price_cols = list(df.columns[:cfg.n_assets])         # fall back: first n_assets cols
    rate_candidates = [c for c in df.columns if c not in price_cols]
    y_col = "y10_real" if "y10_real" in df.columns else rate_candidates[-1]
    if verbose:
        print(f"[data] price_cols={price_cols}")
        print(f"[data] rate col = '{y_col}'")

    df = df.dropna(how="any")                                # 1. drop NaN rows
    logret = np.log(df[price_cols]).diff()                   # 2. prices -> logret
    dy = df[y_col].diff()                                    # 3. y -> Δy
    logret_cols = [f"{c}_logret" for c in price_cols]
    logret.columns = logret_cols
    data = pd.concat([logret, dy.rename("dy")], axis=1).dropna()   # 4. align
    return data, logret_cols, price_cols, y_col


def load_data(cfg: Config):
    """Returns a dict with standardized tensors, labels, and inverse-transform stats."""
    data, logret_cols, price_cols, y_col = build_clean_frame(cfg)

    # optional date window (e.g. chosen by select_window.py) to reduce regime drift
    if cfg.start_date or cfg.end_date:
        before = len(data)
        data = data.loc[cfg.start_date:cfg.end_date]
        print(f"[data] window [{cfg.start_date or '...'} : {cfg.end_date or '...'}] "
              f"-> {len(data)}/{before} rows")

    # 5. time-ordered split (NO shuffle)
    n_train = int(cfg.train_frac * len(data))
    train, test = data.iloc[:n_train], data.iloc[n_train:]
    print(f"[data] total={len(data)}  train={len(train)}  test={len(test)}")

    # 6. standardize logret columns using TRAIN statistics only
    mu = train[logret_cols].mean()
    sd = train[logret_cols].std(ddof=0)
    X_train = ((train[logret_cols] - mu) / sd).values.astype(np.float32)
    X_test = ((test[logret_cols] - mu) / sd).values.astype(np.float32)

    # 7. event labels B from TRAIN threshold (test reuses train threshold)
    thr = train["dy"].quantile(cfg.event_quantile)
    B_train = (train["dy"] >= thr).astype(np.float32).values
    B_test = (test["dy"] >= thr).astype(np.float32).values
    pos = int(B_train.sum())
    print(f"[data] Δy threshold (train q{cfg.event_quantile})={thr:.6f}")
    print(f"[data] train positives={pos}/{len(B_train)} ({pos/len(B_train):.1%})")

    dev = cfg.device
    return {
        "X_train": torch.tensor(X_train, device=dev),
        "B_train": torch.tensor(B_train, device=dev),
        "X_test": torch.tensor(X_test, device=dev),
        "B_test": torch.tensor(B_test, device=dev),
        "mu": torch.tensor(mu.values, dtype=torch.float32, device=dev),
        "sd": torch.tensor(sd.values, dtype=torch.float32, device=dev),
        "thr": float(thr),
        "logret_cols": logret_cols,
        "tickers": price_cols,
    }


def inverse_standardize(x_std: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """x_raw = x_std * sd + mu  (per column)."""
    return x_std * sd + mu


def print_data_stats(data):
    """Per-stock mean/std of RAW daily logret for 4 groups:
    train(all), test(all), train(event), test(event)."""
    mu = data["mu"].cpu().numpy()
    sd = data["sd"].cpu().numpy()
    tickers = data["tickers"]

    def raw(x):                                   # standardized tensor -> raw logret np
        return x.cpu().numpy() * sd + mu

    Xtr, Xte = raw(data["X_train"]), raw(data["X_test"])
    Btr = data["B_train"].cpu().numpy() > 0.5
    Bte = data["B_test"].cpu().numpy() > 0.5
    groups = {
        "train_all":   Xtr,
        "test_all":    Xte,
        "train_event": Xtr[Btr],
        "test_event":  Xte[Bte],
    }
    means = pd.DataFrame({g: X.mean(0) for g, X in groups.items()}, index=tickers)
    stds = pd.DataFrame({g: X.std(0) for g, X in groups.items()}, index=tickers)

    print("\n[stats] RAW daily logret — MEAN per stock:")
    print(means.to_string(float_format=lambda v: f"{v:+.5f}"))
    print("\n[stats] RAW daily logret — STD per stock:")
    print(stds.to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"\n[stats] counts: train={len(Xtr)} (event {int(Btr.sum())}, {Btr.mean():.1%}), "
          f"test={len(Xte)} (event {int(Bte.sum())}, {Bte.mean():.1%})")
    print("[stats] note: standardization uses train_all stats, so train_all is ~0 mean / "
          "~1 std in standardized space; values above are back in raw logret units.\n")
