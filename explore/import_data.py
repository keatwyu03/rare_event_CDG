"""Build the data files the pipeline consumes (single injection point,
mirroring the cdg_finance explore/import_data.py):

  explore/macro_data_new.csv
      daily adjusted-close PRICES for the 10-ticker universe (yfinance,
      Date index + one column per ticker) — data.py computes logret itself.
      config.csv_path points here by default.

  latent_state_estimation/inflation_state.csv
      latent inflation-pressure state from LatentStateEstimator:
      `s` = Kalman-filtered daily state level, `delta_s` = its daily
      increment (drives the surge event label; row 1 is set to s[0] so
      s = cumsum(delta_s) exactly). config.state_csv points here by default.

The estimator runs off the committed macro panel CSVs in
latent_state_estimation/; refresh those first (rarely, needs FRED_API_KEY)
with latent_state_estimation/macro_importer.py. The Kalman MLE (Nelder-Mead
over the full daily sample) can take several minutes.

Usage:
  python explore/import_data.py            # both files
  python explore/import_data.py --stocks   # prices only
  python explore/import_data.py --state    # latent state only
  python explore/import_data.py --joint    # growth+inflation joint state
"""
import argparse
import os
import sys

import pandas as pd
import yfinance as yf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LSE = os.path.join(_ROOT, "latent_state_estimation")
for p in (_ROOT, _LSE):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import TICKERS                      # noqa: E402
from macro_main import LatentStateEstimator     # noqa: E402

START = "2000-01-01"


def build_stocks():
    px = yf.download(TICKERS, start=START, auto_adjust=True)["Close"]
    px = px[TICKERS].dropna(how="any")          # fixed column order, aligned rows
    if len(px) < 1000:
        raise RuntimeError(
            f"yfinance returned only {len(px)} rows — likely rate-limited or a "
            "ticker failed; wait a few minutes and rerun.")
    out = os.path.join(_HERE, "macro_data_new.csv")
    px.to_csv(out, index_label="Date")
    print(f"[import] {out}: {len(px)} rows, {px.index[0].date()} -> {px.index[-1].date()}")


def build_state(variables=("inflation",)):
    est = LatentStateEstimator(method="state_space", variables=variables)
    s = est.fit().rename("s")                   # filtered daily state level
    delta = s.diff().rename("delta_s")
    delta.iloc[0] = s.iloc[0]                   # keep row 1 (s = cumsum(delta_s))
    out = os.path.join(_LSE, "inflation_state.csv")
    pd.DataFrame({"delta_s": delta, "s": s}).to_csv(out)
    print(f"[import] {out}: {len(s)} rows, {s.index[0].date()} -> {s.index[-1].date()}  "
          f"(variables={list(variables)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build macro_data_new.csv + inflation_state.csv")
    p.add_argument("--stocks", action="store_true", help="only download stock prices")
    p.add_argument("--state", action="store_true", help="only estimate the latent state")
    p.add_argument("--joint", action="store_true",
                   help="joint growth+inflation state instead of inflation-only")
    args = p.parse_args()
    run_both = not (args.stocks or args.state)
    if args.stocks or run_both:
        build_stocks()
    if args.state or run_both:
        build_state(("growth", "inflation") if args.joint else ("inflation",))
