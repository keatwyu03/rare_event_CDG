"""All hyper-parameters and constants for the rate-shock conditional diffusion project.

Everything trainable / tunable lives here as a single dataclass so it can be
overridden from the command line (see main.py) and driven from run.sh.
"""
from dataclasses import dataclass, field
from typing import List
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

# Stock universe (order is fixed; data.py will re-detect from the CSV header anyway).
TICKERS: List[str] = ["IBM", "CSCO", "AAPL", "MSFT", "ORCL",
                      "INTC", "TXN", "QCOM", "AMAT", "ADBE"]


@dataclass
class Config:
    # ---- data ----
    # Both inputs are built by explore/import_data.py (single injection point):
    #   csv_path  — daily adjusted-close prices for the 10 tickers (yfinance)
    #   state_csv — latent inflation-pressure state (s + daily increment delta_s,
    #               Kalman-filtered via latent_state_estimation/); delta_s drives
    #               the surge event. resolve_csv/resolve_state fall back to the
    #               legacy Desktop / project-dir files if these don't exist yet.
    csv_path: str = os.path.join(_HERE, "explore", "macro_data_new.csv")
    state_csv: str = os.path.join(_HERE, "latent_state_estimation", "inflation_state.csv")
    train_frac: float = 0.80          # time-ordered split, no shuffle
    n_assets: int = 10
    # ---- 10x10 window ----
    seq_len: int = 10                 # trading days per window (each sample is 10 days x 10 stocks)
    window_shift: int = 1             # sliding stride between consecutive training windows
                                      # (1 = every day starts a window, as in cdg_finance)
    ema_span: int = 60                # EMA span for per-window (r - EMA_mean)/EMA_vol standardization
    # event = a window whose largest single-day up-move of the inflation state
    # (surge = max_{t in window} delta_s_t) is in the TRAIN top decile.
    event_quantile: float = 0.90      # top 10% of the surge metric = event
    # optional date window (YYYY-MM-DD) to restrict data and reduce train/test
    # regime drift; None = use all data. Pick with select_window.py.
    start_date: str = None
    end_date: str = None

    # ---- VP SDE ----
    beta_min: float = 0.01
    beta_max: float = 10.0
    eps0: float = 1e-3                 # avoid t=0 singularity during training/sampling

    # ---- pretrain: score backbone (unconditional diffusion) ----
    # Score matching is the harder task, so this network is given a richer
    # architecture than the h-function. Tune independently.
    pre_d_model: int = 256
    pre_n_heads: int = 16
    pre_n_layers: int = 8
    pre_dim_ff: int = 512
    pre_dropout: float = 0.0
    pre_epochs: int = 500
    pre_batch_size: int = 256
    pre_lr: float = 1e-4
    ema_decay: float = 0.999

    # ---- h-function: time-dependent classifier (lighter network) ----
    h_d_model: int = 256
    h_n_heads: int = 8
    h_n_layers: int = 6
    h_dim_ff: int = 256
    h_dropout: float = 0.0
    h_epochs: int = 500
    h_batch_size: int = 256
    h_lr: float = 1e-4
    # Only diffusion times t in [eps0, h_t_max] are used to TRAIN h and to APPLY
    # its guidance during sampling. Near the noise end (t -> 1) the noised data
    # carries almost no class signal, so restricting to small t (near clean data)
    # stabilizes h-training and avoids guiding with an untrained noise regime.
    h_t_max: float = 1.0
    # Positive-class weight for BCEWithLogitsLoss (the event is only ~10% of days).
    # <= 0 means AUTO = #neg/#pos (≈9 here); set a positive value to override and
    # study sensitivity to the imbalance weighting.
    h_pos_weight: float = -1.0

    # ---- sampling ----
    n_sample: int = 10000             # M samples for histograms / correlation
    n_steps: int = 100                # Euler-Maruyama reverse steps
    sample_batch: int = 1000          # mini-batch size for sampling (GPU memory)
    delta: float = 1e-3               # numerical floor in log(h + delta)  (note eq 14)
    gamma: float = 2.0                # guidance strength (1.0 = exact Doob)

    # ---- misc ----
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fig_dir: str = "figures"
    ckpt_dir: str = "ckpt"
    tickers: List[str] = field(default_factory=lambda: list(TICKERS))

    def htag(self) -> str:
        """Tag for h-function artifacts (ckpt + figures), encoding the event quantile
        and h_t_max, so different (quantile, t_max) runs never overwrite each other.
        E.g. event_quantile=0.99, h_t_max=0.6 -> 'q99_tmax0.6'."""
        q = round(self.event_quantile * 100, 4)
        return f"q{q:g}_tmax{self.h_t_max:g}"

    def hfunction_ckpt(self) -> str:
        return os.path.join(self.ckpt_dir, f"hfunction_{self.htag()}.pt")

    @property
    def data_dim(self) -> int:
        """Flattened window dimension fed to the SDE/model = seq_len * n_assets (10*10=100)."""
        return self.seq_len * self.n_assets

    def resolve_state(self) -> str:
        """Locate the inflation-state CSV: cfg path (latent_state_estimation output
        by default), else the legacy Desktop / project-dir copies."""
        for p in (self.state_csv,
                  os.path.join(_HERE, "latent_state_estimation", "inflation_state.csv"),
                  os.path.expanduser("~/Desktop/inflation_state.csv"),
                  os.path.join(_HERE, "inflation_state.csv")):
            if p and os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"No inflation state CSV found (tried {self.state_csv} + legacy fallbacks). "
            f"Build it with: python explore/import_data.py --state")

    def resolve_csv(self) -> str:
        """Locate the stock price CSV: cfg path (explore/import_data.py output by
        default), else the legacy Desktop / project-dir tech_stocks_tips.csv."""
        for p in (self.csv_path,
                  os.path.join(_HERE, "explore", "macro_data_new.csv"),
                  os.path.expanduser("~/Desktop/tech_stocks_tips.csv"),
                  os.path.join(_HERE, "tech_stocks_tips.csv")):
            if p and os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"No stock price CSV found (tried {self.csv_path} + legacy fallbacks). "
            f"Build it with: python explore/import_data.py --stocks")
