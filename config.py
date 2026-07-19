"""All hyper-parameters and constants for the rate-shock conditional diffusion project.

Everything trainable / tunable lives here as a single dataclass so it can be
overridden from the command line (see main.py) and driven from run.sh.
"""
from dataclasses import dataclass, field
from typing import List
import os
import torch


# Stock universe (order is fixed; data.py will re-detect from the CSV header anyway).
TICKERS: List[str] = ["IBM", "CSCO", "AAPL", "MSFT", "ORCL",
                      "INTC", "TXN", "QCOM", "AMAT", "ADBE"]


@dataclass
class Config:
    # ---- data ----
    csv_path: str = os.path.expanduser("~/Desktop/tech_stocks_tips.csv")
    # inflation-pressure state (from the inflation_pressure_state project); its daily
    # increment delta_s drives the surge event. Falls back to a same-named file in
    # the project dir (see resolve_state).
    state_csv: str = os.path.expanduser("~/Desktop/inflation_state.csv")
    train_frac: float = 0.80          # time-ordered split, no shuffle
    n_assets: int = 10
    # ---- 10x10 window ----
    seq_len: int = 10                 # trading days per window (each sample is 10 days x 10 stocks)
    window_shift: int = 2             # sliding stride between consecutive training windows
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
    pre_n_heads: int = 8
    pre_n_layers: int = 6
    pre_dim_ff: int = 512
    pre_dropout: float = 0.0
    pre_epochs: int = 300
    pre_batch_size: int = 256
    pre_lr: float = 2e-4
    ema_decay: float = 0.999

    # ---- h-function: time-dependent classifier (lighter network) ----
    h_d_model: int = 128
    h_n_heads: int = 4
    h_n_layers: int = 4
    h_dim_ff: int = 256
    h_dropout: float = 0.0
    h_epochs: int = 300
    h_batch_size: int = 256
    h_lr: float = 2e-4
    # Only diffusion times t in [eps0, h_t_max] are used to TRAIN h and to APPLY
    # its guidance during sampling. Near the noise end (t -> 1) the noised data
    # carries almost no class signal, so restricting to small t (near clean data)
    # stabilizes h-training and avoids guiding with an untrained noise regime.
    h_t_max: float = 0.6
    # Positive-class weight for BCEWithLogitsLoss (the event is only ~10% of days).
    # <= 0 means AUTO = #neg/#pos (≈9 here); set a positive value to override and
    # study sensitivity to the imbalance weighting.
    h_pos_weight: float = -1.0

    # ---- sampling ----
    n_sample: int = 20000             # M samples for histograms / correlation
    n_steps: int = 500                # Euler-Maruyama reverse steps
    sample_batch: int = 4000          # mini-batch size for sampling (GPU memory)
    delta: float = 1e-3               # numerical floor in log(h + delta)  (note eq 14)
    gamma: float = 1.0                # guidance strength (1.0 = exact Doob)

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
        """Locate the inflation-state CSV: ~/Desktop path, else the project dir copy."""
        if os.path.exists(self.state_csv):
            return self.state_csv
        local = os.path.join(os.path.dirname(__file__), "inflation_state.csv")
        if os.path.exists(local):
            return local
        raise FileNotFoundError(
            f"Could not find inflation_state.csv at {self.state_csv} or in the project dir.")

    def resolve_csv(self) -> str:
        """Spec: use ~/Desktop path; fall back to a same-named file in the project dir."""
        if os.path.exists(self.csv_path):
            return self.csv_path
        local = os.path.join(os.path.dirname(__file__), "tech_stocks_tips.csv")
        if os.path.exists(local):
            return local
        raise FileNotFoundError(
            f"Could not find tech_stocks_tips.csv at {self.csv_path} or in the project dir.")
