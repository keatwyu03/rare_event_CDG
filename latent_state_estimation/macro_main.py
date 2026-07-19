"""LatentStateEstimator — estimates a daily latent macro state from the
monthly macro panels + daily futures tracking portfolios in this directory
(ported from the cdg_finance / diffusion_stress_testing repo).

Default is the JOINT growth+inflation state (both monthly factors and both
daily tracking portfolios combine into one vector observation for a single
Kalman filter — same as the cdg_finance default); pass
variables=("inflation",) for the scalar inflation-only case.

method:
    "state_space"         — Kalman filter: daily tracking portfolio(s) drive
                            one latent state, observed through the monthly
                            factor(s) via an intramonth cumulator
    "tracking_regression" — standardized average of the daily tracking
                            portfolios (no Kalman filter)
"""
import os
import sys

import pandas as pd

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
from tracking_regression import TrackingRegression
from state_space import StateSpace


class LatentStateEstimator:

    def __init__(self, method: str = "state_space",
                 variables=("growth", "inflation"), data_dir: str = _dir):
        self.method = method
        self.variables = tuple(variables)
        self.data_dir = data_dir
        self.trackers = {}       # name -> fitted TrackingRegression
        self.state_space = None  # fitted StateSpace (state_space method only)
        self.latent = None

    def fit(self) -> pd.Series:
        for name in self.variables:
            macro = pd.read_csv(os.path.join(self.data_dir, f"{name}_macro.csv"),
                                index_col=0, parse_dates=True)
            daily = pd.read_csv(os.path.join(self.data_dir, f"{name}_daily.csv"),
                                index_col=0, parse_dates=True)
            tr = TrackingRegression(macro, daily)
            tr.fit()
            self.trackers[name] = tr
            print(f"[latent] {name}: PCA explained var {tr.explained_var:.3f}, "
                  f"daily tracking portfolio {len(tr.ut)} days")

        uts = pd.concat({n: tr.ut for n, tr in self.trackers.items()}, axis=1).dropna()

        if self.method == "state_space":
            factors = pd.concat({n: tr.factor for n, tr in self.trackers.items()}, axis=1)
            self.state_space = StateSpace(y=factors, x=uts).fit()
            res = self.state_space.res
            print(f"[latent] Kalman MLE: loglik={-res.fun:.1f}  converged={res.success}")
            print("[latent] params: " + "  ".join(
                f"{k}={v:.4f}" for k, v in
                zip(self.state_space.param_names, self.state_space.params)))
            self.latent = self.state_space.filtered_states()
        elif self.method == "tracking_regression":
            z = (uts - uts.mean()) / uts.std()
            self.latent = z.mean(axis=1).rename("latent")
        else:
            raise ValueError(f"unknown latent method: {self.method!r}")
        return self.latent
