# Rate-Shock Conditional Constrained Diffusion

Model the **joint cross-sectional distribution** of 10 tech-stock daily log returns with a
VP-SDE diffusion model, train a time-dependent classifier (h-function), and use a
**Doob h-transform** to generate samples conditioned on "rate-shock" days (top-10% daily
change in the 10y TIPS real yield, `Δy`). Goal: see how the 10×10 correlation structure
changes under extreme rate shocks.

> Method: Renyuan's note *Conditional Constrained Diffusion* — Bayes tilting + Doob
> h-transform + directly trained time-dependent classifier. `Z = Δy` is used **only** to
> build the event label `B`; it never enters any network.

## Install

```bash
pip install -r requirements.txt
```

## Data

Both inputs are built by **`python explore/import_data.py`** (single injection
point, mirroring cdg_finance):

- `explore/macro_data_new.csv` — daily adjusted-close **prices** for the 10
  tickers (IBM, CSCO, AAPL, MSFT, ORCL, INTC, TXN, QCOM, AMAT, ADBE) from
  yfinance; `data.py` computes log returns itself. → `config.csv_path`
- `latent_state_estimation/inflation_state.csv` — the latent
  **inflation-pressure state**: `s` = Kalman-filtered daily state level,
  `delta_s` = its daily increment (drives the surge event label).
  → `config.state_csv`

The state is estimated by `latent_state_estimation/` (logic matches the
diffusion_stress_testing repo). Both methods start from per-group monthly PC1
factors (growth and inflation get separate PCAs, never a joint one), then:
**state_space** (default) — the RAW daily market variables of both panels,
appended and z-scored, drive one scalar Kalman-filtered state anchored at
month-ends by both monthly factors (MLE, Nelder-Mead — can take a while; no
tracking regression involved); **tracking_regression**
(`--method tracking_regression`) — per-group tracking regressions produce
daily tracking portfolios whose standardized average is the state (no Kalman
filter). The committed macro panel CSVs are used as-is — refresh them (rarely)
with
`FRED_API_KEY=... python latent_state_estimation/macro_importer.py`.

If the built files are missing, the loader falls back to the legacy
`~/Desktop/tech_stocks_tips.csv` / project-dir copies automatically.

## Run

```bash
bash run.sh                 # full pipeline; auto-uses GPU if available
bash run.sh pretrain        # train the diffusion backbone only
bash run.sh hfunction       # train the h-function only (independent of backbone)
bash run.sh sample          # load both ckpts -> conditional generation + plots
GPU=1 bash run.sh           # pick CUDA device 1
```

`pretrain` and `hfunction` are **fully independent** (neither needs the other's
checkpoint) and can run in any order, on different machines/GPUs. `sample` needs
both `ckpt/pretrain.pt` and `ckpt/hfunction.pt` to exist.

All key training params (epochs, batch size, lr, model size, sampling steps, Doob `gamma`,
and `H_T_MAX`) live at the top of [run.sh](run.sh) — edit and relaunch. They are passed to
[main.py](main.py); defaults also live in [config.py](config.py).

### `H_T_MAX` (h-function time range)

The h-function is trained and applied only for diffusion times `t ∈ [eps0, H_T_MAX]`,
i.e. the part of the trajectory **near clean data** (`t=0`). Near the noise end (`t→1`)
the noised sample carries almost no class signal, so including it weakens training and
guiding there just uses an untrained regime. During conditional sampling, steps with
`t > H_T_MAX` fall back to the pure unconditional score. Default `0.6`; lower it if the
classifier struggles, raise it toward `1.0` to guide earlier in the reverse process.

## Pipeline (main.py)

1. **data.py** — logret + Δy, time-ordered 80/20 split (no shuffle), per-column
   standardization (train stats only), event labels from the train 90th-percentile Δy.
2. **train_pretrain.py** — unconditional ε-prediction diffusion backbone (+ EMA weights).
3. **fig 1** — generated vs actual marginal histograms.
4. **train_hfunction.py** — time-dependent binary classifier (`t ≤ H_T_MAX`) with
   `pos_weight` for the 10/90 imbalance; prints AUC (warns if ≈0.5).
5. **sample.py** — unconditional + Doob h-guided conditional sampling.
6. **fig 2** — four-panel 10×10 correlation comparison + Frobenius distances.

## Outputs

| file | meaning |
|------|---------|
| `ckpt/pretrain.pt` | backbone weights (raw + EMA) |
| `ckpt/hfunction.pt` | h-function classifier weights |
| `figures/loss_pretrain.png` / `loss_hfunction.png` | training curves |
| `figures/hist_pretrain_vs_actual.png` | backbone marginal sanity check |
| `figures/hist_insample.png` | **in-sample**: conditional generated vs TRAIN event-day marginals |
| `figures/hist_outsample.png` | **out-of-sample**: conditional generated vs TEST event-day marginals |
| `figures/corr_grid.png` | **main result**: 2×3 correlation grid — row1 train-all/test-all/uncond-gen, row2 train-event/test-event/Doob-gen |

**Key check:** `||corr(cond) − corr(actual_event)||_F` should be **smaller** than
`||corr(uncond) − corr(actual_event)||_F` — i.e. conditioning pulls the correlation
structure toward real rate-shock days (typically a broad rise / "clustering" of
cross-sectional correlation).

## Analysis & evaluation (`analysis/`, ported from cdg_finance)

`bash run.sh analysis` runs every script below against the pipeline outputs:
loss CSVs in `ckpt/`, checkpoints, and the generated windows that
`run.sh sample` saves to `results/samples/samples_{htag}_g{gamma}.pt`.
Outputs land in `results/analysis/<htag>/`. All comparisons are in
**standardized (z) space** and use the **last day of each 10-day window**
(the ACF script alone uses the full day axis — it measures within-window
volatility clustering).

| script | what it shows |
|---|---|
| `analysis/losses.py` | pretrain + h-function loss curves (from `ckpt/*_losses*.csv`) |
| `analysis/h_function_eval.py` | h calibration vs noise level τ, split by true event label (no sampling involved) |
| `analysis/unconditional_gen.py` | real (all) vs uncond-generated last-day KDEs + diagnostics table |
| `analysis/conditional_gen.py` | real EVENT windows vs Doob-conditional last-day KDEs + table |
| `analysis/cov.py` | last-day 10×10 corr & cov heatmaps: real all/event vs uncond/cond + Frobenius check |
| `analysis/distribution_metrics.py` | per-stock Wasserstein-1 distances + log-log tail plots (last-day) |
| `analysis/dependency_metric.py` | ACF of squared returns (vol clustering) real vs generated + t-test p-values |

Each script also runs standalone, e.g.
`python analysis/cov.py --event-quantile 0.9 --h-t-max 1 --gamma 2` — the flags
must match the run that produced the checkpoints/samples (files are keyed by them).

## Math reference (VP SDE, t∈[0,1], t=0 data → t=1 noise)

```
beta(t)  = bmin + t (bmax - bmin)
B(t)     = bmin t + 0.5 (bmax - bmin) t^2
alpha(t) = exp(-0.5 B(t)) ,  sigma(t) = sqrt(1 - exp(-B(t)))
x_t      = alpha(t) x0 + sigma(t) z,   z ~ N(0,I)
score    = -eps_theta / sigma
reverse  : drift = -0.5 beta x - beta * cond_score,   x += drift dt + sqrt(beta) sqrt(-dt) N(0,I)
Doob     : cond_score = score + gamma * grad_x log(h + 1e-3)   (autograd w.r.t. x)
```
