"""Run the pipeline, by stage.

Stages (each reloads data deterministically, so they are fully independent):
  pretrain  : train unconditional diffusion backbone (+EMA) -> ckpt/pretrain.pt
              + fig1 (generated vs actual marginals)
  hfunction : train time-dependent classifier h           -> ckpt/hfunction.pt
  sample    : load both ckpts, unconditional + Doob-conditional sampling
              + fig2 (correlation comparison) + Frobenius distances
  all       : pretrain -> hfunction -> sample  (default)

pretrain and hfunction do NOT depend on each other and can be run in any order;
`sample` needs both checkpoints to exist.

Key training params come from config.py defaults, overridable on the command
line (run.sh wires the common ones).
"""
import argparse
import os
import numpy as np
import torch

from config import Config
from data import load_data, print_data_stats, destandardize, sample_entries
from models import TransformerScore, TransformerClassifier
from train_pretrain import train_pretrain
from train_hfunction import train_hfunction
from sample import sample_unconditional, sample_conditional
import viz


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config():
    cfg = Config()
    p = argparse.ArgumentParser(description="Rate-shock conditional diffusion")
    p.add_argument("--stage",
                   choices=["stats", "pretrain", "hfunction", "sample", "all"],
                   default="all", help="which part of the pipeline to run")
    # data / device
    p.add_argument("--csv-path", default=cfg.csv_path)
    p.add_argument("--state-csv", default=cfg.state_csv,
                   help="latent inflation-state CSV (delta_s drives the event label)")
    p.add_argument("--start-date", default=cfg.start_date, help="data window start YYYY-MM-DD")
    p.add_argument("--end-date", default=cfg.end_date, help="data window end YYYY-MM-DD")
    p.add_argument("--event-quantile", type=float, default=cfg.event_quantile,
                   help="Δy quantile defining the event (0.90=top10%%, 0.99=top1%%/rarer)")
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--gpu", type=int, default=None, help="CUDA device index to use")
    p.add_argument("--seed", type=int, default=cfg.seed)
    # pretrain: score backbone arch + training
    p.add_argument("--pre-d-model", type=int, default=cfg.pre_d_model)
    p.add_argument("--pre-n-heads", type=int, default=cfg.pre_n_heads)
    p.add_argument("--pre-n-layers", type=int, default=cfg.pre_n_layers)
    p.add_argument("--pre-dim-ff", type=int, default=cfg.pre_dim_ff)
    p.add_argument("--pre-dropout", type=float, default=cfg.pre_dropout)
    p.add_argument("--pre-epochs", type=int, default=cfg.pre_epochs)
    p.add_argument("--pre-batch-size", type=int, default=cfg.pre_batch_size)
    p.add_argument("--pre-lr", type=float, default=cfg.pre_lr)
    # h-function: classifier arch + training
    p.add_argument("--h-d-model", type=int, default=cfg.h_d_model)
    p.add_argument("--h-n-heads", type=int, default=cfg.h_n_heads)
    p.add_argument("--h-n-layers", type=int, default=cfg.h_n_layers)
    p.add_argument("--h-dim-ff", type=int, default=cfg.h_dim_ff)
    p.add_argument("--h-dropout", type=float, default=cfg.h_dropout)
    p.add_argument("--h-epochs", type=int, default=cfg.h_epochs)
    p.add_argument("--h-batch-size", type=int, default=cfg.h_batch_size)
    p.add_argument("--h-lr", type=float, default=cfg.h_lr)
    p.add_argument("--h-t-max", type=float, default=cfg.h_t_max,
                   help="max diffusion time used to train/apply h (near clean data)")
    p.add_argument("--h-pos-weight", type=float, default=cfg.h_pos_weight,
                   help="BCE positive-class weight; <=0 means auto (#neg/#pos)")
    # sampling
    p.add_argument("--n-sample", type=int, default=cfg.n_sample)
    p.add_argument("--n-steps", type=int, default=cfg.n_steps)
    p.add_argument("--sample-batch", type=int, default=cfg.sample_batch)
    p.add_argument("--gamma", type=float, default=cfg.gamma)
    args = p.parse_args()

    for k, v in vars(args).items():
        if k in ("gpu", "stage"):
            continue
        setattr(cfg, k.replace("-", "_"), v)
    if args.gpu is not None and torch.cuda.is_available():
        cfg.device = f"cuda:{args.gpu}"
    cfg.start_date = cfg.start_date or None     # empty string -> None
    cfg.end_date = cfg.end_date or None
    return cfg, args.stage


def _apply_arch(cfg, sd):
    """Rebuild-time arch override: use the architecture stored in the ckpt so a
    changed run.sh/config can't cause a state_dict mismatch. Older ckpts without
    'arch' fall back to the current cfg (set matching PRE_*/H_* manually)."""
    for k, v in sd.get("arch", {}).items():
        setattr(cfg, k, v)


def load_backbone(cfg):
    path = os.path.join(cfg.ckpt_dir, "pretrain.pt")
    sd = torch.load(path, map_location=cfg.device)
    _apply_arch(cfg, sd)
    m = TransformerScore(cfg).to(cfg.device)
    m.load_state_dict(sd["ema"])           # use EMA weights for sampling
    print(f"[main] loaded backbone (EMA) from {path}"
          f"{' [arch from ckpt]' if 'arch' in sd else ''}")
    return m


def load_hfunction(cfg):
    path = cfg.hfunction_ckpt()                 # keyed by h_t_max
    sd = torch.load(path, map_location=cfg.device)
    _apply_arch(cfg, sd)
    m = TransformerClassifier(cfg).to(cfg.device)
    m.load_state_dict(sd["model"])
    print(f"[main] loaded h-function (h_t_max={cfg.h_t_max:g}) from {path}"
          f"{' [arch from ckpt]' if 'arch' in sd else ''}")
    return m


def stage_sample(cfg, data, score_model=None, h_model=None):
    if score_model is None:
        score_model = load_backbone(cfg)
    if h_model is None:
        h_model = load_hfunction(cfg)

    X_uncond = sample_unconditional(cfg, score_model)
    X_cond = sample_conditional(cfg, score_model, h_model)

    # persist generated windows (standardized space) for the analysis scripts,
    # keyed by (event_quantile, h_t_max, gamma) so sweep runs never collide
    sdir = os.path.join("results", "samples")
    os.makedirs(sdir, exist_ok=True)
    spath = os.path.join(sdir, f"samples_{cfg.htag()}_g{cfg.gamma:g}.pt")
    torch.save({"X_uncond": X_uncond.cpu(), "X_cond": X_cond.cpu(),
                "event_quantile": cfg.event_quantile, "h_t_max": cfg.h_t_max,
                "gamma": cfg.gamma}, spath)
    print(f"[main] saved generated windows -> {spath}")

    seq, n, tickers = data["seq_len"], data["n_assets"], data["tickers"]
    tm = data["B_train"] > 0.5
    te = data["B_test"] > 0.5
    tm_np, te_np = tm.cpu().numpy(), te.cpu().numpy()
    print(f"[main] event windows: train={int(tm.sum())}  test={int(te.sum())}")

    # entry (mu,sig) pools that set the raw-logret scale for de-standardization
    mtr, str_ = data["mu_entry_train"], data["sig_entry_train"]
    mte, ste = data["mu_entry_test"], data["sig_entry_test"]

    def dstd(X, mu_e, sig_e):
        return destandardize(X, mu_e, sig_e, seq, n)

    # actual -> raw logret using each window's OWN entry EMA
    tr_all = dstd(data["X_train"], mtr, str_)
    te_all = dstd(data["X_test"], mte, ste)
    tr_evt = dstd(data["X_train"][tm], mtr[tm_np], str_[tm_np])
    te_evt = dstd(data["X_test"][te], mte[te_np], ste[te_np])

    # generated -> raw logret using entry (mu,sig) SAMPLED from the comparison set
    gu_tr = dstd(X_uncond, *sample_entries(mtr, str_, len(X_uncond), cfg.seed))
    gc_tr = dstd(X_cond,   *sample_entries(mtr, str_, len(X_cond),   cfg.seed))
    gu_te = dstd(X_uncond, *sample_entries(mte, ste, len(X_uncond), cfg.seed))
    gc_te = dstd(X_cond,   *sample_entries(mte, ste, len(X_cond),   cfg.seed))

    # global window indices for deduped-neighborhood pooling (calendar day = g*shift + d)
    n_train = data["X_train"].shape[0]
    shift = cfg.window_shift
    g_tr_all = np.arange(n_train)
    g_te_all = n_train + np.arange(data["X_test"].shape[0])
    g_tr_evt = np.where(tm_np)[0]
    g_te_evt = n_train + np.where(te_np)[0]

    # --- per-stock marginal stats (mean ± std), printed to log ---
    viz.print_marginal_stats({
        "train_uncond": tr_all,   "train_cond": tr_evt,
        "test_uncond":  te_all,   "test_cond":  te_evt,
        "gen_uncond(tr)": gu_tr,  "gen_cond(tr)": gc_tr,
        "gen_uncond(te)": gu_te,  "gen_cond(te)": gc_te,
    }, tickers)

    # outputs go under figures/tmax{h_t_max}/ , filenames tagged with gamma so a
    # t_max sweep (and a gamma sweep within it) never overwrites earlier figures.
    sub = cfg.htag()                       # e.g. "tmax0.6"
    g = f"g{cfg.gamma:g}"                   # e.g. "g1"
    tag = f"{sub}_{g}"

    # --- histograms: actual event side DEDUPED by calendar day; generated pooled ---
    viz.hist_compare(cfg, gc_tr, tr_evt, tickers,
                     f"{sub}/hist_insample_{g}.png",
                     f"In-sample [{tag}]: conditional (train-scaled) vs TRAIN event logret",
                     gen_label="conditional", actual_label="train event",
                     actual_gidx=g_tr_evt, seq=seq, shift=shift)
    viz.hist_compare(cfg, gc_te, te_evt, tickers,
                     f"{sub}/hist_outsample_{g}.png",
                     f"Out-of-sample [{tag}]: conditional (test-scaled) vs TEST event logret",
                     gen_label="conditional", actual_label="test event",
                     actual_gidx=g_te_evt, seq=seq, shift=shift)

    # --- correlation grid (2x3), TWO versions ---
    # (a) raw logret: actual de-standardized w/ own entries, generated w/ entries
    #     sampled from the comparison set (train/test-scaled). Reflects the real
    #     (vol-weighted) correlation but mixes in the de-standardization/entry pairing.
    viz.corr_grid(cfg,
                  {"train all": (tr_all, g_tr_all), "test all": (te_all, g_te_all),
                   "train event": (tr_evt, g_tr_evt), "test event": (te_evt, g_te_evt)},
                  gu_tr, gc_tr, gu_te, gc_te, tickers, seq, shift,
                  filename=f"{sub}/corr_grid_{g}.png", tag=tag, space="raw logret")

    # (b) standardized (z) space: no de-standardization. Isolates how well the model
    #     learned the vol-normalized cross-sectional structure (no entry/vol confound;
    #     generated z is used directly for both train and test comparisons).
    Xtr, Xte = data["X_train"], data["X_test"]
    viz.corr_grid(cfg,
                  {"train all": (Xtr, g_tr_all), "test all": (Xte, g_te_all),
                   "train event": (Xtr[tm], g_tr_evt), "test event": (Xte[te], g_te_evt)},
                  X_uncond, X_cond, X_uncond, X_cond, tickers, seq, shift,
                  filename=f"{sub}/corr_grid_std_{g}.png", tag=tag, space="standardized")


def main():
    cfg, stage = build_config()
    set_seed(cfg.seed)

    print(f"[main] CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
        idx = 0 if ":" not in cfg.device else int(cfg.device.split(":")[1])
        if idx >= torch.cuda.device_count():       # e.g. masked by CUDA_VISIBLE_DEVICES
            print(f"[main] WARN: {cfg.device} not visible (device_count="
                  f"{torch.cuda.device_count()}); falling back to cuda:0", flush=True)
            cfg.device, idx = "cuda:0", 0
        print(f"[main] using {cfg.device} -> {torch.cuda.get_device_name(idx)}", flush=True)
    print(f"[main] stage={stage}  device={cfg.device}  seed={cfg.seed}")

    data = load_data(cfg)

    if stage == "stats":
        print_data_stats(data)
        return

    score_model = h_model = None
    if stage in ("pretrain", "all"):
        score_model = train_pretrain(cfg, data)
        X_gen = sample_unconditional(cfg, score_model)
        seq, n = data["seq_len"], data["n_assets"]
        te_all = destandardize(data["X_test"], data["mu_entry_test"], data["sig_entry_test"], seq, n)
        gen = destandardize(X_gen, *sample_entries(data["mu_entry_test"], data["sig_entry_test"],
                                                   len(X_gen), cfg.seed), seq, n)
        g_te_all = data["X_train"].shape[0] + np.arange(data["X_test"].shape[0])
        viz.hist_pretrain_vs_actual(cfg, gen, te_all, data["tickers"],
                                    actual_gidx=g_te_all, seq=seq, shift=cfg.window_shift)
    if stage in ("hfunction", "all"):
        h_model = train_hfunction(cfg, data)
    if stage in ("sample", "all"):
        stage_sample(cfg, data, score_model, h_model)

    print("[main] done. See figures/ and ckpt/.")


if __name__ == "__main__":
    main()
