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
from data import load_data, print_data_stats
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
    p.add_argument("--stage", choices=["stats", "pretrain", "hfunction", "sample", "all"],
                   default="all", help="which part of the pipeline to run")
    # data / device
    p.add_argument("--csv-path", default=cfg.csv_path)
    p.add_argument("--start-date", default=cfg.start_date, help="data window start YYYY-MM-DD")
    p.add_argument("--end-date", default=cfg.end_date, help="data window end YYYY-MM-DD")
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


def load_backbone(cfg):
    path = os.path.join(cfg.ckpt_dir, "pretrain.pt")
    sd = torch.load(path, map_location=cfg.device)
    m = TransformerScore(cfg).to(cfg.device)
    m.load_state_dict(sd["ema"])           # use EMA weights for sampling
    print(f"[main] loaded backbone (EMA) from {path}")
    return m


def load_hfunction(cfg):
    path = cfg.hfunction_ckpt()                 # keyed by h_t_max
    sd = torch.load(path, map_location=cfg.device)
    m = TransformerClassifier(cfg).to(cfg.device)
    m.load_state_dict(sd["model"])
    print(f"[main] loaded h-function (h_t_max={cfg.h_t_max:g}) from {path}")
    return m


def stage_sample(cfg, data, score_model=None, h_model=None):
    if score_model is None:
        score_model = load_backbone(cfg)
    if h_model is None:
        h_model = load_hfunction(cfg)

    X_uncond = sample_unconditional(cfg, score_model)
    X_cond = sample_conditional(cfg, score_model, h_model)

    mu, sd, tickers = data["mu"], data["sd"], data["tickers"]
    X_train_all, X_test_all = data["X_train"], data["X_test"]
    train_mask = data["B_train"] > 0.5
    test_mask = data["B_test"] > 0.5
    X_event_train = data["X_train"][train_mask]       # in-sample reference
    X_event_test = data["X_test"][test_mask]          # out-of-sample reference
    print(f"[main] event days: train={int(train_mask.sum())}  test={int(test_mask.sum())}")

    # --- per-stock marginal stats (mean ± std), printed to log ---
    viz.print_marginal_stats({
        "train_uncond": X_train_all,   "train_cond": X_event_train,
        "test_uncond":  X_test_all,    "test_cond":  X_event_test,
        "gen_uncond":   X_uncond,      "gen_cond":   X_cond,
    }, mu, sd, tickers)

    # outputs go under figures/tmax{h_t_max}/ , filenames tagged with gamma so a
    # t_max sweep (and a gamma sweep within it) never overwrites earlier figures.
    sub = cfg.htag()                       # e.g. "tmax0.6"
    g = f"g{cfg.gamma:g}"                   # e.g. "g1"
    tag = f"{sub}_{g}"

    # --- histograms ---
    viz.hist_compare(cfg, X_cond, X_event_train, mu, sd, tickers,
                     f"{sub}/hist_insample_{g}.png",
                     f"In-sample [{tag}]: conditional generated vs TRAIN event-day logret",
                     gen_label="conditional", actual_label="train event")
    viz.hist_compare(cfg, X_cond, X_event_test, mu, sd, tickers,
                     f"{sub}/hist_outsample_{g}.png",
                     f"Out-of-sample [{tag}]: conditional generated vs TEST event-day logret",
                     gen_label="conditional", actual_label="test event")

    # --- correlation grid (2x3): actual train/test (all/event) vs generated (uncond/Doob) ---
    viz.corr_grid(cfg, X_train_all, X_test_all, X_uncond,
                  X_event_train, X_event_test, X_cond, mu, sd, tickers,
                  filename=f"{sub}/corr_grid_{g}.png", tag=tag)


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
        viz.hist_pretrain_vs_actual(cfg, X_gen, data["X_test"],
                                    data["mu"], data["sd"], data["tickers"])
    if stage in ("hfunction", "all"):
        h_model = train_hfunction(cfg, data)
    if stage in ("sample", "all"):
        stage_sample(cfg, data, score_model, h_model)

    print("[main] done. See figures/ and ckpt/.")


if __name__ == "__main__":
    main()
