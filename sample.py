"""Unconditional sampling and Doob h-guided conditional sampling.

Samples are produced in standardized space; callers inverse-standardize for plots.
"""
import torch

from sde import VPSDE


def _batched(fn, total, batch, n_assets, device):
    out = []
    done = 0
    while done < total:
        b = min(batch, total - done)
        out.append(fn(b))
        done += b
    return torch.cat(out, dim=0)


def sample_unconditional(cfg, score_model):
    sde = VPSDE(cfg)
    score_model.eval()

    def step(b):
        return sde.sample(score_model, (b, cfg.n_assets), cfg.device, cfg.n_steps, cfg.eps0)

    print(f"[sample] unconditional: {cfg.n_sample} samples, {cfg.n_steps} steps")
    return _batched(step, cfg.n_sample, cfg.sample_batch, cfg.n_assets, cfg.device)


def sample_conditional(cfg, score_model, h_model):
    sde = VPSDE(cfg)
    score_model.eval()
    h_model.eval()

    def step(b):
        return sde.sample_guided(score_model, h_model, (b, cfg.n_assets), cfg.device,
                                 cfg.n_steps, cfg.eps0, delta=cfg.delta, gamma=cfg.gamma,
                                 h_t_max=cfg.h_t_max)

    print(f"[sample] Doob-conditional: {cfg.n_sample} samples, "
          f"gamma={cfg.gamma}, h_t_max={cfg.h_t_max}")
    return _batched(step, cfg.n_sample, cfg.sample_batch, cfg.n_assets, cfg.device)
