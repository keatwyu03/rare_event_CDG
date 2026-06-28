"""VP SDE with a linear beta schedule (closed forms), forward noising and
reverse Euler-Maruyama sampling (unconditional + Doob h-guided).

Time convention: t in [0,1], t=0 is data, t=1 is noise (Song et al. score-SDE).
"""
import torch


class VPSDE:
    def __init__(self, cfg):
        self.bmin = cfg.beta_min
        self.bmax = cfg.beta_max

    # --- closed forms (all accept a batched t vector) ---
    def beta(self, t):
        return self.bmin + t * (self.bmax - self.bmin)

    def beta_integral(self, t):
        return self.bmin * t + 0.5 * (self.bmax - self.bmin) * t ** 2

    def alpha(self, t):                       # forward mean coefficient
        return torch.exp(-0.5 * self.beta_integral(t))

    def sigma(self, t):                       # forward std ; sigma^2 = 1 - alpha^2
        return torch.sqrt(1.0 - torch.exp(-self.beta_integral(t)))

    # --- forward noising (training) ---
    def forward_sample(self, x0, t, z=None):
        """x_t = alpha(t) x0 + sigma(t) z ,  with t shaped (B,) broadcast over assets."""
        if z is None:
            z = torch.randn_like(x0)
        a = self.alpha(t).unsqueeze(-1)
        s = self.sigma(t).unsqueeze(-1)
        return a * x0 + s * z, z

    def score_from_eps(self, eps, t):
        """score = -eps / sigma(t)."""
        return -eps / self.sigma(t).unsqueeze(-1)

    # --- reverse sampling (Euler-Maruyama, t: 1 -> 0, dt < 0) ---
    @torch.no_grad()
    def sample(self, score_model, shape, device, n_steps, eps0):
        x = torch.randn(shape, device=device)
        ts = torch.linspace(1.0, eps0, n_steps + 1, device=device)
        for i in range(n_steps):
            t = ts[i]
            dt = ts[i + 1] - ts[i]                     # negative
            tb = t.expand(shape[0])
            eps = score_model(x, tb)
            score = self.score_from_eps(eps, tb)
            b = self.beta(t)
            drift = -0.5 * b * x - b * score
            noise = torch.randn_like(x) * torch.sqrt(b) * torch.sqrt(-dt)
            x = x + drift * dt + noise
        return x

    def sample_guided(self, score_model, h_model, shape, device, n_steps, eps0,
                      delta=1e-3, gamma=1.0, h_t_max=1.0):
        """Doob-guided reverse SDE: cond_score = score + gamma * grad_x log(h + delta).

        Guidance is applied only for t <= h_t_max (the range where h was trained);
        for larger t (close to pure noise) we fall back to the unconditional score.
        """
        x = torch.randn(shape, device=device)
        ts = torch.linspace(1.0, eps0, n_steps + 1, device=device)
        for i in range(n_steps):
            t = ts[i]
            dt = ts[i + 1] - ts[i]
            tb = t.expand(shape[0])

            with torch.no_grad():
                eps = score_model(x, tb)
                score = self.score_from_eps(eps, tb)

            if t <= h_t_max:
                # guidance term: grad_x log(h + delta) via autograd
                x_g = x.detach().requires_grad_(True)
                h = torch.sigmoid(h_model(x_g, tb))       # (B,)
                torch.log(h + delta).sum().backward()
                guidance = x_g.grad.detach()
            else:
                guidance = 0.0

            cond_score = score + gamma * guidance
            b = self.beta(t)
            drift = -0.5 * b * x - b * cond_score
            noise = torch.randn_like(x) * torch.sqrt(b) * torch.sqrt(-dt)
            x = (x + drift * dt + noise).detach()
        return x
