"""LeWM with a conservative-field sampler for p(z_{t+1} | history, a_t): v = -grad_x U.

Identical to the bridge repo's fm/flow_jepa.py::FlowJEPA (github.com/axel-delmas/fm-jepa) except
for one thing: the head outputs a scalar potential U(x_tau, tau | h_t, z_t, a_t) and the velocity
is its negative gradient w.r.t. x_tau (autograd), instead of a free vector field head(...).

The encoder, projector, action embedder and the ARPredictor trunk are the repo's own modules.
pred_proj is removed; the trunk output h_t conditions a small per-pair PotentialHead. Noisy states
only ever go through the head: they are never fed back into the trunk. Rollouts stay
autoregressive on sampled z_{t+1} exactly like JEPA.rollout.

Variants (per pair: tau ~ U[0,1], eps ~ N(0, I)); same losses as the bridge's A1 / A2:
  P1 (noise around z_t, ODE):   x0 = z_t + sigma*eps, x_tau = (1-tau) x0 + tau z1,
                                loss = mean ||v - (z1 - x0)||^2; Euler sampling x <- x - dt grad U.
  P2 (Brownian bridge, SDE):    x_tau = (1-tau) z_t + tau z1 + sigma sqrt(tau(1-tau)) eps,
                                x1_hat = x_tau + (1-tau) v, loss = mean ||x1_hat - z1||^2;
                                Euler-Maruyama sampling, no noise on the last step.
sigma is a per-dimension vector: sigma_mult * calibrated std of (z_{t+1} - z_t).

Gradients:
  - training (grad mode on): grad_x U is taken with create_graph=True (double backward), so the
    loss reaches the head parameters, h_t, z_t, a_t and (through x_tau) the encoder, along the
    same paths as the bridge's v = head(x_tau, ...).
  - sampling / no-grad loss evaluation: grad_x U is computed under enable_grad (and outside
    inference_mode), w.r.t. a detached copy of x only, and the result is detached. h_t, z_t, a_t
    and the parameters never receive gradients from it.

TODO (not implemented on purpose): unrolled gradient descent on U as the transition
(z_{t+1} = z_t - sum_k eta_k grad U, trained end to end) and denoising score matching
(U as an energy whose gradient is the score). Both would replace flow_loss/sample_next only.
"""

import contextlib

import torch
from einops import rearrange

from jepa import JEPA


class PotentialJEPA(JEPA):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector,
        head,
        variant="P1",
        sigma_mult=0.0,
        sigma_std=None,
        embed_dim=192,
        plan_S=1,
        plan_K=1,
        plan_mode="sample",
        plan_seed=0,
        detach_target=False,
        head_fp32=False,
    ):
        super().__init__(encoder, predictor, action_encoder, projector=projector, pred_proj=None)
        assert variant in ("P1", "P2"), variant
        self.head = head
        self.variant = variant
        self.detach_target = detach_target
        self.head_fp32 = bool(head_fp32)
        std = torch.ones(embed_dim) if sigma_std is None else torch.as_tensor(sigma_std, dtype=torch.float32)
        assert std.numel() == embed_dim
        self.register_buffer("sigma", float(sigma_mult) * std.float())
        # planning-time settings (not part of the state dict; set by the eval script)
        self.plan_S = int(plan_S)
        self.plan_K = int(plan_K)
        self.plan_mode = plan_mode
        self.plan_seed = int(plan_seed)
        self._plan_gen = None

    # ------------------------------------------------------------------ field
    def _precision(self, x):
        """head_fp32: run the head (and grad_x U) outside autocast, in fp32."""
        if not self.head_fp32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=x.device.type, enabled=False)

    def potential(self, x, tau, h, z_t, a):
        """U(x, tau | h, z_t, a): (N,)."""
        if not self.head_fp32:
            return self.head(x, tau, h, z_t, a)
        with self._precision(x):
            return self.head(x.float(), tau.float(), h.float(), z_t.float(), a.float())

    def velocity(self, x, tau, h, z_t, a, create_graph=False):
        """v = -grad_x U(x, tau | h, z_t, a). All inputs (N, D) except tau (N,).

        create_graph=True (training): differentiable in the head parameters and in all inputs.
        create_graph=False: works under no_grad / inference_mode; the result is detached.
        """
        # the grad call sits inside _precision too: on CPU autograd runs the backward on the calling
        # thread, where an enclosing bf16 autocast would otherwise still apply
        if create_graph:
            assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            xg = x if x.requires_grad else x.detach().requires_grad_(True)
            with self._precision(x):
                U = self.potential(xg, tau, h, z_t, a)
                (g,) = torch.autograd.grad(U.sum(), xg, create_graph=True)
            return -g
        inf = torch.is_inference_mode_enabled()
        # inference tensors cannot be saved for backward: copy them outside inference mode
        cut = (lambda t: t.detach().clone()) if inf else (lambda t: t.detach())
        with torch.inference_mode(False), torch.enable_grad(), self._precision(x):
            xg = cut(x).requires_grad_(True)
            U = self.potential(xg, cut(tau), cut(h), cut(z_t), cut(a))
            (g,) = torch.autograd.grad(U.sum(), xg)
        return -g.detach()

    # ------------------------------------------------------------------ training
    def trunk(self, emb, act_emb):
        """ARPredictor output h (B, T, D); no pred_proj."""
        return self.predictor(emb, act_emb)

    def flow_loss(self, z_t, z1, h, a):
        """All inputs (N, D). Returns (loss_mean, per_pair_loss (N,), tau (N,)).
        Under no_grad (validation) the field is still computed, without a graph."""
        n, d = z_t.shape
        tau = torch.rand(n, device=z_t.device)
        eps = torch.randn_like(z_t, dtype=torch.float32)
        sigma = self.sigma.to(z_t.device)
        t = tau[:, None]
        cg = torch.is_grad_enabled()
        if self.variant == "P1":
            x0 = z_t + sigma * eps
            x_tau = (1 - t) * x0 + t * z1
            v = self.velocity(x_tau, tau, h, z_t, a, create_graph=cg)
            err = (v - (z1 - x0)).float().pow(2)
        else:
            x_tau = (1 - t) * z_t + t * z1 + sigma * torch.sqrt(t * (1 - t)) * eps
            v = self.velocity(x_tau, tau, h, z_t, a, create_graph=cg)
            x1_hat = x_tau + (1 - t) * v
            err = (x1_hat - z1).float().pow(2)
        per_pair = err.mean(-1)
        return per_pair.mean(), per_pair.detach(), tau.detach()

    # ------------------------------------------------------------------ sampling
    def n_noise(self, S):
        """Number of noise vectors per sample: P1 -> 1 (x0), P2 -> S-1 (EM steps)."""
        return 1 if self.variant == "P1" else max(S - 1, 0)

    def sample_next(self, h, z_t, a, S=None, noise=None):
        """One-step sample of z_{t+1}. h, z_t, a: (N, D). noise: (N, n_noise, D) or None (mean mode).
        Euler steps x <- x - dt * grad_x U; the returned sample is detached."""
        S = self.plan_S if S is None else S
        dt = 1.0 / S
        sigma = self.sigma.to(z_t.device, z_t.dtype)
        n = z_t.size(0)
        if self.variant == "P1":
            x = z_t if noise is None else z_t + sigma * noise[:, 0]
            for k in range(S):
                tau = torch.full((n,), k * dt, device=z_t.device)
                x = x + dt * self.velocity(x, tau, h, z_t, a)
        else:
            x = z_t
            for k in range(S):
                tau = torch.full((n,), k * dt, device=z_t.device)
                x = x + dt * self.velocity(x, tau, h, z_t, a)
                if noise is not None and k < S - 1:
                    x = x + sigma * (dt ** 0.5) * noise[:, k]
        return x.detach()

    def predict(self, emb, act_emb):
        """Deterministic-path convenience (mean mode, plan_S steps): (B, T, D) -> (B, T, D)."""
        h = self.trunk(emb, act_emb)
        b = emb.size(0)
        flat = lambda x: rearrange(x, "b t d -> (b t) d")
        z = self.sample_next(flat(h), flat(emb), flat(act_emb), noise=None)
        return rearrange(z, "(b t) d -> b t d", b=b)

    # ------------------------------------------------------------------ planning
    def _generator(self, device):
        if self._plan_gen is None or self._plan_gen.device != torch.device(device):
            self._plan_gen = torch.Generator(device=device).manual_seed(self.plan_seed)
        return self._plan_gen

    def reset_plan_rng(self):
        self._plan_gen = None

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Same structure as JEPA.rollout, with K particles per candidate and a sampled
        one-step predictor. Noise uses common random numbers: within one call (= one CEM
        iteration) every candidate shares the same eps for particle k.
        pixels: (B, S, T, C, H, W); action_sequence: (B, S, T, action_dim).
        """
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        K = self.plan_K
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")
        # K particles per candidate: (B*S) -> (B*S*K), particle index fastest
        emb = emb.repeat_interleave(K, dim=0)
        act = act.repeat_interleave(K, dim=0)
        act_future = act_future.repeat_interleave(K, dim=0)

        D = emb.size(-1)
        n_noise = self.n_noise(self.plan_S)
        sample_mode = self.plan_mode == "sample" and n_noise > 0 and bool((self.sigma > 0).any())
        if sample_mode:
            g = self._generator(emb.device)
            # (B, 1, K, steps, n_noise, D) shared across the S candidates
            eps = torch.randn((B, 1, K, n_steps + 1, n_noise, D), generator=g, device=emb.device, dtype=emb.dtype)
            eps = eps.expand(B, S, K, n_steps + 1, n_noise, D).reshape(B * S * K, n_steps + 1, n_noise, D)

        HS = history_size
        for t in range(n_steps + 1):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            h = self.trunk(emb_trunc, act_trunc)[:, -1]
            noise = eps[:, t] if sample_mode else None
            z_next = self.sample_next(h, emb_trunc[:, -1], act_trunc[:, -1], noise=noise)
            emb = torch.cat([emb, z_next.unsqueeze(1)], dim=1)
            if t < n_steps:
                act = torch.cat([act, act_future[:, t : t + 1, :]], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s k) ... -> b s k ...", b=B, s=S, k=K)
        return info

    def criterion(self, info_dict: dict):
        """Mean over particles of ||z_hat_H - z_g||^2 (last step only). Returns (B, S)."""
        pred_emb = info_dict["predicted_emb"]  # (B, S, K, T+1, D)
        goal = info_dict["goal_emb"][:, -1, :].detach()  # (B, D)
        last = pred_emb[:, :, :, -1, :]
        cost = (last - goal[:, None, None, :]).pow(2).sum(-1)  # (B, S, K)
        return cost.mean(-1)
