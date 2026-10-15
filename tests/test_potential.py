"""Unit tests for the potential-field predictor (CPU, tiny dims). Run: python -m pytest tests -q"""

import pytest
import torch

from tiny import (bridge_available, bridge_model, potential_model, randomise_output, toy_batch,
                  toy_plan_inputs)

from module import SIGReg

D = 16


def _pairs(n=6, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    r = lambda: torch.randn(n, D, generator=g, dtype=dtype)
    return r(), torch.rand(n, generator=g, dtype=dtype), r(), r(), r()  # x, tau, h, z_t, a


def _fd_velocity(model, x, tau, h, z, a, eps=1e-6):
    """-dU/dx by central finite differences, one coordinate at a time."""
    v = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(x.size(1)):
            e = torch.zeros_like(x)
            e[:, i] = eps
            v[:, i] = -(model.potential(x + e, tau, h, z, a) - model.potential(x - e, tau, h, z, a)) / (2 * eps)
    return v


# (1) v == -grad U, checked against finite differences (float64), both code paths
@pytest.mark.parametrize("create_graph", [False, True])
def test_velocity_matches_finite_differences(create_graph):
    torch.manual_seed(0)
    model = potential_model(D=D).double().eval()
    randomise_output(model.head)
    x, tau, h, z, a = _pairs(dtype=torch.float64)
    v = model.velocity(x, tau, h, z, a, create_graph=create_graph)
    v_fd = _fd_velocity(model, x, tau, h, z, a)
    assert v.abs().max() > 1e-3, "test is vacuous: field is ~0"
    torch.testing.assert_close(v, v_fd, rtol=1e-5, atol=1e-7)


# (2) zero-initialised output layer -> U == 0 and v == 0 at init
def test_zero_field_at_init():
    torch.manual_seed(0)
    model = potential_model(D=D).eval()
    x, tau, h, z, a = _pairs()
    assert torch.equal(model.potential(x, tau, h, z, a), torch.zeros(x.size(0)))
    assert torch.equal(model.velocity(x, tau, h, z, a), torch.zeros_like(x))
    assert torch.equal(model.sample_next(h, z, a, S=4), z)


# (3) sampling works under no_grad and inference_mode, returns detached samples and leaves
#     no gradients on the parameters or on the conditioning inputs
@pytest.mark.parametrize("ctx", ["no_grad", "inference_mode", "grad"])
@pytest.mark.parametrize("variant", ["P1", "P2", "P3"])
def test_sample_next_no_grad(ctx, variant):
    torch.manual_seed(0)
    model = potential_model(D=D, variant=variant, sigma_mult=1.0).eval()
    randomise_output(model.head)
    model.requires_grad_(ctx == "grad")  # eval_fm sets requires_grad_(False)
    x, tau, h, z, a = _pairs()
    for t in (h, z, a):
        t.requires_grad_(ctx == "grad")
    noise = torch.randn(z.size(0), model.n_noise(4), D)
    cm = {"no_grad": torch.no_grad, "inference_mode": torch.inference_mode, "grad": torch.enable_grad}[ctx]
    with cm():
        out = model.sample_next(h, z, a, S=4, noise=noise)
    assert out.shape == z.shape and torch.isfinite(out).all()
    assert out.grad_fn is None and not out.requires_grad
    assert not torch.allclose(out, z), "field did not move the sample"
    assert all(p.grad is None for p in model.parameters())
    assert all(t.grad is None for t in (h, z, a))
    # same numbers in every mode
    with torch.no_grad():
        ref = model.sample_next(h.detach(), z.detach(), a.detach(), S=4, noise=noise)
    torch.testing.assert_close(out, ref)


def test_rollout_under_inference_mode():
    torch.manual_seed(0)
    model = potential_model(D=D, variant="P2", sigma_mult=1.0, plan_S=3, plan_K=2).eval()
    randomise_output(model.head)
    info, acts = toy_plan_inputs()
    with torch.inference_mode():
        out = model.rollout(dict(info), acts)["predicted_emb"]
    assert torch.isfinite(out).all()


# (4) rollout / criterion shapes match FlowJEPA's
def _plan(model, S_steps=2, K=3):
    model.plan_S, model.plan_K = S_steps, K
    model.reset_plan_rng()
    info, acts = toy_plan_inputs()
    B, S, T = acts.shape[:3]
    with torch.no_grad():
        info = model.rollout(info, acts)
        info["goal_emb"] = torch.randn(B, 1, D)
        cost = model.criterion(info)
    return info["predicted_emb"].shape, cost.shape, (B, S, T)


@pytest.mark.parametrize("variant", ["P1", "P2", "P3"])
def test_rollout_criterion_shapes(variant):
    torch.manual_seed(0)
    model = potential_model(D=D, variant=variant, sigma_mult=1.0).eval()
    pe, c, (B, S, T) = _plan(model)
    assert pe == (B, S, 3, T + 1, D)  # K=3 particles; H history + (T - H) + 1 predicted
    assert c == (B, S)


@pytest.mark.skipif(not bridge_available(), reason="bridge checkout (../le-wm) not found")
@pytest.mark.parametrize("pv,bv", [("P1", "A1"), ("P2", "A2"), ("P3", "A2")])
def test_shapes_match_bridge(pv, bv):
    torch.manual_seed(0)
    pot = potential_model(D=D, variant=pv, sigma_mult=1.0).eval()
    torch.manual_seed(0)
    flow = bridge_model(D=D, variant=bv, sigma_mult=1.0).eval()
    assert _plan(pot) == _plan(flow)
    assert pot.n_noise(4) == flow.n_noise(4)
    x, tau, h, z, a = _pairs()
    lp, pp, tp = pot.flow_loss(z, x, h, a)
    lf, pf, tf = flow.flow_loss(z, x, h, a)
    assert (lp.shape, pp.shape, tp.shape) == (lf.shape, pf.shape, tf.shape)


# (5) one training step on a toy batch reduces the loss (full LeWM objective, fixed randomness)
def _loss(model, sigreg, batch, seed, lam=0.09, history=3):
    torch.manual_seed(seed)
    out = model.encode({k: v.clone() for k, v in batch.items()})
    emb, act = out["emb"], out["act_emb"]
    h = model.trunk(emb[:, :history], act[:, :history])
    flat = lambda t: t.reshape(-1, t.size(-1))
    fl, _, _ = model.flow_loss(flat(emb[:, :history]), flat(emb[:, 1:]), flat(h), flat(act[:, :history]))
    return fl + lam * sigreg(emb.transpose(0, 1))


@pytest.mark.parametrize("variant", ["P1", "P2", "P3"])
def test_train_step_reduces_loss(variant):
    torch.manual_seed(0)
    model = potential_model(D=D, variant=variant, sigma_mult=0.5)
    sigreg = SIGReg(knots=17, num_proj=64)
    batch = toy_batch(B=8)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    losses = []
    for step in range(4):
        loss = _loss(model, sigreg, batch, seed=123)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        if step == 0:  # double backward reaches the potential head's non-output layers only after step 1
            assert model.head.out.weight.grad.abs().sum() > 0
            assert model.encoder.lin.weight.grad.abs().sum() > 0
        else:
            assert model.head.x_in.weight.grad.abs().sum() > 0
        opt.step()
    assert losses[1] < losses[0] and losses[-1] < losses[0], losses


def test_training_gradient_reaches_conditioning():
    """Training path: like the bridge, the loss reaches h_t, z_t and a_t through v."""
    torch.manual_seed(0)
    model = potential_model(D=D)
    randomise_output(model.head)
    x, tau, h, z, a = _pairs()
    for t in (h, z, a):
        t.requires_grad_(True)
    loss, _, _ = model.flow_loss(z, x, h, a)
    loss.backward()
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in (h, z, a))


def test_loss_under_no_grad_matches():
    """train_fm.diagnostics evaluates the loss under @torch.no_grad(): same value, no graph."""
    torch.manual_seed(0)
    model = potential_model(D=D, sigma_mult=1.0).eval()
    randomise_output(model.head)
    x, tau, h, z, a = _pairs()
    torch.manual_seed(5)
    l1, p1, _ = model.flow_loss(z, x, h, a)
    torch.manual_seed(5)
    with torch.no_grad():
        l2, p2, _ = model.flow_loss(z, x, h, a)
    assert not l2.requires_grad
    torch.testing.assert_close(l1.detach(), l2)
    torch.testing.assert_close(p1, p2)


def test_head_fp32_matches_default_in_fp32():
    torch.manual_seed(0)
    model = potential_model(D=D).eval()
    randomise_output(model.head)
    x, tau, h, z, a = _pairs()
    v0 = model.velocity(x, tau, h, z, a)
    model.head_fp32 = True
    torch.testing.assert_close(model.velocity(x, tau, h, z, a), v0)
    # under bf16 autocast, head_fp32 must reproduce the fp32 field exactly (incl. the grad call)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v1 = model.velocity(x, tau, h, z, a)
        v2 = model.velocity(x, tau, h, z, a, create_graph=True)
    assert v1.dtype == torch.float32
    torch.testing.assert_close(v1, v0, rtol=0, atol=0)
    torch.testing.assert_close(v2.detach(), v0, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["P2", "P3"])
def test_cem_solver_end_to_end(variant):
    """The PushT eval's CEMSolver.solve runs under @torch.inference_mode(): planning must still work."""
    swm_solver = pytest.importorskip("stable_worldmodel.solver")
    gym = pytest.importorskip("gymnasium")
    from types import SimpleNamespace

    torch.manual_seed(0)
    model = potential_model(D=D, variant=variant, sigma_mult=1.0, plan_S=2, plan_K=2).eval()
    randomise_output(model.head)
    model.requires_grad_(False)
    n_envs = 2
    solver = swm_solver.CEMSolver(model=model, num_samples=6, n_steps=2, topk=2, device="cpu", seed=0)
    solver.configure(action_space=gym.spaces.Box(-1, 1, shape=(n_envs, 2)), n_envs=n_envs,
                     config=SimpleNamespace(horizon=3, action_block=5))
    info = {"pixels": torch.randn(n_envs, 1, 3, 8, 8), "goal": torch.randn(n_envs, 1, 3, 8, 8),
            "action": torch.randn(n_envs, 1, 10)}
    out = solver.solve(info)
    acts = out["actions"]
    assert acts.shape[0] == n_envs and torch.isfinite(torch.as_tensor(acts)).all()


# (6) P3: structured potential U = ||x - m||^2 / (2(1-tau)) + U_net, m = pred_proj(h)
def test_p3_at_init_is_the_baseline_predictor():
    """With U_net == 0 (its init) the P3 loss is the baseline's MSE ||pred_proj(h) - z1||^2 for every tau,
    and sampling lands exactly on m whatever the noise."""
    torch.manual_seed(0)
    model = potential_model(D=D, variant="P3", sigma_mult=1.0).eval()
    x, tau, h, z, a = _pairs(n=32)
    loss, per_pair, _ = model.flow_loss(z, x, h, a)
    m = model.pred_proj(h)
    torch.testing.assert_close(per_pair, (m - x).pow(2).mean(-1))
    noise = torch.randn(z.size(0), model.n_noise(4), D)
    with torch.inference_mode():
        out = model.sample_next(h, z, a, S=4, noise=noise)
    torch.testing.assert_close(out, m.detach(), rtol=1e-5, atol=1e-5)


def test_p3_field_is_conservative():
    """The full P3 field (bowl + net) has a symmetric Jacobian in x (curl-free)."""
    torch.manual_seed(0)
    model = potential_model(D=D, variant="P3", sigma_mult=1.0).double().eval()
    randomise_output(model.head)
    x, tau, h, z, a = _pairs(n=1, dtype=torch.float64)
    m = model.pred_proj(h).detach()

    def f(xx):
        return model.velocity(xx, tau, h, z, a, create_graph=True) + (m - xx) / (1 - tau)[:, None]

    J = torch.autograd.functional.jacobian(f, x)[0, :, 0, :]
    assert J.abs().max() > 1e-3
    torch.testing.assert_close(J, J.T, rtol=1e-8, atol=1e-10)
    ref = f(x.clone().requires_grad_(True)).detach()
    with torch.no_grad():
        torch.testing.assert_close(model.field(x, tau, h, z, a, m), ref)


def test_p3_requires_bowl_centre():
    with pytest.raises(AssertionError):
        potential_model(D=D, variant="P3", pred_proj=None)
