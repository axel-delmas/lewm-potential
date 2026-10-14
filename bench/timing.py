"""CPU micro-benchmark: potential (v = -grad U) vs bridge (v = head) at the real model sizes.

Builds the repo's model from config/train (ViT-tiny encoder, 6-layer trunk, projector, action
embedder; action input 10 as for PushT) with a W=256 / 4-block PotentialHead, and the bridge
FlowJEPA with a W=256 VelocityHead sharing the same trunk weights (bridge code loaded read-only
from ../le-wm). Measures, in fp32 on CPU (no GPU assumed):
  - parameter counts (baseline JEPA, bridge, potential);
  - one full training step (encode + trunk + flow loss + SIGReg + backward + AdamW) at batch --B;
  - one head-only training step at the real pair count (128 x 3 pairs);
  - one CEM planning call (get_cost: 300 candidates, horizon 5, history 1) under inference_mode,
    for several (variant, sigma, S, K).
Usage: python bench/timing.py [--B 8] [--reps 5] [--out results.json]
"""

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from potential import common as C  # noqa: E402
from tiny import bridge_available, bridge_classes  # noqa: E402

D, W, NB = 192, 256, 4


def build(variant="P1", sigma=0.0):
    cfg = C.load_train_cfg()
    with C.open_dict(cfg):
        cfg.model.action_encoder.input_dim = 10  # frameskip 5 x action dim 2 (PushT)
    torch.manual_seed(0)
    baseline = C.instantiate_model(C.model_config(cfg, {"kind": "baseline"}))
    job = dict(kind="potential", variant=variant, sigma=sigma, head_W=W, head_blocks=NB)
    torch.manual_seed(0)
    pot = C.instantiate_model(C.model_config(cfg, job, sigma_std=[1.0] * D))
    return cfg, baseline, pot


def bridge_from(pot, variant, sigma):
    FlowJEPA, VelocityHead = bridge_classes()
    torch.manual_seed(0)
    head = VelocityHead(d=D, W=W, n_blocks=NB, a_dim=D)
    return FlowJEPA(copy.deepcopy(pot.encoder), copy.deepcopy(pot.predictor), copy.deepcopy(pot.action_encoder),
                    copy.deepcopy(pot.projector), head=head, variant=variant, sigma_mult=sigma,
                    sigma_std=[1.0] * D, embed_dim=D)


def timeit(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def train_step_fn(model, cfg, B):
    sigreg = C.make_sigreg(cfg)
    shim = C.Shim(model, sigreg)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    g = torch.Generator().manual_seed(0)
    batch = {"pixels": torch.randn(B, 4, 3, 224, 224, generator=g), "action": torch.randn(B, 4, 10, generator=g)}
    model.train()

    def step():
        out = C.potential_forward(shim, {k: v.clone() for k, v in batch.items()}, "fit", cfg)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    return step


def head_step_fn(model, n_pairs=128 * 3):
    g = torch.Generator().manual_seed(0)
    r = lambda: torch.randn(n_pairs, D, generator=g, requires_grad=True)
    z, z1, h, a = r(), r(), r(), r()
    params = list(model.head.parameters())

    def step():
        loss, _, _ = model.flow_loss(z, z1, h, a)
        grads = torch.autograd.grad(loss, params + [z, z1, h, a], allow_unused=True)
        return grads

    return step


def plan_fn(model, S_steps, K, n_cand=300, horizon=5):
    g = torch.Generator().manual_seed(0)
    pix = torch.randn(1, 1, 3, 224, 224, generator=g)
    model.eval().requires_grad_(False)
    model.plan_S, model.plan_K, model.plan_mode = S_steps, K, "sample"

    def call():
        info = {"pixels": pix.unsqueeze(1).expand(1, n_cand, 1, 3, 224, 224),
                "goal": pix.unsqueeze(1).expand(1, n_cand, 1, 3, 224, 224),
                "action": torch.zeros(1, n_cand, 1, 10)}
        acts = torch.randn(1, n_cand, horizon, 10, generator=g)
        model.reset_plan_rng()
        with torch.inference_mode():
            return model.get_cost(info, acts)

    return call


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=8, help="train batch (repo: 128; smaller on CPU)")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    assert bridge_available(), "needs the bridge checkout at ../le-wm (BRIDGE_REPO)"
    torch.set_grad_enabled(True)
    res = {"device": "cpu", "torch": torch.__version__, "threads": torch.get_num_threads(), "B": args.B}

    cfg, baseline, pot = build("P1", 0.0)
    flow = bridge_from(pot, "A1", 0.0)
    res["params"] = {"baseline": C.param_counts(baseline), "potential": C.param_counts(pot),
                     "bridge": {**C.param_counts(flow), "velocity_head": sum(q.numel() for q in flow.head.parameters())}}
    print(json.dumps(res["params"]), flush=True)

    res["train_step_s"] = {name: timeit(train_step_fn(m, cfg, args.B), args.reps)
                           for name, m in (("bridge_A1", flow), ("potential_P1", pot))}
    print("train", res["train_step_s"], flush=True)
    res["head_step_s_384pairs"] = {name: timeit(head_step_fn(m), args.reps * 4)
                                   for name, m in (("bridge_A1", flow), ("potential_P1", pot))}
    print("head", res["head_step_s_384pairs"], flush=True)

    res["plan_call_s"] = {}
    for pv, bv, sg, S_steps, K in (("P1", "A1", 0.0, 1, 1), ("P1", "A1", 1.0, 4, 1), ("P2", "A2", 1.0, 4, 1),
                                   ("P2", "A2", 1.0, 4, 4)):
        _, _, pm = build(pv, sg)
        bm = bridge_from(pm, bv, sg)
        tag = f"{pv}/{bv} sigma={sg:g} S={S_steps} K={K}"
        res["plan_call_s"][tag] = {"bridge": timeit(plan_fn(bm, S_steps, K), max(args.reps // 2, 2), warm=1),
                                   "potential": timeit(plan_fn(pm, S_steps, K), max(args.reps // 2, 2), warm=1)}
        print("plan", tag, res["plan_call_s"][tag], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
