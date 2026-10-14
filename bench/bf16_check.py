"""bf16 sanity check for v = -grad_x U vs the bridge's v = head(x) (CPU autocast as a proxy for CUDA).

Caveat: on CPU the autograd engine runs backward ops on the calling thread, so the inner grad_x U
(computed inside the autocast region) also sees autocast; on CUDA it runs on the device thread.
The CPU numbers are therefore a proxy, not the exact CUDA behaviour; preflight logs the same
comparison on the GPU.

For both heads at the real sizes (D=192, W=256, 4 blocks), with non-zero output/modulation weights:
  - field error:    ||v_bf16 - v_fp32|| / ||v_fp32||           (sampling / eval path)
  - loss error:     |L_bf16 - L_fp32| / L_fp32                  (training loss value)
  - grad error:     ||dL/dtheta_bf16 - dL/dtheta_fp32|| / ||.||  (training gradient; double backward for U)
  - head_fp32 rows: the potential head run outside autocast inside an otherwise-bf16 step.
Usage: python bench/bf16_check.py [--n 1024] [--seeds 3]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from tiny import bridge_available, bridge_classes, potential_model, randomise_output  # noqa: E402

D, W = 192, 256


def fields(kind, model, x, tau, h, z, a, head_fp32=False):
    if kind == "bridge":
        return model(x, tau, h, z, a)
    model.head_fp32 = head_fp32
    return model.velocity(x, tau, h, z, a, create_graph=torch.is_grad_enabled())


def loss_and_grad(kind, head, x0, z1, tau, h, z, a, head_fp32=False, bf16=False):
    """Forward (incl. the inner grad_x U) under autocast, outer backward outside it, as in train_fm."""
    t = tau[:, None]
    x = ((1 - t) * x0 + t * z1)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        v = fields(kind, head, x, tau, h, z, a, head_fp32)
        loss = (v - (z1 - x0)).float().pow(2).mean()
    params = list((head if kind == "bridge" else head.head).parameters())
    g = torch.autograd.grad(loss, params, allow_unused=True)  # U's output bias never affects grad U
    return loss.detach(), torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                                     for gi, p in zip(g, params)]).float()


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    heads = {"potential": lambda: potential_model(D=D, W=W, n_blocks=4)}
    if bridge_available():
        _, VelocityHead = bridge_classes()
        heads["bridge"] = lambda: VelocityHead(d=D, W=W, n_blocks=4, a_dim=D)
    rows = []
    for kind, make in heads.items():
        for fp32 in ([False, True] if kind == "potential" else [False]):
            acc = {"field": [], "loss": [], "grad": []}
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                head = make()
                randomise_output(head if kind == "bridge" else head.head, scale=1.0, seed=seed)
                g = torch.Generator().manual_seed(100 + seed)
                r = lambda: torch.randn(args.n, D, generator=g)
                x0, z1, h, z, a = r(), r(), r(), r(), r()
                tau = torch.rand(args.n, generator=g)
                with torch.no_grad():
                    v32 = fields(kind, head, z1, tau, h, z, a)
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        v16 = fields(kind, head, z1, tau, h, z, a, fp32)
                L32, G32 = loss_and_grad(kind, head, x0, z1, tau, h, z, a)
                L16, G16 = loss_and_grad(kind, head, x0, z1, tau, h, z, a, fp32, bf16=True)
                acc["field"].append(rel(v16, v32))
                acc["loss"].append(abs(float(L16 - L32)) / float(L32))
                acc["grad"].append(rel(G16, G32))
            row = {"head": kind + ("+head_fp32" if fp32 else ""),
                   **{f"rel_err_{k}": sum(v) / len(v) for k, v in acc.items()}}
            rows.append(row)
            print(json.dumps(row))
    return rows


if __name__ == "__main__":
    main()
