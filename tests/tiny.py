"""Tiny CPU models for tests and micro-benchmarks (no stable_worldmodel / dataset needed).

The predictor trunk, action embedder and projector are the repo's own module.py classes; only
the ViT encoder is replaced by a linear stand-in. The bridge model (FlowJEPA + VelocityHead) is
loaded read-only from the sibling bridge checkout (BRIDGE_REPO, default ../le-wm) with bytecode
writing disabled, so nothing is written into that folder.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module import ARPredictor, Embedder, MLP  # noqa: E402
from potential.potential_head import PotentialHead  # noqa: E402
from potential.potential_jepa import PotentialJEPA  # noqa: E402

BRIDGE_REPO = Path(os.environ.get("BRIDGE_REPO", REPO_ROOT.parent / "le-wm"))


class TinyEncoder(nn.Module):
    """Stand-in for the HF ViT: returns an object with last_hidden_state (N, 1, W)."""

    def __init__(self, in_dim, W):
        super().__init__()
        self.lin = nn.Linear(in_dim, W)

    def forward(self, pixels, interpolate_pos_encoding=True):
        return SimpleNamespace(last_hidden_state=self.lin(pixels.flatten(1))[:, None])


def trunk_parts(D=16, A=10, img=(3, 8, 8), depth=2, heads=2, dim_head=8, mlp_dim=32, history=3):
    enc = TinyEncoder(img[0] * img[1] * img[2], D)
    pred = ARPredictor(num_frames=history, input_dim=D, hidden_dim=D, output_dim=D, depth=depth, heads=heads,
                       mlp_dim=mlp_dim, dim_head=dim_head, dropout=0.0, emb_dropout=0.0)
    act = Embedder(input_dim=A, emb_dim=D)
    proj = MLP(input_dim=D, output_dim=D, hidden_dim=4 * D, norm_fn=nn.BatchNorm1d)
    return enc, pred, act, proj


def potential_model(D=16, W=32, n_blocks=2, variant="P1", sigma_mult=0.0, **kw):
    head = PotentialHead(d=D, W=W, n_blocks=n_blocks, a_dim=D)
    if variant == "P3":  # bowl centre: same module type as the baseline's pred_proj
        kw.setdefault("pred_proj", MLP(input_dim=D, output_dim=D, hidden_dim=4 * D, norm_fn=nn.BatchNorm1d))
    return PotentialJEPA(*trunk_parts(D), head=head, variant=variant, sigma_mult=sigma_mult, embed_dim=D, **kw)


def _load(name, path):
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = old


def bridge_available():
    return (BRIDGE_REPO / "fm" / "flow_jepa.py").is_file()


def bridge_classes():
    """(FlowJEPA, VelocityHead) from the bridge checkout. Its flow_jepa imports `jepa.JEPA`, which
    resolves to this repo's jepa.py (byte-identical to the bridge's)."""
    fh = _load("bridge_flow_head", BRIDGE_REPO / "fm" / "flow_head.py")
    fj = _load("bridge_flow_jepa", BRIDGE_REPO / "fm" / "flow_jepa.py")
    return fj.FlowJEPA, fh.VelocityHead


def bridge_model(D=16, W=32, n_blocks=2, variant="A1", sigma_mult=0.0, **kw):
    FlowJEPA, VelocityHead = bridge_classes()
    head = VelocityHead(d=D, W=W, n_blocks=n_blocks, a_dim=D)
    return FlowJEPA(*trunk_parts(D), head=head, variant=variant, sigma_mult=sigma_mult, embed_dim=D, **kw)


def toy_batch(B=4, T=4, A=10, img=(3, 8, 8), seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"pixels": torch.randn(B, T, *img, generator=g), "action": torch.randn(B, T, A, generator=g)}


def toy_plan_inputs(B=2, S=5, H=1, T=5, A=10, img=(3, 8, 8), seed=0):
    """(info, action_sequence) shaped like the CEM solver's get_cost inputs."""
    g = torch.Generator().manual_seed(seed)
    info = {"pixels": torch.randn(B, S, H, *img, generator=g), "goal_emb": None}
    return info, torch.randn(B, S, T, A, generator=g)


def randomise_output(head, scale=0.5, seed=0):
    """Non-zero output (and modulation) weights so U and grad U are non-trivial."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in head.named_parameters():
            if name.startswith("out.") or ".mod." in name:
                p.copy_(scale * torch.randn(p.shape, generator=g) / max(p.shape[-1], 1) ** 0.5)
    return head
