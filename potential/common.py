"""Shared helpers: config composition, data pipeline, model building, forward passes.

The data pipeline and the baseline loss are taken from the repo unchanged:
- datasets/transforms/split are built with the same calls as train.py::run
- the baseline loss IS train.py::lejepa_forward (imported, called through a tiny shim)
"""

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame for PushT rendering

import stable_pretraining as spt  # noqa: E402  (registers the ${eval:...} resolver)
import stable_worldmodel as swm  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from module import SIGReg  # noqa: E402
from utils import get_column_normalizer, get_img_preprocessor  # noqa: E402

TRAIN_CFG_DIR = REPO_ROOT / "config" / "train"
EVAL_CFG_DIR = REPO_ROOT / "config" / "eval"


# ----------------------------------------------------------------------------- utils
def now():
    return time.time()


def fmt_hms(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "?"
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def torch_save_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def git_commit():
    import subprocess

    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------- configs
def compose_cfg(config_dir: Path, config_name: str, overrides=()):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def load_train_cfg(overrides=()):
    """The repo's training config (config/train/lewm.yaml + data=pusht), as train.py sees it."""
    return compose_cfg(TRAIN_CFG_DIR, "lewm", ["data=pusht", *overrides])


def load_eval_cfg(overrides=()):
    return compose_cfg(EVAL_CFG_DIR, "pusht", list(overrides))


# ----------------------------------------------------------------------------- data
class ResumableEpochSampler(torch.utils.data.Sampler):
    """Deterministic per-epoch permutation that can start mid-epoch (for resume)."""

    def __init__(self, n, seed):
        self.n = n
        self.seed = int(seed)
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch, start_index=0):
        self.epoch = int(epoch)
        self.start = int(start_index)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        return iter(perm[self.start :])

    def __len__(self):
        return self.n - self.start


def build_datasets(cfg):
    """Same calls as train.py::run (dataset, transforms, normalizers, split). Mutates cfg
    (action_encoder.input_dim) exactly like train.py does."""
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    return dataset, train_set, val_set


def build_loaders(cfg, train_set, val_set, shuffle_seed, num_workers=None):
    loader_kw = OmegaConf.to_container(cfg.loader, resolve=True)
    if num_workers is not None:
        loader_kw["num_workers"] = num_workers
    if loader_kw["num_workers"] == 0:
        loader_kw.pop("persistent_workers", None)
        loader_kw.pop("prefetch_factor", None)
    sampler = ResumableEpochSampler(len(train_set), shuffle_seed)
    train = torch.utils.data.DataLoader(train_set, sampler=sampler, drop_last=True, **loader_kw)
    val = torch.utils.data.DataLoader(val_set, shuffle=False, drop_last=False, **loader_kw)
    return train, val, sampler


# ----------------------------------------------------------------------------- models
def head_cfg(job, embed_dim):
    return {
        "_target_": "potential.potential_head.PotentialHead",
        "d": embed_dim,
        "W": int(job.get("head_W", 256)),
        "n_blocks": int(job.get("head_blocks", 4)),
        "mlp_ratio": 4,
        "a_dim": embed_dim,
        "p_drop": float(job.get("head_dropout", 0.0)),
    }


def model_config(cfg, job, sigma_std=None):
    """Resolved hydra config (dict with _target_) for the job's model."""
    mc = OmegaConf.to_container(cfg.model, resolve=True)
    if job["kind"] == "baseline":
        return mc
    mc.pop("pred_proj")
    mc["_target_"] = "potential.potential_jepa.PotentialJEPA"
    mc["head"] = head_cfg(job, int(cfg.embed_dim))
    mc["variant"] = job["variant"]
    mc["sigma_mult"] = float(job["sigma"])
    mc["sigma_std"] = None if float(job["sigma"]) == 0.0 else list(map(float, sigma_std))
    mc["embed_dim"] = int(cfg.embed_dim)
    mc["detach_target"] = bool(job.get("detach_target", False))
    mc["head_fp32"] = bool(job.get("head_fp32", False))
    return mc


def instantiate_model(mc: dict):
    import hydra

    return hydra.utils.instantiate(OmegaConf.create(mc), _convert_="object")


def load_model_from_ckpt(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = instantiate_model(ck["model_cfg"])
    model.load_state_dict(ck["model"])
    return model, ck


def param_counts(model):
    def n(m):
        return sum(p.numel() for p in m.parameters()) if m is not None else 0

    out = {
        "encoder": n(model.encoder),
        "encoder_projector": n(model.projector),
        "predictor_trunk": n(model.predictor),
        "pred_proj": n(model.pred_proj),
        "action_embedder": n(model.action_encoder),
    }
    if hasattr(model, "head"):
        out["potential_head"] = n(model.head)
    out["total"] = n(model)
    return out


# ----------------------------------------------------------------------------- forward passes
class Shim:
    """The attributes of spt.Module that train.py::lejepa_forward touches."""

    def __init__(self, model, sigreg, ema=None):
        self.model = model
        self.sigreg = sigreg
        self.ema = ema

    def log_dict(self, *args, **kwargs):
        pass


def baseline_forward(shim, batch, stage, cfg):
    from train import lejepa_forward  # the repo's loss, unchanged

    return lejepa_forward(shim, batch, stage, cfg)


def potential_forward(shim, batch, stage, cfg):
    """Mirror of lejepa_forward with the MSE replaced by the flow loss of v = -grad U (mean over dims)."""
    model = shim.model
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    if shim.ema is not None:
        tgt_emb = shim.ema.target_emb(batch)[:, n_preds:]
    elif model.detach_target:
        tgt_emb = tgt_emb.detach()

    h = model.trunk(ctx_emb, ctx_act)
    flat = lambda x: x.reshape(-1, x.size(-1))
    flow_loss, per_pair, tau = model.flow_loss(flat(ctx_emb), flat(tgt_emb), flat(h), flat(ctx_act))

    output["pred_loss"] = flow_loss
    output["sigreg_loss"] = shim.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    output["per_pair"] = per_pair
    output["tau"] = tau
    output["h"] = h
    return output


class EMATarget:
    """Optional (--ema, off by default): EMA copy of encoder+projector for the targets."""

    def __init__(self, model, decay=0.996):
        import copy

        self.decay = decay
        self.encoder = copy.deepcopy(model.encoder).requires_grad_(False)
        self.projector = copy.deepcopy(model.projector).requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for src, dst in ((model.encoder, self.encoder), (model.projector, self.projector)):
            for ps, pd in zip(src.parameters(), dst.parameters()):
                pd.mul_(self.decay).add_(ps.detach(), alpha=1 - self.decay)
            for bs, bd in zip(src.buffers(), dst.buffers()):
                bd.copy_(bs)

    @torch.no_grad()
    def target_emb(self, batch):
        from einops import rearrange

        pixels = batch["pixels"].float()
        b = pixels.size(0)
        out = self.encoder(rearrange(pixels, "b t ... -> (b t) ..."), interpolate_pos_encoding=True)
        emb = self.projector(out.last_hidden_state[:, 0])
        return rearrange(emb, "(b t) d -> b t d", b=b)

    def state_dict(self):
        return {"encoder": self.encoder.state_dict(), "projector": self.projector.state_dict()}

    def load_state_dict(self, sd):
        self.encoder.load_state_dict(sd["encoder"])
        self.projector.load_state_dict(sd["projector"])

    def to(self, device):
        self.encoder.to(device)
        self.projector.to(device)
        return self


def make_sigreg(cfg):
    return SIGReg(**cfg.loss.sigreg.kwargs)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def autocast_ctx(cfg, device):
    """Same numerics as the repo: Lightning precision 'bf16' == 'bf16-mixed' autocast."""
    prec = str(cfg.trainer.precision)
    if device.type == "cuda" and prec in ("bf16", "bf16-mixed"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and prec in ("16", "16-mixed"):
        raise NotImplementedError("fp16-mixed would need a GradScaler; the repo uses bf16")
    import contextlib

    return contextlib.nullcontext()


def stablewm_home():
    return Path(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel")))


# ----------------------------------------------------------------------------- bridge results (fm-jepa)
def materialise_out_dir(src, cache_root):
    """A bridge-repo results source as a local out/ dir (the directory holding manifest.json).
    `src` is an out/ directory (used in place, read-only) or a bundle .tar.gz, whose non-checkpoint
    members are extracted once under cache_root."""
    import tarfile

    src = Path(src).expanduser().resolve()
    if src.is_dir():
        root = src
    elif src.is_file():
        root = Path(cache_root) / src.name.removesuffix(".tar.gz").removesuffix(".tgz")
        if not (root / ".extracted").exists():
            with tarfile.open(src, "r:*") as tar:
                members = [m for m in tar.getmembers() if m.isfile() and not m.name.endswith((".pt", ".ckpt"))]
                tar.extractall(root, members=members, filter="data")
            (root / ".extracted").touch()
    else:
        raise FileNotFoundError(f"bridge results source not found: {src}")
    hits = sorted((p.parent for p in root.rglob("manifest.json") if "jobs" not in p.relative_to(root).parts),
                  key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"no manifest.json (bridge out/ dir) under {src}")
    return hits[0]
