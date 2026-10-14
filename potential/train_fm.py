"""Train one job (baseline LeWM or potential variant) with resumable checkpoints.

Reproduces train.py's training computation without Lightning so that checkpoints carry
optimizer / scheduler / RNG state and can resume mid-epoch:
  - same config (config/train/lewm.yaml + data=pusht), dataset, transforms, split
  - AdamW from spt.optim.create_optimizer, spt LinearWarmupCosineAnnealingLR with the
    defaults spt.Module uses (warmup = 1% of total steps, cosine to 0 at total steps),
    stepped after every optimizer step (spt.Module does manual optimization)
  - bf16 autocast (Lightning precision "bf16" == "bf16-mixed"), grad-norm clip 1.0
  - baseline loss = train.py::lejepa_forward itself

Usage: python -m potential.train_fm --job_dir out/jobs/<job_id> [--deadline <unix ts>]
Exit codes: 0 done, 3 interrupted (deadline / SIGTERM; checkpoint written), 1 error.
"""

import argparse
import csv
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from potential import common as C

EXIT_INTERRUPTED = 3
PROCESS_T0 = time.time()


class Stop(Exception):
    pass


_STOP = {"flag": False, "why": ""}


def _on_signal(signum, frame):
    _STOP["flag"] = True
    _STOP["why"] = f"signal {signum}"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--job_dir", required=True)
    p.add_argument("--deadline", type=float, default=None, help="unix time; checkpoint + exit 3 when reached")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--val_batches", type=int, default=100, help="val batches for per-epoch diagnostics (-1 = all)")
    p.add_argument("--ckpt_minutes", type=float, default=10.0, help="best-effort mid-epoch checkpoint period")
    p.add_argument("--benchmark_steps", type=int, default=0, help="time N train steps (+val) and exit; no ckpts")
    p.add_argument("--preflight", action="store_true", help="one fwd/bwd pass on a real batch and exit")
    p.add_argument("--max_steps_per_epoch", type=int, default=0, help="debug only: truncate epochs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ----------------------------------------------------------------------------- setup
def setup(job, args, device):
    overrides = [
        f"loss.sigreg.weight={job['lambda']}",
        f"trainer.max_epochs={job['epochs']}",
    ]
    # debug/testing only (e.g. a remote dataset or a tiny batch); recorded in the log
    overrides += os.environ.get("POT_DEBUG_OVERRIDES", "").split()
    cfg = C.load_train_cfg(overrides)
    dataset, train_set, val_set = C.build_datasets(cfg)
    train_loader, val_loader, sampler = C.build_loaders(
        cfg, train_set, val_set, shuffle_seed=job["seed"], num_workers=args.num_workers
    )
    sigma_std = None
    if job["kind"] == "potential" and float(job["sigma"]) > 0:
        calib = C.read_json(job["calibration_path"])
        if calib is None:
            raise RuntimeError(f"calibration missing: {job['calibration_path']}")
        sigma_std = calib["delta_std"]
    mc = C.model_config(cfg, job, sigma_std)
    C.seed_everything(job["seed"])  # model init, SIGReg projections, flow noise
    model = C.instantiate_model(mc).to(device)
    sigreg = C.make_sigreg(cfg).to(device)
    ema = C.EMATarget(model).to(device) if job.get("ema") else None
    shim = C.Shim(model, sigreg, ema)

    from stable_pretraining.optim import create_optimizer
    from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR

    opt = create_optimizer(list(model.parameters()), dict(C.OmegaConf.to_container(cfg.optimizer, resolve=True)),
                           named_params=list(model.named_parameters()))
    steps_per_epoch = len(train_loader) if not args.max_steps_per_epoch else min(len(train_loader), args.max_steps_per_epoch)
    total_steps = steps_per_epoch * int(cfg.trainer.max_epochs)
    sched = LinearWarmupCosineAnnealingLR(
        opt, warmup_steps=max(1, int(0.01 * total_steps)), max_steps=total_steps, warmup_start_lr=0.0, eta_min=0.0
    )
    forward = C.baseline_forward if job["kind"] == "baseline" else C.potential_forward
    return dict(cfg=cfg, mc=mc, model=model, sigreg=sigreg, ema=ema, shim=shim, opt=opt, sched=sched,
                train_loader=train_loader, val_loader=val_loader, sampler=sampler, forward=forward,
                steps_per_epoch=steps_per_epoch, total_steps=total_steps, n_train=len(train_set), n_val=len(val_set))


def full_state(S, job, epoch, batch_in_epoch, global_step, extra=None):
    st = {
        "model": S["model"].state_dict(),
        "model_cfg": S["mc"],
        "optimizer": S["opt"].state_dict(),
        "scheduler": S["sched"].state_dict(),
        "scaler": None,  # bf16 autocast: no GradScaler (same as Lightning bf16-mixed)
        "epoch": epoch,  # number of completed epochs
        "batch_in_epoch": batch_in_epoch,  # batches done in the current (epoch+1)
        "global_step": global_step,
        "rng": C.rng_state(),
        "job": job,
        "total_steps": S["total_steps"],
        "steps_per_epoch": S["steps_per_epoch"],
    }
    if S["ema"] is not None:
        st["ema"] = S["ema"].state_dict()
    if extra:
        st.update(extra)
    return st


def model_state(S, job, epoch, global_step):
    return {"model": S["model"].state_dict(), "model_cfg": S["mc"], "epoch": epoch, "global_step": global_step,
            "job": job}


def latest_full_ckpt(ckpt_dir: Path):
    best, best_step = None, -1
    for p in ckpt_dir.glob("full_*.pt"):
        try:
            meta = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if meta.get("global_step", -1) > best_step:
            best, best_step = p, meta["global_step"]
    return best


def restore(S, ck, device):
    S["model"].load_state_dict(ck["model"])
    S["opt"].load_state_dict(ck["optimizer"])
    S["sched"].load_state_dict(ck["scheduler"])
    if S["ema"] is not None and "ema" in ck:
        S["ema"].load_state_dict(ck["ema"])
    C.set_rng_state(ck["rng"])


# ----------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def diagnostics(S, job, device, n_batches):
    """Per-epoch val diagnostics on a fixed val subset. Uses a forked RNG so the training
    random stream is unaffected."""
    cfg, model, shim = S["cfg"], S["model"], S["shim"]
    model.eval()
    devs = [device] if device.type == "cuda" else []
    rows = {"val_loss": [], "val_pred_loss": [], "val_sigreg": [], "straightness": []}
    bins = np.zeros(5)
    bin_n = np.zeros(5)
    onestep = []
    embs = []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(12345)
        for i, batch in enumerate(S["val_loader"]):
            if 0 <= n_batches <= i:
                break
            batch = C.to_device(batch, device)
            with C.autocast_ctx(cfg, device):
                out = S["forward"](shim, batch, "validate", cfg)
            emb = out["emb"].float()
            rows["val_loss"].append(out["loss"].item())
            rows["val_pred_loss"].append(out["pred_loss"].item())
            rows["val_sigreg"].append(out["sigreg_loss"].item())
            v = emb[:, 1:] - emb[:, :-1]
            cos = torch.nn.functional.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1)
            rows["straightness"].append(cos.mean().item())
            if sum(e.shape[0] for e in embs) < 8192:
                embs.append(emb.reshape(-1, emb.size(-1)).cpu())
            if job["kind"] == "potential":
                b = torch.clamp((out["tau"] * 5).long(), max=4).cpu().numpy()
                pp = out["per_pair"].float().cpu().numpy()
                np.add.at(bins, b, pp)
                np.add.at(bin_n, b, 1)
                # one-step sample MSE with the job's planning S (comparable to baseline pred_loss)
                ctx = emb[:, : cfg.history_size]
                tgt = emb[:, cfg.num_preds:]
                act = out["act_emb"][:, : cfg.history_size].float()
                h = out["h"].float()
                flat = lambda x: x.reshape(-1, x.size(-1))
                n_noise = model.n_noise(job["plan_S"])
                noise = torch.randn(flat(ctx).size(0), max(n_noise, 1), ctx.size(-1), device=device) \
                    if (n_noise > 0 and float(job["sigma"]) > 0) else None
                with C.autocast_ctx(cfg, device):
                    z = model.sample_next(flat(h), flat(ctx), flat(act), S=job["plan_S"], noise=noise)
                onestep.append((z.float() - flat(tgt)).pow(2).mean().item())
    model.train()
    Z = torch.cat(embs)[:8192]
    std = Z.std(0)
    Zc = Z - Z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Zc)
    p = s / s.sum()
    eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
    d = {k: float(np.mean(v)) if v else float("nan") for k, v in rows.items()}
    d.update(z_std_mean=float(std.mean()), z_std_min=float(std.min()), z_std_max=float(std.max()),
             eff_rank=eff_rank)
    for k in range(5):
        d[f"val_flow_tau{k}"] = float(bins[k] / bin_n[k]) if bin_n[k] else float("nan")
    d["val_onestep_mse"] = float(np.mean(onestep)) if onestep else d["val_pred_loss"]
    return d


# ----------------------------------------------------------------------------- main loop
def main():
    args = parse()
    job_dir = Path(args.job_dir)
    job = C.read_json(job_dir / "config.json")
    device = torch.device(args.device)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    ckpt_dir = job_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(job_dir / "train.log", "a", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_f.write(line + "\n")
        print(line, flush=True)

    S = setup(job, args, device)
    model, opt, sched, cfg = S["model"], S["opt"], S["sched"], S["cfg"]
    pc = C.param_counts(model)
    log(f"job {job['job_id']} | params {pc} | train {S['n_train']} val {S['n_val']} | "
        f"steps/epoch {S['steps_per_epoch']} total {S['total_steps']}")
    if "potential_head" in pc:
        base_total = pc["total"] - pc["potential_head"] + 792768  # baseline has pred_proj (Linear-BN-GELU-Linear 192/2048)
        log(f"potential head {pc['potential_head']:,} params | trunk+head {pc['predictor_trunk'] + pc['potential_head']:,} "
            f"vs baseline trunk+pred_proj {pc['predictor_trunk'] + 792768:,} | total {pc['total']:,} vs baseline {base_total:,}")
    if os.environ.get("POT_DEBUG_OVERRIDES"):
        log(f"DEBUG overrides active: {os.environ['POT_DEBUG_OVERRIDES']}")
    C.write_json_atomic(job_dir / "params.json", pc)

    if args.preflight:
        return preflight_step(S, job, device, log)
    if args.benchmark_steps:
        return benchmark(S, job, device, args, log, job_dir)

    epochs = int(job["epochs"])
    milestones = set(job["milestones"])
    start_epoch, start_batch, global_step = 0, 0, 0
    ck_path = latest_full_ckpt(ckpt_dir)
    if ck_path is not None:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        if ck["total_steps"] != S["total_steps"]:
            raise RuntimeError(f"checkpoint total_steps {ck['total_steps']} != {S['total_steps']} (epochs changed?)")
        restore(S, ck, device)
        start_epoch, start_batch, global_step = ck["epoch"], ck["batch_in_epoch"], ck["global_step"]
        log(f"resumed from {ck_path.name}: epoch {start_epoch} batch {start_batch} step {global_step}")
    elif job.get("extend_from_ckpt"):
        # extension mode "continue": warm-start from a finished shorter run, schedule rebuilt
        ck = torch.load(job["extend_from_ckpt"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if S["ema"] is not None and "ema" in ck:
            S["ema"].load_state_dict(ck["ema"])
        C.set_rng_state(ck["rng"])
        start_epoch, global_step = ck["epoch"], ck["global_step"]
        sched.base_lrs = [float(cfg.optimizer.lr)] * len(opt.param_groups)
        sched.last_epoch = global_step
        for g, lr in zip(opt.param_groups, sched.get_lr()):
            g["lr"] = lr
        log(f"extended from {job['extend_from_ckpt']} at epoch {start_epoch} (schedule rebuilt for {epochs} epochs)")
    _dedupe_metrics(job_dir / "metrics.csv", start_epoch)

    params = [p for p in model.parameters() if p.requires_grad]
    clip = cfg.trainer.gradient_clip_val
    last_ckpt_t = time.time()
    it_ema, t_last = None, time.time()
    prog_t = 0.0
    t_job0 = time.time()

    def write_progress(epoch, bi, phase="train"):
        rate = it_ema or 0.0
        remaining_steps = S["total_steps"] - global_step
        C.write_json_atomic(job_dir / "progress.json", {
            "phase": phase, "epoch": epoch, "epochs": epochs, "step_in_epoch": bi,
            "steps_per_epoch": S["steps_per_epoch"], "global_step": global_step, "total_steps": S["total_steps"],
            "it_s": rate, "train_eta_s": remaining_steps / rate if rate > 0 else None, "time": time.time(),
            "elapsed_s": time.time() - t_job0,
        })

    def save_mid(epoch, bi):
        C.torch_save_atomic(full_state(S, job, epoch, bi, global_step), ckpt_dir / "full_mid.pt")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            S["sampler"].set_epoch(epoch, start_batch * S["train_loader"].batch_size)
            sums = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0, "grad_norm": 0.0}
            n_sum = 0
            bins = torch.zeros(5, device=device)
            bin_n = torch.zeros(5, device=device)
            t_ep = time.time()
            bi = start_batch
            for batch in S["train_loader"]:
                if args.max_steps_per_epoch and bi >= args.max_steps_per_epoch:
                    break
                batch = C.to_device(batch, device)
                with C.autocast_ctx(cfg, device):
                    out = S["forward"](S["shim"], batch, "fit", cfg)
                out["loss"].backward()
                gn = torch.nn.utils.clip_grad_norm_(params, clip) if clip else torch.tensor(0.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if S["ema"] is not None:
                    S["ema"].update(model)
                global_step += 1
                bi += 1
                for k in ("loss", "pred_loss", "sigreg_loss"):
                    sums[k] += out[k].detach().float()
                sums["grad_norm"] += gn.detach().float()
                n_sum += 1
                if "tau" in out:
                    b = torch.clamp((out["tau"] * 5).long(), max=4)
                    bins.index_add_(0, b, out["per_pair"].float())
                    bin_n.index_add_(0, b, torch.ones_like(out["per_pair"], dtype=torch.float32))

                tnow = time.time()
                if global_step % 20 == 0:
                    rate = 20 / max(tnow - t_last, 1e-6)
                    it_ema = rate if it_ema is None else 0.9 * it_ema + 0.1 * rate
                    t_last = tnow
                if global_step % 200 == 0:
                    log(f"ep {epoch + 1}/{epochs} step {bi}/{S['steps_per_epoch']} gs {global_step} "
                        f"loss {out['loss'].item():.4f} pred {out['pred_loss'].item():.4f} "
                        f"sigreg {out['sigreg_loss'].item():.4f} lr {opt.param_groups[0]['lr']:.3e} "
                        f"gn {float(gn):.3f} {it_ema or 0:.2f} it/s")
                if tnow - prog_t > 10:
                    write_progress(epoch, bi)
                    prog_t = tnow
                if global_step % 50 == 0 and not math.isfinite(out["loss"].item()):
                    raise RuntimeError(f"non-finite loss at step {global_step}")
                if _STOP["flag"] or (args.deadline and tnow > args.deadline):
                    why = _STOP["why"] or "deadline"
                    save_mid(epoch, bi)
                    log(f"stopping ({why}); mid-epoch checkpoint written at epoch {epoch} batch {bi}")
                    raise Stop(why)
                if tnow - last_ckpt_t > args.ckpt_minutes * 60:
                    save_mid(epoch, bi)
                    last_ckpt_t = tnow
            start_batch = 0
            train_time = time.time() - t_ep

            # --- end of epoch: diagnostics, metrics row, checkpoints
            t_val = time.time()
            write_progress(epoch, bi, phase="val")
            diag = diagnostics(S, job, device, args.val_batches)
            row = {"epoch": epoch + 1, "global_step": global_step, "lr": opt.param_groups[0]["lr"],
                   "train_time_s": train_time, "val_time_s": time.time() - t_val}
            for k, v in sums.items():
                row[f"train_{k}"] = float(v / max(n_sum, 1))
            for k in range(5):
                row[f"train_flow_tau{k}"] = float(bins[k] / bin_n[k]) if bin_n[k] > 0 else float("nan")
            row.update(diag)
            _append_metrics(job_dir / "metrics.csv", row)
            log("epoch " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

            ep_done = epoch + 1
            C.torch_save_atomic(full_state(S, job, ep_done, 0, global_step), ckpt_dir / f"full_ep{ep_done}.pt")
            (ckpt_dir / "full_mid.pt").unlink(missing_ok=True)
            (ckpt_dir / "full_resume.pt").unlink(missing_ok=True)  # restored from a GitHub release
            if ep_done in milestones and ep_done != epochs:
                C.torch_save_atomic(model_state(S, job, ep_done, global_step), ckpt_dir / f"model_ep{ep_done}.pt")
            _rotate_full(ckpt_dir, keep=2)
            if ep_done in milestones and ep_done != epochs:  # final epoch: TRAIN_DONE triggers the bundle
                (job_dir / f"milestone_ep{ep_done}").touch()
            last_ckpt_t = time.time()
            if _STOP["flag"] or (args.deadline and time.time() > args.deadline):
                log(f"stopping after epoch {ep_done} ({_STOP['why'] or 'deadline'})")
                raise Stop("deadline")

        _rotate_full(ckpt_dir, keep=1)  # finished: keep only the final full checkpoint
        (job_dir / "TRAIN_DONE").touch()
        write_progress(epochs, 0, phase="train_done")
        log(f"training done in {C.fmt_hms(time.time() - t_job0)}")
        return 0
    except Stop:
        write_progress(epoch, bi, phase="interrupted")
        return EXIT_INTERRUPTED


def _rotate_full(ckpt_dir, keep):
    eps = sorted((int(p.stem.split("ep")[1]), p) for p in ckpt_dir.glob("full_ep*.pt"))
    for _, p in eps[:-keep]:
        p.unlink(missing_ok=True)


def _append_metrics(path, row):
    path = Path(path)
    rows = []
    if path.exists():
        with open(path) as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if int(r["epoch"]) != row["epoch"]] + [row]
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def _dedupe_metrics(path, max_epoch):
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= max_epoch]
    if not rows:
        path.unlink()
        return
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def preflight_step(S, job, device, log):
    model, cfg = S["model"], S["cfg"]
    batch = next(iter(S["train_loader"]))
    batch = C.to_device(batch, device)
    with C.autocast_ctx(cfg, device):
        out = S["forward"](S["shim"], batch, "fit", cfg)
    out["loss"].backward()
    gn = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
    ok = all(torch.isfinite(t).all() for t in (out["loss"], gn))
    log(f"preflight fwd/bwd ok={ok} loss={out['loss'].item():.4f} pred={out['pred_loss'].item():.4f} "
        f"sigreg={out['sigreg_loss'].item():.4f} grad_norm={float(gn):.3f} pixels={tuple(batch['pixels'].shape)} "
        f"action={tuple(batch['action'].shape)}")
    return 0 if ok else 1


def benchmark(S, job, device, args, log, job_dir):
    model, opt, sched, cfg = S["model"], S["opt"], S["sched"], S["cfg"]
    params = list(model.parameters())
    setup_s = time.time() - PROCESS_T0  # imports + dataset open + normalizers + model build
    n, warm = args.benchmark_steps, min(10, args.benchmark_steps // 3)
    S["sampler"].set_epoch(0)
    it = iter(S["train_loader"])
    t0 = None
    t_data = 0.0
    for i in range(n):
        if i == warm:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        td = time.time()
        batch = C.to_device(next(it), device)
        t_data += time.time() - td if i >= warm else 0.0
        with C.autocast_ctx(cfg, device):
            out = S["forward"](S["shim"], batch, "fit", cfg)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    it_s = (n - warm) / dt
    tv = time.time()
    diagnostics(S, job, device, 3)  # warm-up: spawns the (persistent) val workers once, like epoch 1 does
    val_startup_s = time.time() - tv
    nv = 20
    tv = time.time()
    diagnostics(S, job, device, nv)
    val_s_per_batch = (time.time() - tv) / nv
    res = {"train_it_s": it_s, "data_wait_frac": t_data / dt, "val_s_per_batch": val_s_per_batch, "val_startup_s": val_startup_s, "setup_s": setup_s,
           "steps_per_epoch": S["steps_per_epoch"], "n_train": S["n_train"], "n_val": S["n_val"],
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
           "params": C.param_counts(model)}
    C.write_json_atomic(job_dir / "benchmark_train.json", res)
    log(f"benchmark: {it_s:.2f} it/s (data wait {100 * t_data / dt:.0f}%), val {val_s_per_batch:.3f} s/batch")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
