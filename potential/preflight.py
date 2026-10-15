"""Pre-flight: fail fast before any training.

Checks GPU, both PushT dataset files (exist + load), free disk, one fwd/bwd pass of the
baseline and the potential models (P1, P2, P3) on a real batch with the training numerics, and
an eval smoke test (HDF5 eval dataset + PushT env + CEM) on random-init checkpoints.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from potential import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out")
    p.add_argument("--min_free_gb", type=float, default=20.0)
    p.add_argument("--allow_cpu", action="store_true", help="testing only")
    p.add_argument("--skip_eval_smoke", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    pre = out / "preflight"
    pre.mkdir(parents=True, exist_ok=True)
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    fails = []

    def check(name, ok, detail):
        res["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"[preflight] {'OK  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            fails.append(name)

    # 1. GPU
    gpu = torch.cuda.is_available()
    check("gpu", gpu or args.allow_cpu,
          torch.cuda.get_device_name(0) if gpu else "no CUDA device visible to torch")
    if fails:
        return finish(out, res, fails)
    device = torch.device("cuda" if gpu else "cpu")

    # 2. datasets
    home = C.stablewm_home()
    lance = home / "datasets" / "pusht_expert_train.lance"
    h5 = home / "datasets" / "pusht_expert_train.h5"
    debug = os.environ.get("POT_DEBUG_OVERRIDES", "")
    check("lance_train_data", lance.is_dir() or "data.dataset.name=" in debug,
          f"{lance} ({_du(lance):.1f} GB)" if lance.exists() else f"missing: {lance}")
    check("hdf5_eval_data", h5.is_file(), f"{h5} ({_du(h5):.1f} GB)" if h5.exists() else f"missing: {h5}")
    if fails:
        return finish(out, res, fails)

    # 3. disk
    free = shutil.disk_usage(out).free / 1e9
    check("disk", free >= args.min_free_gb, f"{free:.1f} GB free under {out.resolve()} (need >= {args.min_free_gb:.1f})")

    # 4. datasets load + fwd/bwd
    t = time.time()
    cfg = C.load_train_cfg(debug.split())
    try:
        dataset, train_set, val_set = C.build_datasets(cfg)
        spe = len(train_set) // int(cfg.loader.batch_size)
        check("lance_load", True, f"{len(dataset)} windows ({len(dataset.lengths)} episodes), train {len(train_set)}, "
                                  f"val {len(val_set)}, {spe} steps/epoch, action_dim*frameskip="
                                  f"{cfg.model.action_encoder.input_dim} ({time.time() - t:.0f}s)")
        res["steps_per_epoch"] = spe
    except Exception as e:  # noqa: BLE001
        check("lance_load", False, repr(e))
        return finish(out, res, fails)
    try:
        import stable_worldmodel as swm

        ds = swm.data.HDF5Dataset("pusht_expert_train", keys_to_cache=["action", "proprio", "state"],
                                  cache_dir=os.environ.get("STABLEWM_HOME"))
        check("hdf5_load", True, f"{len(ds.lengths)} episodes, {int(ds.lengths.sum())} frames, columns {ds.column_names}")
        res["hdf5_episodes"] = int(len(ds.lengths))
        res["lance_episodes"] = int(len(dataset.lengths))
        same = len(ds.lengths) == len(dataset.lengths) and int(ds.lengths.sum()) == int(dataset.lengths.sum())
        print(f"[preflight] note: Lance vs HDF5 episode/frame counts {'match' if same else 'DIFFER'} "
              f"({len(dataset.lengths)}/{int(dataset.lengths.sum())} vs {len(ds.lengths)}/{int(ds.lengths.sum())})")
        res["lance_hdf5_counts_match"] = bool(same)
    except Exception as e:  # noqa: BLE001
        check("hdf5_load", False, repr(e))
        return finish(out, res, fails)

    loader = torch.utils.data.DataLoader(train_set, batch_size=int(cfg.loader.batch_size), shuffle=True,
                                         num_workers=min(4, int(cfg.loader.num_workers)) if gpu else 0)
    batch = C.to_device(next(iter(loader)), device)
    sigreg = C.make_sigreg(cfg).to(device)
    ckpts = {}
    for job in (dict(job_id="pf_baseline", kind="baseline"),
                dict(job_id="pf_P1", kind="potential", variant="P1", sigma=0.0, plan_S=1),
                dict(job_id="pf_P2", kind="potential", variant="P2", sigma=1.0, plan_S=4),
                dict(job_id="pf_P3", kind="potential", variant="P3", sigma=1.0, plan_S=4)):
        try:
            mc = C.model_config(cfg, job, sigma_std=[1.0] * int(cfg.embed_dim))
            torch.manual_seed(0)
            model = C.instantiate_model(mc).to(device)
            fwd = C.baseline_forward if job["kind"] == "baseline" else C.potential_forward
            b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            if gpu:
                torch.cuda.reset_peak_memory_stats()
            with C.autocast_ctx(cfg, device):
                o = fwd(C.Shim(model, sigreg), b, "fit", cfg)
            o["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ok = bool(torch.isfinite(o["loss"]) and torch.isfinite(gn))
            mem = torch.cuda.max_memory_allocated() / 1e9 if gpu else 0
            check(f"fwd_bwd_{job['job_id']}", ok, f"loss {o['loss'].item():.4f} pred {o['pred_loss'].item():.4f} "
                                                  f"sigreg {o['sigreg_loss'].item():.4f} grad {float(gn):.3f} "
                                                  f"peak mem {mem:.1f} GB | params {C.param_counts(model)['total']}")
            if job["kind"] == "potential":
                res.setdefault("bf16_field_rel_err", {})[job["job_id"]] = rel = _bf16_field_err(model, o, cfg, device)
                print(f"[preflight] note: {job['job_id']} grad_x U bf16 vs fp32 relative error {rel:.4f} on a real batch "
                      f"(random non-zero output layer){' -- large: consider --head_fp32' if rel > 0.05 else ''}", flush=True)
            if job["job_id"] in ("pf_baseline", "pf_P2", "pf_P3"):
                path = pre / f"{job['job_id']}.pt"
                C.torch_save_atomic({"model": model.state_dict(), "model_cfg": mc, "epoch": 0, "job": job}, path)
                ckpts[job["job_id"]] = path
            del model, o
        except Exception as e:  # noqa: BLE001
            check(f"fwd_bwd_{job['job_id']}", False, repr(e))

    # 5. eval smoke (subprocess: same code path as the real evals)
    if not args.skip_eval_smoke and not fails:
        for name, path in ckpts.items():
            outj = pre / f"smoke_{name}.json"
            cmd = [sys.executable, "-m", "potential.eval_fm", "--ckpt", str(path), "--out", str(outj), "--smoke"]
            t = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            ok = r.returncode == 0 and outj.exists()
            check(f"eval_smoke_{name}", ok, f"{time.time() - t:.0f}s" if ok else (r.stdout + r.stderr)[-2000:])
    return finish(out, res, fails)


def _bf16_field_err(model, o, cfg, device):
    """||v_bf16 - v_fp32|| / ||v_fp32|| for v = -grad_x U on real (h, z_t, a_t) pairs, with a copy of the
    model whose zero-initialised output/modulation layers are randomised (else v == 0)."""
    import copy

    m = copy.deepcopy(model).float().eval()
    g = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for name, p in m.head.named_parameters():
            if name.startswith("out.") or ".mod." in name:
                p.copy_(torch.randn(p.shape, generator=g, device=device) / max(p.shape[-1], 1) ** 0.5)
    H = int(cfg.history_size)
    flat = lambda t: t.detach().float()[:, :H].reshape(-1, t.size(-1))
    h, z, a = o["h"].detach().float().reshape(-1, o["h"].size(-1)), flat(o["emb"]), flat(o["act_emb"])
    x = z + 0.1 * torch.randn(z.shape, generator=g, device=device)
    tau = torch.rand(z.size(0), generator=g, device=device)
    v32 = m.velocity(x, tau, h, z, a)
    with C.autocast_ctx(cfg, device):
        v16 = m.velocity(x, tau, h, z, a)
    return float((v16.float() - v32).norm() / v32.norm().clamp_min(1e-30))


def _du(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e9
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def finish(out, res, fails):
    res["ok"] = not fails
    res["failed"] = fails
    C.write_json_atomic(Path(out) / "preflight.json", res)
    print(f"[preflight] {'PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
