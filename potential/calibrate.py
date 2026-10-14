"""Sigma calibration: per-dim std of Delta = z_{t+1} - z_t on the training split, using the
trained seed-0 baseline encoder (eval mode, training numerics). Writes calibration.json."""

import argparse
import os
import sys
import time

import torch

from potential import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_batches", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    cfg = C.load_train_cfg(os.environ.get("POT_DEBUG_OVERRIDES", "").split())
    _, train_set, val_set = C.build_datasets(cfg)
    train_loader, _, sampler = C.build_loaders(cfg, train_set, val_set, shuffle_seed=args.seed,
                                               num_workers=args.num_workers)
    sampler.set_epoch(0)
    model, ck = C.load_model_from_ckpt(args.ckpt)
    model = model.to(device).eval()

    D = int(cfg.embed_dim)
    s1 = torch.zeros(D, dtype=torch.float64, device=device)
    s2 = torch.zeros(D, dtype=torch.float64, device=device)
    z1 = torch.zeros(D, dtype=torch.float64, device=device)
    z2 = torch.zeros(D, dtype=torch.float64, device=device)
    n = nz = 0
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= args.num_batches:
                break
            batch = C.to_device(batch, device)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            with C.autocast_ctx(cfg, device):
                emb = model.encode(batch)["emb"]
            emb = emb.double()
            delta = (emb[:, 1:] - emb[:, :-1]).reshape(-1, D)
            s1 += delta.sum(0)
            s2 += delta.pow(2).sum(0)
            n += delta.size(0)
            z = emb.reshape(-1, D)
            z1 += z.sum(0)
            z2 += z.pow(2).sum(0)
            nz += z.size(0)
    mean = s1 / n
    std = (s2 / n - mean.pow(2)).clamp_min(0).sqrt() * (n / max(n - 1, 1)) ** 0.5
    zmean = z1 / nz
    zstd = (z2 / nz - zmean.pow(2)).clamp_min(0).sqrt()
    res = {
        "delta_std": std.cpu().tolist(),
        "delta_mean": mean.cpu().tolist(),
        "delta_std_mean": float(std.mean()), "delta_std_min": float(std.min()), "delta_std_max": float(std.max()),
        "z_std_mean": float(zstd.mean()), "delta_over_z_std": float((std / zstd.clamp_min(1e-8)).mean()),
        "n_deltas": n, "num_batches": args.num_batches, "ckpt": args.ckpt, "ckpt_epoch": ck.get("epoch"),
        "source_job": (ck.get("job") or {}).get("job_id"), "seconds": time.time() - t0,
    }
    C.write_json_atomic(args.out, res)
    print(f"[calibration] n={n} deltas | delta std mean {res['delta_std_mean']:.4f} "
          f"(min {res['delta_std_min']:.4f}, max {res['delta_std_max']:.4f}) | z std mean {res['z_std_mean']:.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
