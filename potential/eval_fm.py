"""PushT planning evaluation for baseline and potential checkpoints.

Mirrors eval.py::run line by line (same config/eval/pusht.yaml, same CEM solver config,
same normalisation, same episode/start sampling incl. its quirks); only the model
loading, the output (JSON instead of an appended txt) and the progress reporting
differ. Videos are not written (eval.py writes one per episode; it does not affect
results).

Usage:
  python -m potential.eval_fm --ckpt <path> --out <json> [--S 4 --K 1 --mode sample]
  python -m potential.eval_fm --ckpt <path> --out <json> --benchmark "1x1,4x1,8x1,4x4"
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from potential import common as C

import hydra  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from sklearn import preprocessing  # noqa: E402

from eval import get_dataset, get_episodes_length, img_transform  # noqa: E402  (repo helpers, unchanged)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--progress", default=None, help="progress json path (default: next to --out)")
    p.add_argument("--num_eval", type=int, default=200)
    p.add_argument("--S", type=int, default=None, help="flow sampling steps (default: job plan_S)")
    p.add_argument("--K", type=int, default=1, help="particles per candidate")
    p.add_argument("--mode", default="sample", choices=["sample", "mean"])
    p.add_argument("--plan_seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--benchmark", default=None, help="comma list of SxK to time (one solve each)")
    p.add_argument("--smoke", action="store_true", help="2 envs, tiny CEM: dependency check only")
    return p.parse_args()


def _first_env(v):
    """Slice one env out of a planning info dict value (tensor / ndarray / list)."""
    if torch.is_tensor(v):
        return v[:1].clone()
    if isinstance(v, np.ndarray):
        return v[:1].copy()
    if isinstance(v, (list, tuple)):
        return type(v)(v[:1])
    return v


class Progress:
    def __init__(self, path, n_envs, n_iters):
        self.path = path
        self.n_envs = n_envs
        self.n_iters = n_iters
        self.calls = 0
        self.t0 = time.time()
        self.t_last = 0.0
        self.terminated = 0
        self.solve_times = []  # (seconds, n_envs_in_call)
        self.env_step = 0

    def expected_calls(self):
        # each alive env replans at env-step 0 and 25 (receding horizon 5 blocks x frameskip 5)
        return self.n_iters * self.n_envs * 2

    def write(self, force=False):
        t = time.time()
        if not force and t - self.t_last < 10:
            return
        self.t_last = t
        frac = min(self.calls / max(self.expected_calls(), 1), 1.0)
        el = t - self.t0
        C.write_json_atomic(self.path, {
            "phase": "eval", "calls": self.calls, "expected_calls": self.expected_calls(), "frac": frac,
            "episodes_done": int(self.terminated), "episodes_total": self.n_envs, "env_step": self.env_step,
            "elapsed_s": el, "eta_s": (el / frac - el) if frac > 0.02 else None, "time": t,
        })


def main():
    args = parse()
    device = torch.device(args.device)
    out_path = Path(args.out)
    prog_path = Path(args.progress) if args.progress else out_path.with_name("progress_eval.json")

    n_eval = 2 if args.smoke else args.num_eval
    overrides = [f"eval.num_eval={n_eval}", "policy=potential"]
    if device.type != "cuda":
        overrides.append(f"solver.device={device.type}")
    if args.smoke:
        overrides += ["solver.num_samples=8", "solver.n_steps=2", "solver.topk=2"]
    overrides += os.environ.get("POT_DEBUG_EVAL_OVERRIDES", "").split()  # testing only
    cfg = C.load_eval_cfg(overrides)

    # ---------------- from eval.py::run (unchanged logic) ----------------
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    model, ck = C.load_model_from_ckpt(args.ckpt)
    model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    job = ck.get("job", {})
    is_flow = hasattr(model, "head")
    S_steps = args.S if args.S is not None else int(job.get("plan_S", 1))
    if is_flow:
        model.plan_S, model.plan_K, model.plan_mode, model.plan_seed = S_steps, args.K, args.mode, args.plan_seed
        model.reset_plan_rng()
    elif args.K != 1 or args.mode != "sample" or (args.S not in (None, 1)):
        raise ValueError("S/K/mode only apply to sampled (potential) models")

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")
    # ---------------------------------------------------------------------

    prog = Progress(prog_path, cfg.eval.num_eval, int(cfg.solver.n_steps))
    orig_get_cost = model.get_cost
    captured = {}

    def counting_get_cost(info_dict, action_candidates):
        prog.calls += 1
        prog.write()
        return orig_get_cost(info_dict, action_candidates)

    model.get_cost = counting_get_cost
    orig_solve = solver.solve

    def timed_solve(info_dict, init_action=None):
        if "info" not in captured:
            captured["info"] = {k: _first_env(v) for k, v in info_dict.items()}
        n = len(next(iter(info_dict.values())))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        res = orig_solve(info_dict, init_action=init_action)
        if device.type == "cuda":
            torch.cuda.synchronize()
        prog.solve_times.append((time.time() - t, n))
        return res

    solver.solve = timed_solve
    orig_get_action = policy.get_action

    def counting_get_action(info_dict, **kw):
        a = orig_get_action(info_dict, **kw)
        prog.env_step += 1
        if world.terminateds is not None:
            prog.terminated = max(prog.terminated, int(np.sum(world.terminateds)))
        prog.write()
        return a

    policy.get_action = counting_get_action
    world.set_policy(policy)

    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=C.OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=None,
    )
    wall = time.time() - t0
    prog.write(force=True)

    succ = np.asarray(metrics["episode_successes"], dtype=bool)
    n = int(succ.size)
    p = float(succ.mean()) if n else float("nan")
    solve_total = sum(t for t, _ in prog.solve_times)
    env_solves = sum(k for _, k in prog.solve_times)
    res = {
        "ckpt": str(args.ckpt), "epoch": ck.get("epoch"), "job_id": job.get("job_id"),
        "is_flow": is_flow, "variant": job.get("variant"), "sigma_mult": job.get("sigma"),
        "S": S_steps if is_flow else None, "K": args.K if is_flow else None,
        "mode": args.mode if is_flow else None, "plan_seed": args.plan_seed,
        "n_episodes": n, "success_rate": 100.0 * p, "se": 100.0 * float(np.sqrt(p * (1 - p) / n)) if n else None,
        "episode_successes": succ.astype(int).tolist(),
        "eval_episode_idx": np.asarray(eval_episodes).tolist(), "eval_start_idx": np.asarray(eval_start_idx).tolist(),
        "eval_set_hash": C.config_hash({"e": np.asarray(eval_episodes).tolist(), "s": np.asarray(eval_start_idx).tolist()}),
        "wall_s": wall, "solve_total_s": solve_total, "env_solves": env_solves,
        "solve_s_per_env": solve_total / max(env_solves, 1),
        "solver": C.OmegaConf.to_container(cfg.solver, resolve=True),
        "plan_config": C.OmegaConf.to_container(cfg.plan_config, resolve=True),
        "eval_cfg": {k: v for k, v in C.OmegaConf.to_container(cfg.eval, resolve=True).items() if k != "callables"},
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "smoke": bool(args.smoke),
    }

    if args.benchmark:
        info = captured.get("info")
        bench = {}
        for spec in args.benchmark.split(","):
            s, k = (int(x) for x in spec.lower().split("x"))
            if is_flow:
                model.plan_S, model.plan_K = s, k
                model.reset_plan_rng()
            elif (s, k) != (1, 1):
                continue
            ts = []
            for _ in range(2):
                inf = {kk: _first_env(v) for kk, v in info.items()}
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t = time.time()
                orig_solve(inf)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.time() - t)
            bench[spec] = min(ts)
        res["benchmark_solve_s"] = bench
        res["env_overhead_s_per_episode"] = max(wall - solve_total, 0.0) / max(n, 1)
    C.write_json_atomic(out_path, res)
    print(f"[eval] {Path(args.ckpt).name} S={res['S']} K={res['K']} mode={res['mode']}: "
          f"success {res['success_rate']:.1f}% +- {res['se'] or 0:.1f} (n={n}) | "
          f"{res['solve_s_per_env']:.2f} s/solve | wall {C.fmt_hms(wall)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
