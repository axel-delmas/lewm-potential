"""Unattended job runner for the PushT potential-field study (called by run_all.sh).

The baseline and the sigma calibration are not re-run here: they are imported from the bridge
repo's results (--bridge_out, github.com/axel-delmas/fm-jepa), so that P1/P2 are compared with the
exact baseline the bridge variants were compared with (same eval start states, same epochs).

Runs the tiered job plan top to bottom with dependencies, writes out/manifest.json,
out/progress.json, out/results.csv, out/report.md, rebuilds out -> bundle.tar.gz after
every finished job and every milestone checkpoint, uploads to the HF Hub if configured,
handles --max_hours, SIGTERM/SIGINT and crashes.
"""

import argparse
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

from potential import common as C

PY = sys.executable
ONE_GB = 1 << 30
HF_EVERY_S = 15 * 60
FALLBACK_IT_S = {"baseline": 6.0, "potential": 5.5}
FALLBACK_SOLVE_S = 1.0
RESERVE_S = 600  # time kept for the final bundle + upload when --max_hours is set


# ============================================================================ args / logging
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.environ.get("OUT_DIR", "out"))
    p.add_argument("--tiers", default="1,2,3")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--max_hours", type=float, default=None)
    p.add_argument("--price_per_hour", type=float, default=0.0)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run jobs whose config hash changed")
    p.add_argument("--hold_after_done", action="store_true")
    p.add_argument("--num_eval", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=None,
                   help="dataloader workers; default max(6 (repo), min(cpus-2, 16)). Throughput only, not numerics")
    p.add_argument("--val_batches", type=int, default=100)
    p.add_argument("--sweep_grid", action="store_true", help="tier-4 planning sweep as full S x K grid")
    p.add_argument("--t2_variants", default="P1,P2", help="tier-2 variants to train (budget control)")
    p.add_argument("--t2_sigmas", default="0.5,1", help="tier-2 sigma multipliers to train (budget control)")
    p.add_argument("--final_eval_only", action="store_true",
                   help="evaluate only the final checkpoint (skip epoch-5/10 milestone evals; weights still saved)")
    p.add_argument("--detach_target", action="store_true", help="potential jobs: stop-gradient on z_{t+1} (off by default)")
    p.add_argument("--ema", action="store_true", help="potential jobs: EMA target encoder (off by default)")
    p.add_argument("--head_dropout", type=float, default=0.0, help="potential jobs: potential-head dropout (default 0)")
    p.add_argument("--head_fp32", action="store_true",
                   help="potential jobs: run the potential head and grad_x U in fp32 inside the bf16 step (default off)")
    p.add_argument("--bridge_out", action="append", default=[],
                   help="bridge-repo results (out/ dir or bundle .tar.gz; repeatable): source of the imported "
                        "baseline jobs and calibration.json, and of the joined comparison in the report")
    p.add_argument("--extend_mode", default="fresh", choices=["fresh", "continue"])
    p.add_argument("--no_auto_benchmark", action="store_true")
    p.add_argument("--no_auto_diagnose", action="store_true")
    p.add_argument("--skip_preflight", action="store_true")
    p.add_argument("--bundle_path", default=None)
    p.add_argument("--allow_cpu", action="store_true", help="testing only")
    p.add_argument("--allow_no_upload", action="store_true",
                   help="run even if no GitHub/HF upload is configured (results then only live on this machine)")
    p.add_argument("--debug_steps", type=int, default=0, help="testing only: truncate epochs to N steps")
    return p.parse_args()


class Log:
    def __init__(self, out: Path):
        self.f = None
        if os.environ.get("POT_STDOUT_IS_RUNLOG") != "1":
            out.mkdir(parents=True, exist_ok=True)
            self.f = open(out / "run.log", "a", buffering=1)
        self.lock = threading.Lock()

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self.lock:
            print(line, flush=True)
            if self.f:
                self.f.write(line + "\n")


# ============================================================================ plan
MILESTONES = (5, 10, 20, 50, 100)


def default_S(kind, variant, sigma):
    if kind == "baseline":
        return None
    return 1 if (variant == "P1" and float(sigma) == 0.0) else 4


def eval_tag(epoch, S, K, mode):
    return f"ep{epoch}" if S is None else f"ep{epoch}_S{S}_K{K}_{mode}"


def canonical_id(kind, variant, sigma, seed, epochs, lam, W=256, out_norm=None):
    lam_part = "" if abs(lam - 0.09) < 1e-12 else f"_lam{lam:g}"
    if kind == "baseline":
        return f"baseline{lam_part}_seed{seed}_e{epochs}"
    s = f"{variant}_s{float(sigma):g}"
    if int(W) != 256:
        s += f"_W{W}"
    if out_norm:
        s += f"_norm{out_norm}"
    return s + f"{lam_part}_seed{seed}_e{epochs}"


def train_config(args, tier, kind, seed, variant=None, sigma=0.0, lam=0.09, W=256, out_norm=None, final_only=False,
                 job_id=None):
    E = args.epochs
    S = default_S(kind, variant, sigma)
    milestones = sorted({m for m in MILESTONES if m <= E} | {E})
    eval_epochs = [E] if (final_only or args.final_eval_only) else sorted({m for m in (5, 10, 20) if m <= E} | {E})
    jid = job_id
    if jid is None:
        jid = canonical_id(kind, variant, sigma, seed, E, lam, W, out_norm)
        if kind == "potential" and flag_suffix(args):  # non-default training flags never share ids with default runs
            jid = jid.replace(f"_seed{seed}", f"{flag_suffix(args)}_seed{seed}")
    cfg = {
        "job_id": jid,
        "tier": tier, "type": "train", "kind": kind, "seed": int(seed), "epochs": E, "lambda": float(lam),
        "milestones": milestones, "num_eval": args.num_eval,
        "evals": [{"tag": eval_tag(e, S, 1, "sample"), "epoch": e, "S": S, "K": 1, "mode": "sample"} for e in eval_epochs],
    }
    if kind == "potential":
        cfg.update(variant=variant, sigma=float(sigma), plan_S=S, head_W=int(W), head_blocks=4,
                   head_dropout=float(args.head_dropout), head_out_norm=out_norm,
                   detach_target=bool(args.detach_target), ema=bool(args.ema), head_fp32=bool(args.head_fp32))
    return cfg


def imported(cfg):
    """Mark a baseline / calibration job as imported from the bridge repo instead of run here."""
    return dict(cfg, imported_from="bridge")


def flag_suffix(args):
    return ("_detach" if args.detach_target else "") + ("_ema" if args.ema else "") + \
        (f"_hdrop{args.head_dropout:g}" if args.head_dropout else "") + ("_fp32" if args.head_fp32 else "")


def build_plan(args, tiers, seeds):
    """Ordered slots. Each slot: {id, tier, config | resolve...}; slot id == job id == job directory."""
    s0 = seeds[0]
    E = args.epochs
    sfx = flag_suffix(args)
    plan = []

    def add(tier, cfg, needs=()):
        plan.append({"id": cfg["job_id"], "tier": tier, "config": cfg, "needs": list(needs)})
        return cfg["job_id"]

    base0 = train_config(args, 1, "baseline", s0)["job_id"]
    ctrl0 = train_config(args, 1, "potential", s0, "P1", 0.0)["job_id"]
    calib_id = f"calibration_seed{s0}_e{E}"
    # without tier 1 there is no sigma=0 control and no gate; the seed-0 baseline and the calibration are still
    # imported as prerequisites (sigma is a multiple of the calibrated std; the baseline is the comparison number).
    # Both come from the bridge repo's results (same job ids), never re-run here.
    gate = ["gate"] if 1 in tiers else []
    if 1 in tiers or any(t in tiers for t in (2, 3, 4)):
        add(1, imported(train_config(args, 1, "baseline", s0)))
        plan.append({"id": calib_id, "tier": 1, "config": imported({"job_id": calib_id, "tier": 1, "type": "calibrate",
                                                                      "ref_job": base0, "seed": s0, "num_batches": 200})})
    if 1 in tiers:
        add(1, train_config(args, 1, "potential", s0, "P1", 0.0))
    t2 = []
    if 2 in tiers:
        for v in [x.strip() for x in str(args.t2_variants).split(",") if x.strip()]:
            for sg in [float(x) for x in str(args.t2_sigmas).split(",") if x.strip()]:
                t2.append(add(2, train_config(args, 2, "potential", s0, v, sg), ["calibration", *gate]))
    if 3 in tiers:
        for s in seeds[1:]:
            add(3, imported(train_config(args, 3, "baseline", s)), gate)
            add(3, train_config(args, 3, "potential", s, "P1", 0.0), gate)
            for which in ("best_P1", "best_P2"):
                plan.append({"id": f"{which.replace('_', '')}{sfx}_seed{s}_e{E}", "tier": 3, "resolve": which,
                             "seed": s, "needs": [*gate, "calibration", which]})
    if 4 in tiers:
        for v in ("P1", "P2"):
            for sg in (0.25, 2.0):
                add(4, train_config(args, 4, "potential", s0, v, sg, final_only=True), ["calibration", *gate])
        for lam in (0.03, 0.3):
            plan.append({"id": f"bestT2_lam{lam:g}{sfx}_seed{s0}_e{E}", "tier": 4, "resolve": "best_T2", "lam": lam,
                         "seed": s0, "needs": [*gate, "calibration", "best_T2"]})
        for W in (128, 512):
            add(4, train_config(args, 4, "potential", s0, "P1", 0.0, W=W, final_only=True), gate)
        for which in ("best_P1", "best_P2"):
            plan.append({"id": f"plansweep_{which.replace('_', '')}_seed{s0}_e{E}", "tier": 4, "resolve": "sweep",
                         "ref": which, "needs": [*gate, which]})
    return plan, {"base0": base0, "ctrl0": ctrl0, "calib": calib_id, "t2": t2}


def sweep_evals(args, S0=4):
    if args.sweep_grid:
        specs = [(s, k, "sample") for s in (1, 2, 4, 8) for k in (1, 2, 4)]
    else:
        specs = [(s, 1, "sample") for s in (1, 2, 4, 8)] + [(S0, k, "sample") for k in (2, 4)]
    specs.append((S0, 1, "mean"))
    E = args.epochs
    return [{"tag": eval_tag(E, s, k, m), "epoch": E, "S": s, "K": k, "mode": m} for s, k, m in specs]


# ============================================================================ orchestrator
class Orchestrator:
    def __init__(self, args):
        if args.num_workers is None:
            args.num_workers = max(6, min((os.cpu_count() or 8) - 2, 16))
        self.args = args
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.out / "jobs"
        self.jobs_dir.mkdir(exist_ok=True)
        self.log = Log(self.out)
        self.t_start = float(os.environ.get("POT_RUN_START", time.time()))
        self.deadline = self.t_start + args.max_hours * 3600 if args.max_hours else None
        self.tiers = [int(t) for t in str(args.tiers).split(",") if t.strip()]
        self.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
        self.stop = threading.Event()
        self.stop_reason = ""
        self.child = None
        self.current = {"slot": None, "phase": None, "progress_file": None, "index": 0}
        self.man_lock = threading.Lock()
        ws = Path("/workspace")
        self.bundle_path = Path(args.bundle_path) if args.bundle_path else (
            ws / "bundle.tar.gz" if ws.is_dir() else Path.cwd() / "bundle.tar.gz")
        self.bundler = Bundler(self)
        from potential.github_sync import GitHubSync

        self.gh = GitHubSync(self.out, log=self.log)
        self.manifest = C.read_json(self.out / "manifest.json") or {}
        self.bench = C.read_json(self.out / "benchmark.json") or {}
        self.seen_milestones = set()
        self._bridge_dirs = None

    # ------------------------------------------------------------------ manifest
    def save_manifest(self):
        with self.man_lock:
            self.manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            C.write_json_atomic(self.out / "manifest.json", self.manifest)

    def jentry(self, jid):
        return self.manifest.setdefault("jobs", {}).setdefault(jid, {"status": "pending", "timings": {}})

    def set_status(self, jid, status, **kw):
        e = self.jentry(jid)
        e["status"] = status
        e.update(kw)
        self.save_manifest()

    def init_manifest(self):
        m = self.manifest
        m.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
        m["git_commit"] = C.git_commit()
        try:
            import torch

            m["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:
            m["gpu"] = "?"
        freeze = self.out / "env" / "pip_freeze.txt"
        m["pip_freeze_file"] = str(freeze.relative_to(self.out)) if freeze.exists() else None
        m["pip_freeze"] = freeze.read_text().splitlines() if freeze.exists() else None
        m["args"] = {k: v for k, v in vars(self.args).items()}
        m.setdefault("choices", {})
        m.setdefault("runs", []).append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": m["args"]})
        self.save_manifest()

    # ------------------------------------------------------------------ resolution
    def calib_path(self):
        return self.out / "calibration.json"

    def resolve(self, slot):
        """Return the concrete job config for a slot, or None if not resolvable yet."""
        if "config" in slot:
            return slot["config"]
        ch = self.manifest.get("choices", {})
        if slot["resolve"] in ("best_P1", "best_P2", "best_T2"):
            c = ch.get(slot["resolve"])
            if not c:
                return None
            lam = slot.get("lam", 0.09)
            cfg = train_config(self.args, slot["tier"], "potential", slot["seed"], c["variant"], c["sigma"], lam=lam,
                               final_only=slot["tier"] == 4, job_id=slot["id"])
            cfg["resolved_from"] = {"choice": slot["resolve"], "source_job": c["job_id"]}
            return cfg
        if slot["resolve"] == "sweep":
            c = ch.get(slot["ref"])
            if not c:
                return None
            ref = c["job_id"]
            rcfg = self.jentry(ref).get("config") or {}
            return {"job_id": slot["id"], "tier": 4, "type": "eval_sweep", "ref_job": ref, "kind": "potential",
                    "variant": rcfg.get("variant"), "sigma": rcfg.get("sigma"), "seed": rcfg.get("seed"),
                    "epochs": self.args.epochs, "num_eval": self.args.num_eval,
                    "evals": sweep_evals(self.args, rcfg.get("plan_S") or 4)}
        raise ValueError(slot)

    def full_config(self, cfg):
        """Job config + things that change its meaning (calibration content for sigma>0)."""
        cfg = dict(cfg)
        if cfg.get("type") == "train" and cfg.get("kind") == "potential" and float(cfg.get("sigma", 0)) > 0:
            calib = C.read_json(self.calib_path())
            cfg["calibration_path"] = str(self.calib_path())
            cfg["calibration_hash"] = C.config_hash(calib["delta_std"]) if calib else None
        if self.args.debug_steps:
            cfg["debug_steps"] = self.args.debug_steps
        return cfg

    def config_hash(self, cfg):
        return C.config_hash({k: v for k, v in cfg.items() if k not in ("calibration_path", "resolved_from")})

    # ------------------------------------------------------------------ needs
    def needs_ok(self, slot):
        ch = self.manifest.get("choices", {})
        for need in slot.get("needs", []):
            if need == "gate":
                g = ch.get("gate")
                if not g:
                    return False, "gate not evaluated"
                if g["verdict"] != "pass":
                    return False, "blocked: gate failed"
            elif need == "calibration":
                if not self.calib_path().exists():
                    return False, "calibration missing"
            elif need in ("best_P1", "best_P2", "best_T2"):
                if not ch.get(need):
                    return False, f"{need} not selected"
        return True, ""

    # ------------------------------------------------------------------ projections
    def spe(self):
        for jid, e in self.manifest.get("jobs", {}).items():
            pr = C.read_json(self.jobs_dir / jid / "progress.json")
            if pr and pr.get("steps_per_epoch"):
                return pr["steps_per_epoch"]
        return self.bench.get("steps_per_epoch") or 13934

    def train_type(self, cfg):
        return "baseline" if cfg.get("kind") == "baseline" else f"potential_W{cfg.get('head_W', 256)}"

    def measured_it_s(self, ttype):
        rates = []
        for jid, e in self.manifest.get("jobs", {}).items():
            cfg = e.get("config") or {}
            if cfg.get("type") != "train" or self.train_type(cfg) != ttype:
                continue
            pr = C.read_json(self.jobs_dir / jid / "progress.json") or {}
            if pr.get("it_s"):
                rates.append(pr["it_s"])
        return sum(rates) / len(rates) if rates else None

    def it_s(self, ttype):
        m = self.measured_it_s(ttype)
        if m:
            return m, "measured"
        b = (self.bench.get("train") or {}).get(ttype) or (self.bench.get("train") or {}).get("potential_W256" if ttype.startswith("potential") else "baseline")
        if b:
            return b["train_it_s"], "benchmark"
        return FALLBACK_IT_S["baseline" if ttype == "baseline" else "potential"], "fallback"

    def val_s_per_epoch(self, ttype):
        """Measured per-epoch val time from metrics.csv of jobs of this type, else the benchmark."""
        import csv

        vals = []
        for jid, e in self.manifest.get("jobs", {}).items():
            cfg = e.get("config") or {}
            mp = self.jobs_dir / jid / "metrics.csv"
            if cfg.get("type") != "train" or self.train_type(cfg) != ttype or not mp.exists():
                continue
            try:
                with open(mp) as f:
                    vals += [float(r["val_time_s"]) for r in csv.DictReader(f) if r.get("val_time_s")]
            except (OSError, ValueError, KeyError):
                pass
        if vals:
            vals.sort()
            return vals[len(vals) // 2]
        b = (self.bench.get("train") or {}).get(ttype) or {}
        return b.get("val_s_per_batch", 0.15) * max(self.args.val_batches, 0) + 10

    def solve_s(self, cfg_kind, W, S, K):
        """Measured from finished evals, else benchmark table, else a rough scaling."""
        for jid, e in self.manifest.get("jobs", {}).items():
            for p in (self.jobs_dir / jid / "eval").glob("*.json") if (self.jobs_dir / jid / "eval").is_dir() else []:
                r = C.read_json(p) or {}
                if "solve_s_per_env" not in r or r.get("smoke"):
                    continue
                if (cfg_kind == "baseline" and not r.get("is_flow")) or (
                        cfg_kind != "baseline" and r.get("is_flow") and r.get("S") == S and r.get("K") == K):
                    return r["solve_s_per_env"]
        be = self.bench.get("eval") or {}
        if cfg_kind == "baseline" and be.get("baseline"):
            return be["baseline"]["solve_s"].get("1x1", FALLBACK_SOLVE_S)
        fl = be.get("potential") or {}
        if fl.get("solve_s", {}).get(f"{S}x{K}"):
            return fl["solve_s"][f"{S}x{K}"]
        base = (be.get("baseline") or {}).get("solve_s", {}).get("1x1", FALLBACK_SOLVE_S)
        if cfg_kind == "baseline":
            return base
        return base * (K or 1) * (1 + 0.15 * (S or 1))

    def eval_s(self, cfg_kind, W, S, K, n=None):
        n = n or self.args.num_eval
        be = self.bench.get("eval") or {}
        b = be.get("potential" if cfg_kind != "baseline" else "baseline") or {}
        setup = b.get("setup_s", 90.0)
        envo = b.get("env_overhead_s_per_episode", 0.5)
        return setup + n * (2 * self.solve_s(cfg_kind, W, S, K) + envo)

    def project(self, slot, cfg=None, remaining_only=True):
        """Projected seconds for a slot (remaining work if remaining_only)."""
        cfg = cfg or self.resolve(slot) or self.representative(slot)
        jd = self.jobs_dir / slot["id"]
        t = 0.0
        if cfg.get("imported_from"):
            return 0.0 if (remaining_only and (jd / "DONE").exists()) else 5.0
        if cfg.get("type") == "calibrate":
            return 0.0 if (remaining_only and self.calib_path().exists()) else 180.0
        if cfg.get("type") == "train" and not (remaining_only and (jd / "TRAIN_DONE").exists()):
            ttype = self.train_type(cfg)
            it, _ = self.it_s(ttype)
            spe = self.args.debug_steps or self.spe()
            epochs_left = cfg["epochs"]
            if remaining_only:
                pr = C.read_json(jd / "progress.json") or {}
                if pr.get("total_steps") and pr.get("global_step") is not None and pr.get("phase") != "train_done":
                    frac_left = 1 - pr["global_step"] / pr["total_steps"]
                    epochs_left = cfg["epochs"] * frac_left
            setup = ((self.bench.get("train") or {}).get(ttype) or {}).get("setup_s", 60.0)
            t += setup + epochs_left * (spe / it + self.val_s_per_epoch(ttype) + 8)
        for ev in cfg.get("evals", []):
            if remaining_only and (jd / "eval" / f"{ev['tag']}.json").exists():
                continue
            t += self.eval_s(cfg.get("kind"), cfg.get("head_W", 256), ev.get("S"), ev.get("K"))
        return t

    def representative(self, slot):
        if slot.get("resolve") == "sweep":
            return {"type": "eval_sweep", "kind": "potential", "evals": sweep_evals(self.args)}
        return train_config(self.args, slot["tier"], "potential", slot.get("seed", 0), "P2", 1.0,
                            final_only=slot["tier"] == 4, job_id=slot["id"])

    # ------------------------------------------------------------------ subprocess
    def run_sub(self, cmd, log_path, progress_file=None, phase=None, deadline=None):
        self.current.update(phase=phase, progress_file=progress_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        with open(log_path, "a") as lf:
            lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
            lf.flush()
            self.child = subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT, cwd=str(C.REPO_ROOT),
                                          env=env, start_new_session=True)
            while True:
                try:
                    rc = self.child.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if deadline and time.time() > deadline + 300 and self.child.poll() is None:
                    self.log(f"deadline passed by 5 min; terminating {phase}")
                    self.child.send_signal(signal.SIGTERM)
                    deadline = None
            self.child = None
        self.current.update(phase=None, progress_file=None)
        if rc not in (0, 3):
            tail = log_path.read_text()[-3000:] if log_path.exists() else ""
            self.log(f"subprocess failed (rc={rc}): {' '.join(map(str, cmd[:4]))} ...\n----- tail of {log_path} -----\n{tail}\n-----")
        return rc

    def train_deadline(self):
        return self.deadline - RESERVE_S if self.deadline else None

    def remaining_budget(self):
        return (self.deadline - RESERVE_S - time.time()) if self.deadline else math.inf

    # ------------------------------------------------------------------ job execution
    def run_train(self, slot, cfg):
        jid = cfg["job_id"]
        jd = self.jobs_dir / jid
        e = self.jentry(jid)
        t0 = time.time()
        if not (jd / "TRAIN_DONE").exists():
            self.current["phase_label"] = "train"
            cmd = [PY, "-m", "potential.train_fm", "--job_dir", jd, "--val_batches", self.args.val_batches]
            if self.train_deadline():
                cmd += ["--deadline", self.train_deadline()]
            if self.args.num_workers is not None:
                cmd += ["--num_workers", self.args.num_workers]
            if self.args.debug_steps:
                cmd += ["--max_steps_per_epoch", self.args.debug_steps]
            if self.args.allow_cpu:
                cmd += ["--device", "cpu"]
            rc = self.run_sub(cmd, jd / "train.out", jd / "progress.json", "train", self.train_deadline())
            e["timings"]["train_s"] = e["timings"].get("train_s", 0) + time.time() - t0
            if rc == 3:
                return "interrupted"
            if rc != 0:
                return "failed"
            self.bundler.request(f"{jid} training finished", force_upload=True)
        return self.run_evals(cfg, jd, final_epoch=cfg["epochs"], ckpt_job_dir=jd)

    def ckpt_for(self, job_dir, epoch, final_epoch):
        if epoch == final_epoch:
            return job_dir / "ckpt" / f"full_ep{epoch}.pt"
        return job_dir / "ckpt" / f"model_ep{epoch}.pt"

    def run_evals(self, cfg, jd, final_epoch, ckpt_job_dir):
        jid = cfg["job_id"]
        e = self.jentry(jid)
        for i, ev in enumerate(cfg.get("evals", [])):
            outj = jd / "eval" / f"{ev['tag']}.json"
            if outj.exists():
                continue
            if self.stop.is_set():
                return "interrupted"
            proj = self.eval_s(cfg.get("kind"), cfg.get("head_W", 256), ev.get("S"), ev.get("K"))
            if proj * 1.1 > self.remaining_budget():
                self.log(f"budget: {C.fmt_hms(self.remaining_budget())} left < eval projection {C.fmt_hms(proj * 1.1)}; stopping")
                self.stop_reason = "max_hours"
                return "interrupted"
            ckpt = self.ckpt_for(ckpt_job_dir, ev["epoch"], final_epoch)
            if not ckpt.exists():
                self.log(f"missing checkpoint {ckpt} for eval {ev['tag']}")
                return "failed"
            self.current["phase_label"] = f"eval {ev['tag']}"
            cmd = [PY, "-m", "potential.eval_fm", "--ckpt", ckpt, "--out", outj, "--num_eval", cfg.get("num_eval", self.args.num_eval),
                   "--progress", jd / "eval" / "progress_eval.json"]
            if ev.get("S") is not None:
                cmd += ["--S", ev["S"], "--K", ev["K"], "--mode", ev["mode"]]
            if self.args.allow_cpu:
                cmd += ["--device", "cpu"]
            t0 = time.time()
            rc = self.run_sub(cmd, jd / "eval.out", jd / "eval" / "progress_eval.json", "eval",
                              self.deadline - 60 if self.deadline else None)
            e["timings"]["eval_s"] = e["timings"].get("eval_s", 0) + time.time() - t0
            self.save_manifest()
            if rc != 0:
                return "interrupted" if self.stop.is_set() else "failed"
            r = C.read_json(outj) or {}
            self.log(f"[{jid}] eval {ev['tag']}: success {r.get('success_rate', float('nan')):.1f}% ± "
                     f"{r.get('se') or 0:.1f} (n={r.get('n_episodes')}), {r.get('solve_s_per_env', 0):.2f} s/solve")
            self.after_eval()
        return "done"

    def bridge_dirs(self):
        """--bridge_out sources as local out/ dirs (bundles extracted once under out/bridge_src/)."""
        if self._bridge_dirs is None:
            self._bridge_dirs = [C.materialise_out_dir(s, self.out / "bridge_src") for s in self.args.bridge_out]
        return self._bridge_dirs

    def run_import(self, slot, cfg):
        """Copy a finished bridge-repo job (baseline training+evals, or the sigma calibration) instead of
        re-running it. The bridge job must have the same job id, i.e. the same kind/seed/epochs/lambda.
        Checkpoints are not copied (nothing here needs the baseline weights)."""
        jid = cfg["job_id"]
        jd = self.jobs_dir / jid
        try:
            srcs = [d for d in self.bridge_dirs() if (d / "jobs" / jid / "DONE").exists()]
        except FileNotFoundError as ex:
            self.log(f"[{jid}] import: {ex}")
            return "failed"
        if not srcs:
            self.log(f"[{jid}] import: no finished bridge job '{jid}' in --bridge_out {self.args.bridge_out}. "
                     f"The baseline is never trained in this repo: pass the bridge results that contain it.")
            return "failed"
        src = srcs[0]
        sj = src / "jobs" / jid
        bcfg = C.read_json(sj / "config.json") or {}
        keys = ("type", "kind", "seed", "epochs", "lambda", "num_eval") if cfg["type"] == "train" else ("type", "ref_job", "seed")
        bad = [f"{k}: bridge {bcfg.get(k)!r} vs here {cfg.get(k)!r}" for k in keys if bcfg.get(k) != cfg.get(k)]
        if bad:
            self.log(f"[{jid}] import refused, bridge job differs: {'; '.join(bad)}")
            return "failed"
        if cfg["type"] == "train":
            final = [ev["tag"] for ev in cfg["evals"] if ev["epoch"] == cfg["epochs"]]
            missing = [t for t in final if not (sj / "eval" / f"{t}.json").exists()]
            if missing:
                self.log(f"[{jid}] import refused: bridge job has no final eval {missing}")
                return "failed"
        else:
            calib = C.read_json(src / "calibration.json") or C.read_json(sj / "calibration.json")
            if not calib or calib.get("source_job") != cfg["ref_job"]:
                self.log(f"[{jid}] import refused: bridge calibration.json missing or not from {cfg['ref_job']}")
                return "failed"
            C.write_json_atomic(self.calib_path(), calib)
        shutil.copytree(sj, jd, dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt", "*.pt", "config.json"))
        if (sj / "config.json").exists():
            shutil.copy(sj / "config.json", jd / "bridge_config.json")
        bman = C.read_json(src / "manifest.json") or {}
        C.write_json_atomic(jd / "imported_from.json", {
            "source": str(src), "bridge_out_args": self.args.bridge_out, "bridge_git_commit": bman.get("git_commit"),
            "bridge_gpu": bman.get("gpu"), "bridge_job_status": (bman.get("jobs", {}).get(jid) or {}).get("status"),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        hashes = sorted({(C.read_json(p) or {}).get("eval_set_hash") for p in (jd / "eval").glob("ep*.json")} - {None}) \
            if (jd / "eval").is_dir() else []
        self.log(f"[{jid}] imported from bridge results {src} (bridge git {bman.get('git_commit')}"
                 + (f", eval_set_hash {hashes}" if hashes else "") + ")")
        return "done"

    def run_calibrate(self, slot, cfg):
        jd = self.jobs_dir / cfg["job_id"]
        ref = self.jobs_dir / cfg["ref_job"]
        ckpt = ref / "ckpt" / f"full_ep{self.args.epochs}.pt"
        if not ckpt.exists():
            self.log(f"calibration: missing {ckpt}")
            return "failed"
        cmd = [PY, "-m", "potential.calibrate", "--ckpt", ckpt, "--out", self.calib_path(), "--seed", cfg["seed"],
               "--num_batches", 2 if self.args.debug_steps else cfg["num_batches"]]
        if self.args.num_workers is not None:
            cmd += ["--num_workers", self.args.num_workers]
        if self.args.allow_cpu:
            cmd += ["--device", "cpu"]
        t0 = time.time()
        rc = self.run_sub(cmd, jd / "calibrate.out", None, "calibrate")
        self.jentry(cfg["job_id"])["timings"]["train_s"] = time.time() - t0
        if rc != 0:
            return "failed"
        shutil.copy(self.calib_path(), jd / "calibration.json")
        return "done"

    def run_sweep(self, slot, cfg):
        jd = self.jobs_dir / cfg["job_id"]
        ref = self.jobs_dir / cfg["ref_job"]
        (jd / "eval").mkdir(parents=True, exist_ok=True)
        for ev in cfg["evals"]:  # reuse identical evals already run on the ref job
            src = ref / "eval" / f"{ev['tag']}.json"
            dst = jd / "eval" / f"{ev['tag']}.json"
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        return self.run_evals(cfg, jd, final_epoch=self.args.epochs, ckpt_job_dir=ref)

    # ------------------------------------------------------------------ hooks
    def final_eval(self, jid):
        cfg = self.jentry(jid).get("config") or {}
        E = cfg.get("epochs")
        for ev in cfg.get("evals", []):
            if ev["epoch"] == E and (ev.get("S") is None or (ev["S"] == cfg.get("plan_S") and ev["K"] == 1 and ev["mode"] == "sample")):
                return C.read_json(self.jobs_dir / jid / "eval" / f"{ev['tag']}.json")
        return None

    def after_eval(self):
        ch = self.manifest.setdefault("choices", {})
        ids = self.ids
        # gate
        jobs = self.manifest.get("jobs", {})
        if "gate" not in ch and 1 in self.tiers and jobs.get(ids["base0"], {}).get("status") == "done" and \
                jobs.get(ids["ctrl0"], {}).get("status") == "done":
            b, c = self.final_eval(ids["base0"]), self.final_eval(ids["ctrl0"])
            if b and c:
                thr = max(2 * math.sqrt(b["se"] ** 2 + c["se"] ** 2), 2.0)
                diff = c["success_rate"] - b["success_rate"]
                verdict = "pass" if diff >= -thr else "fail"
                ch["gate"] = {"verdict": verdict, "baseline": b["success_rate"], "baseline_se": b["se"],
                              "control": c["success_rate"], "control_se": c["se"], "diff": diff, "threshold": thr,
                              "rule": "pass iff control >= baseline - max(2*sqrt(se_b^2+se_c^2), 2 points)"}
                self.log(f"GATE {verdict.upper()}: baseline {b['success_rate']:.1f}±{b['se']:.1f} vs P1 sigma=0 S=1 "
                         f"{c['success_rate']:.1f}±{c['se']:.1f} (diff {diff:+.1f}, threshold -{thr:.1f})")
                self.save_manifest()
        # best P1 / P2 (tier 2, seed 0, final success; ties -> lower sigma)
        t2 = ids["t2"]
        if t2 and all(jobs.get(j, {}).get("status") in ("done", "failed") for j in t2):
            for v in ("P1", "P2"):
                key = f"best_{v}"
                if key in ch:
                    continue
                cands = []
                for j in t2:
                    cfg = self.jentry(j).get("config") or {}
                    r = self.final_eval(j) if self.jentry(j).get("status") == "done" else None
                    if cfg.get("variant") == v and r:
                        cands.append((r["success_rate"], -cfg["sigma"], j, cfg["sigma"]))
                if cands:
                    best = max(cands)
                    ch[key] = {"variant": v, "sigma": best[3], "job_id": best[2], "success": best[0],
                               "candidates": {c[2]: c[0] for c in cands}}
                    self.log(f"selected {key}: {best[2]} ({best[0]:.1f}%)")
            if "best_T2" not in ch and ch.get("best_P1") and ch.get("best_P2"):
                a1, a2 = ch["best_P1"], ch["best_P2"]
                pick = max([(a1["success"], -a1["sigma"], 1, a1), (a2["success"], -a2["sigma"], 0, a2)])[3]
                ch["best_T2"] = dict(pick, candidates={a1["job_id"]: a1["success"], a2["job_id"]: a2["success"]})
                self.log(f"selected best_T2: {pick['job_id']}")
            self.save_manifest()

    # ------------------------------------------------------------------ progress
    def progress_line(self):
        slot = self.current.get("slot")
        if slot is None:
            return None
        jid = slot["id"]
        idx, N = self.current["index"], len(self.plan)
        cur_rem = None
        parts = [f"[job {idx}/{N} {jid}]"]
        pf = self.current.get("progress_file")
        pr = C.read_json(pf) if pf else None
        phase = self.current.get("phase")
        if phase == "train" and pr:
            spe = pr.get("steps_per_epoch") or 1
            pct = 100 * pr.get("global_step", 0) / max(pr.get("total_steps", 1), 1)
            parts.append(f"ep {pr.get('epoch', 0) + 1}/{pr.get('epochs')} | step {pr.get('step_in_epoch')}/{spe} "
                         f"({pct:.0f}%) | {pr.get('it_s') or 0:.1f} it/s")
        elif phase == "eval" and pr:
            parts.append(f"{self.current.get('phase_label')} | episodes done {pr.get('episodes_done')}/"
                         f"{pr.get('episodes_total')} | env step {pr.get('env_step')}/50 | planning calls "
                         f"{pr.get('calls')}/{pr.get('expected_calls')} ({100 * (pr.get('frac') or 0):.0f}%) | "
                         f"eval ETA {C.fmt_hms(pr.get('eta_s'))}")
        elif phase:
            parts.append(phase)
        cur_cfg = self.jentry(jid).get("config") or self.representative(slot)
        cur_rem = self.project(slot, cur_cfg)
        parts.append(f"job ETA {C.fmt_hms(cur_rem)}")
        done_s = sum((self.jentry(s["id"]).get("timings", {}).get("train_s", 0) +
                      self.jentry(s["id"]).get("timings", {}).get("eval_s", 0)) for s in self.plan)
        rem = cur_rem + sum(self.project(s) for s in self.plan[idx:]
                            if self.jentry(s["id"]).get("status") not in ("done", "failed", "blocked"))
        overall = 100 * done_s / max(done_s + rem, 1)
        elapsed = time.time() - self.t_start
        parts.append(f"overall {overall:.0f}% | overall ETA {C.fmt_hms(rem)} | elapsed {C.fmt_hms(elapsed)}")
        if self.args.price_per_hour:
            parts.append(f"est. total cost ${(elapsed + rem) / 3600 * self.args.price_per_hour:.2f}")
        C.write_json_atomic(self.out / "progress.json", {
            "job_index": idx, "jobs_total": N, "job_id": jid, "phase": phase, "detail": pr,
            "job_eta_s": cur_rem, "overall_pct": overall, "overall_eta_s": rem, "elapsed_s": elapsed,
            "est_total_cost": (elapsed + rem) / 3600 * self.args.price_per_hour if self.args.price_per_hour else None,
            "time": time.time()})
        return " | ".join(parts)

    def progress_loop(self):
        last = 0
        while not self.stop_progress.is_set():
            time.sleep(2)
            for p in self.jobs_dir.glob("*/milestone_ep*"):
                if p not in self.seen_milestones:
                    self.seen_milestones.add(p)
                    if self.started:
                        self.bundler.request(f"milestone {p.parent.name}/{p.name}", force_upload=True)
            if self.started:
                self.bundler.periodic()
            if time.time() - last >= 30:
                last = time.time()
                try:
                    line = self.progress_line()
                    if line:
                        self.log(line)
                except Exception as e:  # noqa: BLE001
                    self.log(f"(progress error: {e!r})")

    # ------------------------------------------------------------------ plan table
    def print_plan(self):
        rows = []
        total = 0.0
        for i, s in enumerate(self.plan, 1):
            cfg = self.resolve(s)
            st = self.jentry(s["id"]).get("status", "pending")
            proj = 0.0 if st == "done" else self.project(s, cfg)
            total += proj
            desc = _describe(cfg) if cfg else f"{s.get('resolve')} -> {s.get('ref', s.get('resolve'))} (TBD)"
            rows.append(f"{i:>3} | T{s['tier']} | {s['id']:<34} | {desc:<44} | {st:<11} | {C.fmt_hms(proj):>9}")
        self.log("JOB PLAN\n  # | tier | job                                | what                                         | status      | projected\n  " + "\n  ".join(rows))
        src = {t: self.it_s(t)[1] for t in ("baseline", "potential_W256")}
        cost = f" | projected cost ${total / 3600 * self.args.price_per_hour:.2f} at ${self.args.price_per_hour}/h" if self.args.price_per_hour else ""
        self.log(f"projected total {C.fmt_hms(total)} ({total / 3600:.1f} h){cost} | throughput source: {src} | "
                 f"steps/epoch {self.args.debug_steps or self.spe()} | evals {self.args.num_eval} episodes")
        if self.deadline:
            self.log(f"--max_hours {self.args.max_hours}: budget ends {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.deadline))}"
                     f" ({C.fmt_hms(self.remaining_budget())} usable after a {RESERVE_S // 60}-min reserve)")
        return total

    # ------------------------------------------------------------------ benchmark
    def run_benchmark(self):
        self.log("BENCHMARK: ~50 training steps per job type + one short eval per eval type (random-init weights)")
        bdir = self.out / "benchmark"
        # no baseline training benchmark: baseline jobs are imported from the bridge repo, never trained here
        types = {"potential_W256": dict(kind="potential", variant="P2", sigma=1.0, plan_S=4, head_W=256)}
        if 4 in self.tiers:
            types["potential_W128"] = dict(kind="potential", variant="P1", sigma=0.0, plan_S=1, head_W=128)
            types["potential_W512"] = dict(kind="potential", variant="P1", sigma=0.0, plan_S=1, head_W=512)
        res = {"gpu": self.manifest.get("gpu"), "train": {}, "eval": {}, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        calib = bdir / "unit_calibration.json"
        C.write_json_atomic(calib, {"delta_std": [1.0] * 192})
        for name, t in types.items():
            jd = bdir / name
            jd.mkdir(parents=True, exist_ok=True)
            cfg = {"job_id": f"bench_{name}", "tier": 0, "type": "train", "seed": 0, "epochs": self.args.epochs,
                   "lambda": 0.09, "milestones": [], "calibration_path": str(calib), **t}
            C.write_json_atomic(jd / "config.json", cfg)
            cmd = [PY, "-m", "potential.train_fm", "--job_dir", jd, "--benchmark_steps", 50, "--val_batches", 10]
            if self.args.num_workers is not None:
                cmd += ["--num_workers", self.args.num_workers]
            if self.args.allow_cpu:
                cmd += ["--device", "cpu", "--benchmark_steps", 4]
            rc = self.run_sub(cmd, jd / "bench.out", None, f"benchmark train {name}")
            r = C.read_json(jd / "benchmark_train.json")
            if rc == 0 and r:
                res["train"][name] = r
                res["steps_per_epoch"] = r["steps_per_epoch"]
                self.log(f"benchmark train {name}: {r['train_it_s']:.2f} it/s (data wait {100 * r['data_wait_frac']:.0f}%, "
                         f"{self.args.num_workers} workers), {r['steps_per_epoch']} steps/epoch -> "
                         f"{C.fmt_hms(r['steps_per_epoch'] / r['train_it_s'])} per epoch, setup {r.get('setup_s', 0):.0f}s")
                if r["data_wait_frac"] > 0.3:
                    self.log(f"  note: {100 * r['data_wait_frac']:.0f}% of step time is waiting for data (JPEG decode in "
                             f"workers); more CPUs / --num_workers would speed training up without changing results")
        # eval benchmark on random-init checkpoints from preflight
        pre = self.out / "preflight"
        specs = {"baseline": (pre / "pf_baseline.pt", "1x1"),
                 "potential": (pre / "pf_P2.pt", "1x1,2x1,4x1,8x1,4x2,4x4" if 4 in self.tiers else "1x1,4x1")}
        for name, (ck, bench) in specs.items():
            if not ck.exists():
                self.log(f"benchmark eval {name}: no checkpoint {ck} (preflight skipped?)")
                continue
            outj = bdir / f"eval_{name}.json"
            n = 4
            cmd = [PY, "-m", "potential.eval_fm", "--ckpt", ck, "--out", outj, "--num_eval", n, "--benchmark", bench]
            if name == "potential":
                cmd += ["--S", 4]
            if self.args.allow_cpu:
                cmd += ["--device", "cpu"]
            t0 = time.time()
            rc = self.run_sub(cmd, bdir / f"eval_{name}.out", None, f"benchmark eval {name}")
            r = C.read_json(outj)
            if rc == 0 and r:
                proc = time.time() - t0
                res["eval"][name] = {"solve_s": r["benchmark_solve_s"], "env_overhead_s_per_episode": r["env_overhead_s_per_episode"],
                                     "setup_s": max(proc - r["wall_s"], 0.0) + 5.0}
                self.log(f"benchmark eval {name}: solve s/env {r['benchmark_solve_s']} | setup {res['eval'][name]['setup_s']:.0f}s")
        C.write_json_atomic(self.out / "benchmark.json", res)
        self.bench = res

    # ------------------------------------------------------------------ main
    def check_hashes(self):
        bad = []
        for s in self.plan:
            cfg = self.resolve(s)
            e = self.manifest.get("jobs", {}).get(s["id"])
            if not cfg or not e or not e.get("config_hash"):
                continue
            h = self.config_hash(self.full_config(cfg))
            if h != e["config_hash"] and e.get("status") != "pending":
                bad.append((s["id"], e["config_hash"], h))
        if bad and not self.args.force:
            for jid, old, new in bad:
                self.log(f"CONFIG HASH MISMATCH {jid}: manifest {old} vs now {new}")
            self.log("refusing to continue (config differs from the resumed run). Re-run with --force to re-run those jobs.")
            return False
        for jid, _, _ in bad:
            d = self.jobs_dir / jid
            if d.exists():
                d.rename(d.with_name(f"{jid}.stale.{int(time.time())}"))
            self.manifest["jobs"][jid] = {"status": "pending", "timings": {}, "note": "re-run (--force, config changed)"}
        return True

    def check_upload_target(self):
        """Fail fast if results could not leave this (ephemeral) machine."""
        if self.gh.enabled():
            from potential.github_sync import TAG_PREFIX, GitHubError

            try:
                self.gh.check()
            except GitHubError as ex:
                self.log(f"GITHUB UPLOAD CHECK FAILED: {ex}\n  -> create a fine-grained token with 'Contents: Read and write' "
                         f"on {self.gh.repo} and export GITHUB_TOKEN=<token> (or clone with https://<token>@github.com/...).")
                return False
            tag = self.manifest.setdefault("github_release_tag", TAG_PREFIX + time.strftime("%Y%m%d-%H%M%S"))
            self.save_manifest()
            self.log(f"results + checkpoints will be saved to the draft GitHub release '{tag}' of {self.gh.repo} "
                     f"(every 15 min, after each job, and at the end)")
            return True
        if self.bundler.hf_enabled():
            return True
        if self.args.benchmark or self.args.allow_no_upload:
            self.log("WARNING: no GitHub/HF upload configured — results stay on this machine only")
            return True
        self.log("NO UPLOAD TARGET: results would be lost when the pod stops. Set GITHUB_TOKEN (or clone the repo "
                 "with https://<token>@github.com/...), or pass --allow_no_upload to run anyway.")
        return False

    def check_bridge_sources(self):
        """Fail fast: baseline jobs and the calibration are imported from the bridge repo, never run here."""
        pending = [s["id"] for s in self.plan if (s.get("config") or {}).get("imported_from")
                   and self.jentry(s["id"]).get("status") != "done"]
        if not pending or self.args.benchmark:
            return True
        if not self.args.bridge_out:
            self.log(f"NO BRIDGE RESULTS: {pending} are imported from the bridge repo (fm-jepa), never trained here. "
                     f"Pass --bridge_out <bridge out/ dir or bundle.tar.gz> (repeatable).")
            return False
        try:
            dirs = self.bridge_dirs()
        except (FileNotFoundError, OSError) as ex:
            self.log(f"BRIDGE RESULTS UNREADABLE: {ex}")
            return False
        self.log(f"bridge results: {[str(d) for d in dirs]} -> importing {pending}")
        return True

    def run(self):
        a = self.args
        self.init_manifest()
        self.plan, self.ids = build_plan(a, self.tiers, self.seeds)
        self.started = False
        self.stop_progress = threading.Event()
        self.manifest["order"] = [s["id"] for s in self.plan]
        for s in self.plan:
            e = self.jentry(s["id"])
            e["tier"] = s["tier"]
            if e.get("status") in ("running",):
                e["status"] = "interrupted"
        self.save_manifest()
        if not self.check_hashes():
            return 2
        self.log(f"out dir {self.out} | bundle {self.bundle_path} | STABLEWM_HOME {C.stablewm_home()} | "
                 f"git {self.manifest['git_commit']} | gpu {self.manifest['gpu']}")
        self.log(self.extension_note())
        if not self.check_upload_target():
            return 1
        if not self.check_bridge_sources():
            return 1

        if not a.skip_preflight:
            n_train = sum(1 for s in self.plan if (s.get("config") or {}).get("type", "train") == "train")
            cmd = [PY, "-m", "potential.preflight", "--out", self.out, "--min_free_gb", 5 + 0.6 * n_train]
            if a.allow_cpu:
                cmd.append("--allow_cpu")
            rc = self.run_sub(cmd, self.out / "preflight.out", None, "preflight")
            self.log((self.out / "preflight.out").read_text()[-4000:])
            if rc != 0:
                self.log("PRE-FLIGHT FAILED — nothing was trained. See out/preflight.out")
                return 1
        gpu_changed = self.bench and self.bench.get("gpu") != self.manifest.get("gpu")
        if a.benchmark or ((not self.bench or gpu_changed) and not a.no_auto_benchmark):
            self.run_benchmark()
        total = self.print_plan()
        if a.benchmark:
            self.log(f"--benchmark: projected {C.fmt_hms(total)} for tiers {a.tiers} "
                     f"(seeds {a.seeds}, {a.epochs} epochs, {a.num_eval}-episode evals). Exiting.")
            return 0

        self.started = True
        self.seen_milestones = set(self.jobs_dir.glob("*/milestone_ep*"))
        prog = threading.Thread(target=self.progress_loop, daemon=True)
        prog.start()
        status = self.loop()
        self.stop_progress.set()
        return status

    def loop(self):
        for idx, slot in enumerate(self.plan, 1):
            jid = slot["id"]
            e = self.jentry(jid)
            # gate / selections may be computable from jobs finished in an earlier run (resume)
            self.after_eval()
            self.maybe_diagnose()
            if e.get("status") == "done" and (self.jobs_dir / jid / "DONE").exists():
                continue
            if self.stop.is_set():
                break
            ok, why = self.needs_ok(slot)
            cfg = self.resolve(slot) if ok else None
            if not ok or cfg is None:
                e["status"] = "blocked"
                e["note"] = why or "unresolved"
                self.save_manifest()
                self.log(f"[{jid}] skipped: {e['note']}")
                continue
            cfg = self.full_config(cfg)
            if self.args.extend_mode == "continue" and cfg.get("type") == "train":
                self.attach_extension(cfg)
            proj = self.project(slot, cfg)
            if proj * 1.1 > self.remaining_budget():
                self.log(f"--max_hours: {C.fmt_hms(self.remaining_budget())} left < projected {C.fmt_hms(proj * 1.1)} "
                         f"for {jid} (+10%); not launching further jobs")
                self.stop_reason = "max_hours"
                break
            jd = self.jobs_dir / jid
            jd.mkdir(parents=True, exist_ok=True)
            C.write_json_atomic(jd / "config.json", cfg)
            e["config"] = cfg
            e["config_hash"] = self.config_hash(cfg)
            e["status"] = "running"
            e.pop("note", None)
            e["timings"].setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
            e["projected_s"] = proj
            if cfg.get("extended_from"):
                e["extended_from"] = cfg["extended_from"]
            self.save_manifest()
            self.current.update(slot=slot, index=idx)
            self.log(f"=== START [{idx}/{len(self.plan)}] {jid} (tier {slot['tier']}): {_describe(cfg)} | projected {C.fmt_hms(proj)}")
            try:
                if cfg.get("imported_from"):
                    st = self.run_import(slot, cfg)
                elif cfg["type"] == "train":
                    st = self.run_train(slot, cfg)
                elif cfg["type"] == "calibrate":
                    st = self.run_calibrate(slot, cfg)
                else:
                    st = self.run_sweep(slot, cfg)
            except Exception as ex:  # noqa: BLE001
                self.log(f"[{jid}] exception: {ex!r}\n{traceback.format_exc()}")
                st = "failed"
            e["status"] = st
            e["timings"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if st == "done":
                (jd / "DONE").touch()
            self.save_manifest()
            self.log(f"=== END {jid}: {st}")
            self.after_eval()
            self.maybe_diagnose()
            self.regen_report()
            if st == "done":
                self.bundler.request(f"{jid} done", force_upload=True)
            if st == "interrupted" or self.stop.is_set():
                break
        self.current.update(slot=None)
        return 0

    def attach_extension(self, cfg):
        """--extend_mode continue: warm-start from the same config trained for fewer epochs."""
        E = cfg["epochs"]
        for E_old in sorted((m for m in MILESTONES if m < E), reverse=True):
            old = cfg["job_id"].replace(f"_e{E}", f"_e{E_old}")
            od = self.jobs_dir / old
            ck = od / "ckpt" / f"full_ep{E_old}.pt"
            if (od / "DONE").exists() and ck.exists():
                cfg["extend_from_ckpt"] = str(ck)
                cfg["extended_from"] = f"extended_from={E_old} ({old})"
                jd = self.jobs_dir / cfg["job_id"] / "eval"
                jd.mkdir(parents=True, exist_ok=True)
                for ev in cfg["evals"]:
                    if ev["epoch"] <= E_old:
                        src = od / "eval" / f"{ev['tag']}.json"
                        if src.exists() and not (jd / src.name).exists():
                            shutil.copy(src, jd / src.name)
                cfg["evals"] = [ev for ev in cfg["evals"] if ev["epoch"] > E_old or (jd / f"{ev['tag']}.json").exists()]
                return

    def maybe_diagnose(self):
        """The bridge's auto-diagnostic (BatchNorm on the velocity-head output) has no conservative
        counterpart: normalising U or rescaling grad U per dimension breaks v = -grad U. On a gate
        failure the potential repo only records the failure; later tiers stay blocked."""
        g = self.manifest.get("choices", {}).get("gate")
        if not g or g["verdict"] != "fail" or self.args.no_auto_diagnose:
            return
        ch = self.manifest["choices"]
        if not ch.get("diagnosis"):
            ch["diagnosis"] = ("gate failed; later tiers stopped. No automatic diagnostic job (the bridge's out_norm=bn "
                               "fix does not apply to a potential). Candidates to run manually with --tiers: "
                               "lambda (0.03/0.3), head width (128/512), --head_fp32.")
            self.log("GATE FAILED -> later tiers blocked; no auto-diagnostic for the potential variant")
            self.save_manifest()

    def regen_report(self):
        try:
            from potential import report

            report.build_report(self.out, bridge_dirs=self.bridge_dirs())
        except Exception as ex:  # noqa: BLE001
            self.log(f"(report generation failed: {ex!r})")

    def extension_note(self):
        return ("LR schedule: spt LinearWarmupCosineAnnealingLR, warmup 1% of total steps, cosine to 0 at "
                "epochs*steps_per_epoch, stepped every optimizer step -> depends on the epoch budget. "
                f"--extend_mode={self.args.extend_mode}: "
                + ("longer runs are fresh jobs (equivalent to a fresh N-epoch run)." if self.args.extend_mode == "fresh"
                   else "longer runs warm-start from the shorter run's final checkpoint with the schedule rebuilt "
                        "for the new total (NOT equivalent to a fresh run; marked extended_from)."))

    # ------------------------------------------------------------------ signals / finish
    def on_signal(self, signum, frame):
        if self.stop.is_set():
            return
        self.stop.set()
        self.stop_reason = f"signal {signum}"
        self.log(f"received signal {signum}: checkpointing the running job, then bundling")
        ch = self.child
        if ch is not None and ch.poll() is None:
            try:
                ch.send_signal(signal.SIGTERM)
            except Exception:
                pass

    def finish(self, status_note):
        self.log(f"finishing ({status_note}); writing report and final bundle")
        for s in getattr(self, "plan", []):
            e = self.jentry(s["id"])
            if e.get("status") == "running":
                e["status"] = "interrupted"
        self.save_manifest()
        self.regen_report()
        self.bundler.final()
        self.print_retrieval()

    def print_retrieval(self):
        msgs = ["HOW TO GET THE RESULTS OUT:"]
        tag = self.manifest.get("github_release_tag")
        if self.gh.enabled() and tag:
            msgs.append(f"  0. GitHub (private draft release '{tag}'): {self.gh.release_url()}\n"
                        f"     on your machine, inside the repo:  python -m potential.github_sync download --dest out_{tag}\n"
                        f"     (needs GITHUB_TOKEN or a logged-in gh: GITHUB_TOKEN=$(gh auth token) python -m potential.github_sync ...)")
        if self.bundler.hf_enabled():
            msgs.append(f"  1. HF Hub (private): huggingface-cli download {os.environ['HF_REPO']} --local-dir lewm_potential_results "
                        f"(contains out/ and bundle.tar.gz)")
        else:
            msgs.append("  1. HF Hub: not configured (set HF_TOKEN and HF_REPO to enable automatic upload)")
        if shutil.which("runpodctl"):
            msgs.append(f"  2. runpodctl send {self.bundle_path}   (then run the printed 'runpodctl receive ...' locally)")
        else:
            msgs.append("  2. runpodctl: not found on this pod")
        size = self.bundle_path.stat().st_size / 1e9 if self.bundle_path.exists() else 0
        msgs.append(f"  3. Bundle file: {self.bundle_path} ({size:.2f} GB) — download via the RunPod web UI / Jupyter, or "
                    f"scp -P <port> root@<pod-ip>:{self.bundle_path} .")
        msgs.append("  NOTE: if the pod is terminated when the job ends, ONLY the uploaded copy (GitHub release / HF) "
                    "survives. Use --hold_after_done to keep the pod alive until you have downloaded the bundle.")
        msgs.append("  Resume later with: bash run_all.sh --resume bundle.tar.gz [same flags]")
        self.log("\n".join(msgs))


def _describe(cfg):
    from potential.report import describe

    try:
        return describe(cfg)
    except Exception:
        return cfg.get("job_id", "?")


# ============================================================================ bundling / upload
class Bundler:
    def __init__(self, orch):
        self.o = orch
        self.lock = threading.Lock()
        self.dirty = False
        self.thread = None
        self.last_upload = 0.0
        self.upload_lock = threading.Lock()  # bundle worker, 15-min timer and final sync never overlap
        self.timer_thread = None

    def periodic(self):
        """Called by the progress loop: a real 15-minute timer for uploads (checkpoints written since the
        last sync, incl. the 10-min mid-epoch checkpoint, go up even between milestones). No bundle rebuild."""
        if time.time() - self.last_upload < HF_EVERY_S:
            return
        if self.timer_thread is not None and self.timer_thread.is_alive():
            return

        def run():
            try:
                self.upload(force=False)
            except Exception as ex:  # noqa: BLE001
                self.o.log(f"(periodic upload error: {ex!r})")

        self.timer_thread = threading.Thread(target=run, daemon=True)
        self.timer_thread.start()

    def hf_enabled(self):
        return bool(os.environ.get("HF_TOKEN") and os.environ.get("HF_REPO"))

    def request(self, reason, force_upload=False):
        with self.lock:
            self.dirty = True
            self.reason = reason
            self.force_upload = getattr(self, "force_upload", False) or force_upload
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._worker, daemon=True)
                self.thread.start()

    def _worker(self):
        while True:
            with self.lock:
                if not self.dirty:
                    return
                self.dirty = False
                reason = getattr(self, "reason", "")
                force, self.force_upload = getattr(self, "force_upload", False), False
            try:
                self.build(reason)
                self.upload(force=force)
            except Exception as ex:  # noqa: BLE001
                self.o.log(f"(bundle/upload error: {ex!r})")

    def _files(self):
        out = self.o.out
        files, skipped = [], []
        for p in sorted(out.rglob("*")):
            if not p.is_file() or ".tmp" in p.name:
                continue
            if p.parent.name == "preflight" and p.suffix == ".pt":
                continue  # random-init smoke-test weights
            try:
                if p.stat().st_size > ONE_GB:
                    skipped.append(str(p.relative_to(out)))
                    continue
            except OSError:
                continue
            files.append(p)
        return files, skipped

    def build(self, reason=""):
        """tar.gz of out/ (top dir = out's name), written to a temp file then atomically renamed.
        Uses a niced GNU tar | pigz/gzip subprocess (no GIL contention with the progress thread);
        falls back to Python tarfile where GNU tar is unavailable."""
        out = self.o.out
        dst = self.o.bundle_path
        tmp = dst.with_name(dst.name + ".tmp")
        t0 = time.time()
        files, skipped = self._files()
        gnu = "GNU" in subprocess.run(["tar", "--version"], capture_output=True, text=True).stdout if shutil.which("tar") else False
        how = "python"
        if gnu:
            listf = dst.with_name(dst.name + ".list")
            listf.write_text("".join(f"{out.name}/{p.relative_to(out)}\n" for p in files))
            comp = "pigz -1" if shutil.which("pigz") else "gzip -1"
            q = shlex.quote
            cmd = (f"set -o pipefail; nice -n 10 tar -C {q(str(out.parent))} --ignore-failed-read -cf - -T {q(str(listf))} "
                   f"| nice -n 10 {comp} > {q(str(tmp))}")
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            listf.unlink(missing_ok=True)
            # tar exit 1 = "file changed as we read it" (logs growing): the archive is still valid
            if r.returncode in (0, 1) and tmp.exists() and tmp.stat().st_size > 0:
                how = comp.split()[0]
            else:
                self.o.log(f"(tar bundling failed rc={r.returncode}: {r.stderr[-500:]}; falling back to python tarfile)")
                gnu = False
        if not gnu:
            with tarfile.open(tmp, "w:gz", compresslevel=1) as tar:
                for p in files:
                    try:
                        tar.add(p, arcname=str(Path(out.name) / p.relative_to(out)))
                    except (FileNotFoundError, OSError):
                        continue  # rotated away while bundling
        os.replace(tmp, dst)
        size = dst.stat().st_size / 1e9
        self.o.log(f"bundle rebuilt ({reason}): {dst} {size:.2f} GB in {time.time() - t0:.0f}s [{how}]"
                   + (f" | skipped >1GB: {skipped}" if skipped else ""))

    def upload(self, force=False, include_bundle=False):
        with self.upload_lock:
            self._upload(force, include_bundle)

    def _upload(self, force=False, include_bundle=False):
        if not force and time.time() - self.last_upload < HF_EVERY_S:
            return
        gh = getattr(self.o, "gh", None)
        if gh is not None and gh.enabled() and self.o.manifest.get("github_release_tag"):
            from potential.github_sync import GitHubError

            try:
                summary = gh.sync(self.o.manifest["github_release_tag"], final=include_bundle)
                self.o.log(f"saved to GitHub release {gh.release_url()} ({summary})")
            except (GitHubError, OSError) as ex:
                self.o.log(f"(GitHub upload failed, will retry at the next sync: {ex})")
            self.last_upload = time.time()
        if not self.hf_enabled():
            return
        from huggingface_hub import HfApi

        repo = os.environ["HF_REPO"]
        api = HfApi(token=os.environ["HF_TOKEN"])
        t0 = time.time()
        api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
        big = [str(p.relative_to(self.o.out)) for p in self.o.out.rglob("*") if p.is_file() and p.stat().st_size > ONE_GB]
        api.upload_folder(folder_path=str(self.o.out), path_in_repo="out", repo_id=repo, repo_type="model",
                          ignore_patterns=["*.tmp*", *big], delete_patterns=["**"],
                          commit_message=f"out/ sync {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if include_bundle and self.o.bundle_path.exists():
            api.upload_file(path_or_fileobj=str(self.o.bundle_path), path_in_repo="bundle.tar.gz", repo_id=repo,
                            repo_type="model", commit_message="final bundle")
        try:
            api.super_squash_history(repo_id=repo, repo_type="model")  # keep LFS storage ~ current size
        except Exception:
            pass
        self.last_upload = time.time()
        self.o.log(f"uploaded to HF {repo} ({'with bundle, ' if include_bundle else ''}{time.time() - t0:.0f}s)")

    def final(self):
        for t in (self.thread, self.timer_thread):
            if t is not None:
                t.join(timeout=1800)
        try:
            self.build("final")
        except Exception as ex:  # noqa: BLE001
            self.o.log(f"(final bundle failed: {ex!r})")
        try:
            self.upload(force=True, include_bundle=True)
        except Exception as ex:  # noqa: BLE001
            self.o.log(f"(final upload failed: {ex!r})")


# ============================================================================ entry
def main():
    args = parse()
    orch = Orchestrator(args)
    signal.signal(signal.SIGTERM, orch.on_signal)
    signal.signal(signal.SIGINT, orch.on_signal)
    rc = 0
    note = "completed"
    try:
        rc = orch.run()
        if orch.stop_reason:
            note = f"stopped: {orch.stop_reason}"
    except Exception as ex:  # noqa: BLE001
        orch.log(f"UNHANDLED EXCEPTION: {ex!r}\n{traceback.format_exc()}")
        note = f"crashed: {ex!r}"
        rc = 1
    if args.benchmark and rc == 0:
        return 0
    orch.finish(note)
    if args.hold_after_done:
        orch.log("--hold_after_done: keeping the process alive. Download the bundle, then stop the pod.")
        while True:
            time.sleep(600)
            orch.print_retrieval()
    return rc


if __name__ == "__main__":
    sys.exit(main())
