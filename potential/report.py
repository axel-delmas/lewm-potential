"""Regenerate out/results.csv, out/report.md and out/plots/* from the job directories.

With bridge results (--bridge_out, github.com/axel-delmas/fm-jepa) it also writes
out/joined_results.csv: every potential eval next to the bridge eval with the same settings
(P1<->A1, P2<->A2, same sigma multiple / lambda / head width / seed / epochs / S / K / mode),
plus the imported baseline, paired per episode on the shared eval start states.

Usage: python -m potential.report [out] [--bridge_out DIR_OR_BUNDLE ...]
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from potential import common as C

NOTES = Path(__file__).with_name("REPORT_NOTES.md")


def _f(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def _g(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{nd}g}"


def load_metrics(job_dir):
    p = Path(job_dir) / "metrics.csv"
    if not p.exists():
        return []
    with open(p) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()})
    return out


def load_evals(job_dir):
    res = []
    for p in sorted((Path(job_dir) / "eval").glob("*.json")):
        if p.name.startswith("progress"):
            continue
        r = C.read_json(p)
        if r and "success_rate" in r and not r.get("smoke"):
            r["tag"] = p.stem
            res.append(r)
    return res


def describe(cfg):
    imp = " (imported from bridge)" if cfg.get("imported_from") else ""
    if cfg.get("type") == "calibrate":
        return "sigma calibration" + imp
    if cfg.get("type") == "eval_sweep":
        return f"planning sweep on {cfg.get('ref_job')}"
    if cfg["kind"] == "baseline":
        d = "baseline LeWM" + imp
    else:
        d = f"{cfg['variant']} sigma={cfg['sigma']}xstd S={cfg['plan_S']}"
        if int(cfg.get("head_W", 256)) != 256:
            d += f" W={cfg['head_W']}"
        if cfg.get("head_out_norm"):
            d += f" out_norm={cfg['head_out_norm']}"
    if abs(float(cfg.get("lambda", 0.09)) - 0.09) > 1e-12:
        d += f" lambda={cfg['lambda']}"
    return d + f" seed{cfg['seed']}"


def collect(out_dir):
    out_dir = Path(out_dir)
    man = C.read_json(out_dir / "manifest.json", {}) or {}
    jobs = man.get("jobs", {})
    order = man.get("order", list(jobs))
    rows = []
    per_job = {}
    for jid in order:
        j = jobs.get(jid)
        if not j or not j.get("config"):
            continue
        cfg = j["config"]
        jd = out_dir / "jobs" / jid
        params = C.read_json(jd / "params.json", {}) or {}
        mets = load_metrics(jd)
        evs = load_evals(jd)
        per_job[jid] = dict(cfg=cfg, status=j.get("status"), metrics=mets, evals=evs, params=params, entry=j)
        for e in evs:
            rows.append({
                "job_id": jid, "tier": cfg.get("tier"), "status": j.get("status"),
                "kind": cfg.get("kind"), "variant": cfg.get("variant") or "baseline",
                "sigma_mult": cfg.get("sigma", 0.0), "lambda": cfg.get("lambda"), "head_W": cfg.get("head_W"),
                "seed": cfg.get("seed"), "epochs": cfg.get("epochs"), "eval_epoch": e.get("epoch"),
                "S": e.get("S"), "K": e.get("K"), "mode": e.get("mode"), "n_episodes": e["n_episodes"],
                "success_rate": e["success_rate"], "se": e["se"], "solve_s_per_env": e.get("solve_s_per_env"),
                "params_total": params.get("total"), "eval_set_hash": e.get("eval_set_hash"), "tag": e["tag"],
            })
    return man, per_job, rows


def write_results_csv(out_dir, rows):
    p = Path(out_dir) / "results.csv"
    keys = ["job_id", "tier", "status", "kind", "variant", "sigma_mult", "lambda", "head_W", "seed", "epochs",
            "eval_epoch", "S", "K", "mode", "n_episodes", "success_rate", "se", "solve_s_per_env", "params_total",
            "eval_set_hash", "tag"]
    tmp = p.with_name("results.csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    tmp.replace(p)


def default_eval(info, epoch):
    """The eval at `epoch` with the job's default planning settings (S=plan_S, K=1, sample)."""
    cfg = info["cfg"]
    for e in info["evals"]:
        if e.get("epoch") != epoch:
            continue
        if cfg.get("kind") == "baseline" or (e.get("S") == cfg.get("plan_S") and e.get("K") == 1 and e.get("mode") == "sample"):
            return e
    return None


def plots(out_dir, per_job):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    pdir = Path(out_dir) / "plots"
    pdir.mkdir(exist_ok=True)
    made = []
    train_jobs = {k: v for k, v in per_job.items() if v["cfg"].get("type", "train") == "train" and v["metrics"]}
    curves = [("train_loss", "train loss (total)"), ("train_pred_loss", "train pred / flow loss"),
              ("val_onestep_mse", "val one-step MSE (sampled, plan S)"), ("val_sigreg", "val SIGReg"),
              ("z_std_mean", "z per-dim std (mean)"), ("eff_rank", "effective rank of z"),
              ("straightness", "temporal straightness (Eq. 9)")]
    if train_jobs:
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        for ax, (key, title) in zip(axes.flat, curves):
            for jid, info in train_jobs.items():
                ep = [m["epoch"] for m in info["metrics"]]
                ys = [m.get(key, float("nan")) for m in info["metrics"]]
                ax.plot(ep, ys, marker=".", label=jid)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        ax = axes.flat[-1]
        for jid, info in train_jobs.items():
            if info["cfg"].get("kind") != "potential":
                continue
            last = info["metrics"][-1]
            ax.plot(range(5), [last.get(f"val_flow_tau{k}", float("nan")) for k in range(5)], marker="o", label=jid)
        ax.set_title("val flow loss by tau bin (last epoch)")
        ax.set_xticks(range(5), ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"])
        ax.grid(alpha=0.3)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(pdir / "training_curves.png", dpi=110)
        plt.close(fig)
        made.append("plots/training_curves.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    any_line = False
    for jid, info in per_job.items():
        if info["cfg"].get("type", "train") != "train" or str(info["cfg"].get("tier")) not in ("1", "2", "3", "D"):
            continue
        pts = []
        for ep in sorted({e.get("epoch") for e in info["evals"] if e.get("epoch") is not None}):
            e = default_eval(info, ep)
            if e:
                pts.append((ep, e["success_rate"], e["se"]))
        if pts:
            any_line = True
            x, y, s = zip(*pts)
            ax.errorbar(x, y, yerr=s, marker="o", capsize=3, label=jid)
    if any_line:
        ax.set_xlabel("epoch")
        ax.set_ylabel("PushT success rate (%) ± per-episode SE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(pdir / "success_vs_epoch.png", dpi=110)
        made.append("plots/success_vs_epoch.png")
    plt.close(fig)
    return made


def build_report(out_dir, bridge_dirs=()):
    out_dir = Path(out_dir)
    man, per_job, rows = collect(out_dir)
    write_results_csv(out_dir, rows)
    made = plots(out_dir, per_job)
    choices = man.get("choices", {})
    L = []
    L.append("# Potential-field predictor (v = −∇U) for LeWM — PushT phase\n")
    L.append(f"Generated {C.time.strftime('%Y-%m-%d %H:%M:%S')} · git `{man.get('git_commit', '?')}` · "
             f"GPU `{man.get('gpu', '?')}` · run args `{man.get('args', {})}`\n")

    # --- job status
    L.append("## Job status\n")
    L.append("| # | job | tier | what | status | train | evals | params |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, jid in enumerate(man.get("order", []), 1):
        j = man["jobs"].get(jid, {})
        cfg = j.get("config") or {}
        t = j.get("timings", {})
        pj = per_job.get(jid, {})
        L.append(f"| {i} | `{jid}` | {j.get('tier', '')} | {describe(cfg) if cfg else j.get('note', '')} | "
                 f"{j.get('status')}{' (' + j['extended_from'] + ')' if j.get('extended_from') else ''} | "
                 f"{C.fmt_hms(t.get('train_s'))} | {C.fmt_hms(t.get('eval_s'))} | "
                 f"{(pj.get('params') or {}).get('total', '–')} |")

    # --- main results table (default planning settings)
    base_solve = None
    for jid, info in per_job.items():
        if info["cfg"].get("kind") == "baseline":
            for e in info["evals"]:
                base_solve = base_solve or e.get("solve_s_per_env")
    L.append("\n## Results (default planning settings; success % ± per-episode SE, n episodes)\n")
    L.append("Planning time is the mean CEM solve per environment (300 samples × 30 iterations), "
             "measured on this GPU; ratio is relative to the locally measured baseline.\n")
    L.append("| job | what | ep5 | ep10 | ep20 | final | n | plan s/solve (×base) | params |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for jid, info in per_job.items():
        cfg = info["cfg"]
        if cfg.get("type", "train") != "train":
            continue
        cells = []
        n = None
        for ep in (5, 10, 20, int(cfg["epochs"])):
            e = default_eval(info, ep)
            cells.append(f"{_f(e['success_rate'])} ± {_f(e['se'])}" if e else "–")
            n = e["n_episodes"] if e else n
        e = default_eval(info, int(cfg["epochs"]))
        ps = e.get("solve_s_per_env") if e else None
        ratio = f" ({ps / base_solve:.2f}×)" if (ps and base_solve) else ""
        L.append(f"| `{jid}` | {describe(cfg)} | {' | '.join(cells)} | {n or '–'} | {_f(ps, 2)}{ratio} | "
                 f"{(info['params'] or {}).get('total', '–')} |")

    # --- seed aggregation
    groups = {}
    for jid, info in per_job.items():
        cfg = info["cfg"]
        if cfg.get("type", "train") != "train":
            continue
        key = describe(cfg).rsplit(" seed", 1)[0]
        e = default_eval(info, int(cfg["epochs"]))
        if e:
            groups.setdefault(key, []).append((cfg["seed"], e["success_rate"], e["se"], e["n_episodes"]))
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        L.append("\n## Per-seed results and seed mean (final epoch)\n")
        L.append("n=2 seeds: no significance claims. 'spread' = |seed0 − seed1|.\n")
        L.append("| config | per seed | mean | spread | max per-episode SE |")
        L.append("|---|---|---|---|---|")
        for k, v in multi.items():
            vals = [x[1] for x in v]
            L.append(f"| {k} | {', '.join(f'seed{s}: {_f(r)}' for s, r, _, _ in v)} | {_f(np.mean(vals))} | "
                     f"{_f(max(vals) - min(vals))} | {_f(max(x[2] for x in v))} |")

    # --- gate, choices
    L.append("\n## Gate and selections\n")
    g = choices.get("gate")
    if g:
        L.append(f"- **Gate (P1 σ=0, S=1 vs baseline, seed 0, final epoch):** {g.get('verdict')} — "
                 f"baseline {_f(g.get('baseline'))} ± {_f(g.get('baseline_se'))}, control {_f(g.get('control'))} ± "
                 f"{_f(g.get('control_se'))}, diff {_f(g.get('diff'))}, threshold ±{_f(g.get('threshold'))} "
                 f"(max(2 × combined SE, 2 points)).")
    elif 1 not in [int(t) for t in str((man.get("args") or {}).get("tiers", "1")).split(",") if t.strip().isdigit()]:
        L.append("- Gate: not run (tier 1 skipped: no sigma=0 control; the seed-0 baseline and the sigma "
                 "calibration were imported as prerequisites).")
    else:
        L.append("- Gate: not evaluated yet.")
    for k in ("best_P1", "best_P2", "best_T2"):
        c = choices.get(k)
        if c:
            L.append(f"- **{k}:** {c.get('variant')} σ={c.get('sigma')}×std from `{c.get('job_id')}` "
                     f"({_f(c.get('success'))}%; candidates: {c.get('candidates')})")
    if choices.get("diagnosis"):
        L.append(f"- **Gate diagnosis:** {choices['diagnosis']}")

    # --- planning sweeps
    sweeps = [(jid, info) for jid, info in per_job.items() if info["cfg"].get("type") == "eval_sweep"]
    if sweeps:
        L.append("\n## Planning-time sweeps (final checkpoint, seed 0, no retraining)\n")
        L.append("| ref job | S | K | mode | success ± SE | n | s/solve |")
        L.append("|---|---|---|---|---|---|---|")
        for jid, info in sweeps:
            for e in sorted(info["evals"], key=lambda e: (e.get("mode") or "", e.get("K") or 0, e.get("S") or 0)):
                L.append(f"| `{info['cfg'].get('ref_job')}` | {e.get('S')} | {e.get('K')} | {e.get('mode')} | "
                         f"{_f(e['success_rate'])} ± {_f(e['se'])} | {e['n_episodes']} | {_f(e.get('solve_s_per_env'), 2)} |")

    # --- training diagnostics at the last epoch
    L.append("\n## Latent diagnostics at the last trained epoch (val subset)\n")
    L.append("| job | epoch | val pred/flow loss | one-step MSE | SIGReg | z std (mean/min) | eff. rank | straightness | flow loss by τ bin |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for jid, info in per_job.items():
        if not info["metrics"]:
            continue
        m = info["metrics"][-1]
        taus = " / ".join(_g(m.get(f"val_flow_tau{k}")) for k in range(5)) if info["cfg"].get("kind") == "potential" else "–"
        L.append(f"| `{jid}` | {int(m['epoch'])} | {_g(m.get('val_pred_loss'))} | {_g(m.get('val_onestep_mse'))} | "
                 f"{_g(m.get('val_sigreg'))} | {_g(m.get('z_std_mean'))} / {_g(m.get('z_std_min'))} | "
                 f"{_f(m.get('eff_rank'), 1)} | {_f(m.get('straightness'), 3)} | {taus} |")

    # --- extension rule
    L.append("\n## Extension rule check (epoch 10 → 20)\n")
    L.append("Flagged 'improving' if the final-setting success rises by more than one per-episode SE, or any "
             "val flow-loss τ bin (flow) / val pred loss (baseline) drops by more than 2%.\n")
    any_ext = False
    for jid, info in per_job.items():
        cfg = info["cfg"]
        if cfg.get("type", "train") != "train":
            continue
        e10, e20 = default_eval(info, 10), default_eval(info, 20)
        m = {int(r["epoch"]): r for r in info["metrics"]}
        if 10 not in m or 20 not in m:
            continue
        any_ext = True
        flags = []
        if e10 and e20 and e20["success_rate"] - e10["success_rate"] > max(e20["se"], e10["se"]):
            flags.append(f"success {_f(e10['success_rate'])}→{_f(e20['success_rate'])}")
        keys = [f"val_flow_tau{k}" for k in range(5)] if cfg.get("kind") == "potential" else ["val_pred_loss"]
        for k in keys:
            a, b = m[10].get(k), m[20].get(k)
            if a and b and math.isfinite(a) and math.isfinite(b) and b < 0.98 * a:
                flags.append(f"{k} {a:.4f}→{b:.4f}")
        L.append(f"- `{jid}`: {'**still improving** — ' + '; '.join(flags) if flags else 'flat'}")
    if not any_ext:
        L.append("- (needs epoch-10 and epoch-20 results)")

    if bridge_dirs:
        L += bridge_section(out_dir, per_job, bridge_dirs)

    if made:
        L.append("\n## Plots\n")
        for m in made:
            L.append(f"![{m}]({m})")
    if NOTES.exists():
        L.append("\n" + NOTES.read_text())
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    return out_dir / "report.md"


# ----------------------------------------------------------------------------- join with the bridge repo
BRIDGE_VARIANT = {"P1": "A1", "P2": "A2"}


def _key(cfg, e, variant):
    return (variant, float(cfg.get("sigma", 0.0) or 0.0), float(cfg.get("lambda", 0.09)), int(cfg.get("head_W") or 256),
            cfg.get("seed"), cfg.get("epochs"), e.get("epoch"), e.get("S"), e.get("K"), e.get("mode"))


def _episodes(e):
    return {(int(i), int(s)): int(v) for i, s, v in zip(e.get("eval_episode_idx", []), e.get("eval_start_idx", []),
                                                         e.get("episode_successes", []))}


def _mcnemar(b, c):
    """Exact two-sided McNemar p-value on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(e_ref, e_new):
    """Per-episode pairing on the shared (episode, start) states. None if the eval sets differ."""
    if e_ref.get("eval_set_hash") != e_new.get("eval_set_hash"):
        return None
    a, b = _episodes(e_ref), _episodes(e_new)
    if set(a) != set(b):
        return None
    only_ref = sum(1 for k in a if a[k] and not b[k])
    only_new = sum(1 for k in a if b[k] and not a[k])
    return {"n": len(a), "only_ref": only_ref, "only_new": only_new, "p": _mcnemar(only_ref, only_new)}


def joined_rows(out_dir, per_job, bridge_dirs):
    bridge = {}  # key -> (job_id, eval, cfg, source)
    gpus = {}
    for bd in bridge_dirs:
        bman, bjobs, _ = collect(bd)
        for jid, info in bjobs.items():
            cfg = info["cfg"]
            if cfg.get("type", "train") != "train" or cfg.get("head_out_norm"):
                continue
            v = cfg.get("variant") or "baseline"
            for e in info["evals"]:
                bridge.setdefault(_key(cfg, e, v), (jid, e, cfg, str(bd)))
                gpus.setdefault(str(bd), bman.get("gpu"))
    base = {}
    for jid, info in per_job.items():
        if info["cfg"].get("kind") == "baseline":
            for e in info["evals"]:
                base[(info["cfg"].get("seed"), e.get("epoch"))] = (jid, e)
    rows = []
    for jid, info in per_job.items():
        cfg = info["cfg"]
        if cfg.get("kind") != "potential" or cfg.get("type", "train") != "train":
            continue
        for e in info["evals"]:
            row = {"potential_job": jid, "variant": cfg.get("variant"), "sigma_mult": cfg.get("sigma"),
                   "lambda": cfg.get("lambda"), "head_W": cfg.get("head_W"), "seed": cfg.get("seed"),
                   "epochs": cfg.get("epochs"), "eval_epoch": e.get("epoch"), "S": e.get("S"), "K": e.get("K"),
                   "mode": e.get("mode"), "n": e["n_episodes"], "potential_success": e["success_rate"],
                   "potential_se": e["se"], "potential_solve_s": e.get("solve_s_per_env"),
                   "potential_gpu": e.get("gpu"), "eval_set_hash": e.get("eval_set_hash")}
            hit = bridge.get(_key(cfg, e, BRIDGE_VARIANT.get(cfg.get("variant"))))
            if hit:
                bj, be, _, src = hit
                pr = paired(be, e)
                row.update(bridge_job=bj, bridge_source=src, bridge_success=be["success_rate"], bridge_se=be["se"],
                           bridge_solve_s=be.get("solve_s_per_env"), bridge_gpu=be.get("gpu"),
                           diff_vs_bridge=e["success_rate"] - be["success_rate"],
                           same_eval_set_bridge=pr is not None)
                if pr:
                    row.update(only_bridge=pr["only_ref"], only_potential=pr["only_new"], mcnemar_p_bridge=pr["p"])
            bh = base.get((cfg.get("seed"), e.get("epoch")))
            if bh:
                pr = paired(bh[1], e)
                row.update(baseline_job=bh[0], baseline_success=bh[1]["success_rate"], baseline_se=bh[1]["se"],
                           diff_vs_baseline=e["success_rate"] - bh[1]["success_rate"],
                           same_eval_set_baseline=pr is not None)
                if pr:
                    row.update(mcnemar_p_baseline=pr["p"])
            rows.append(row)
    return rows


JOIN_KEYS = ["potential_job", "variant", "sigma_mult", "lambda", "head_W", "seed", "epochs", "eval_epoch", "S", "K",
             "mode", "n", "potential_success", "potential_se", "bridge_job", "bridge_success", "bridge_se",
             "diff_vs_bridge", "only_bridge", "only_potential", "mcnemar_p_bridge", "same_eval_set_bridge",
             "baseline_job", "baseline_success", "baseline_se", "diff_vs_baseline", "mcnemar_p_baseline",
             "same_eval_set_baseline", "eval_set_hash", "potential_solve_s", "bridge_solve_s", "potential_gpu",
             "bridge_gpu", "bridge_source"]


def bridge_section(out_dir, per_job, bridge_dirs):
    rows = joined_rows(out_dir, per_job, bridge_dirs)
    p = Path(out_dir) / "joined_results.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOIN_KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in JOIN_KEYS})
    L = ["\n## Comparison with the bridge variant (fm-jepa)\n",
         "Each potential eval is matched to the bridge eval with the same settings (P1↔A1, P2↔A2; same σ multiple, "
         "λ, head width, seed, epochs, S, K, mode). Both are evaluated on the same 200 start states "
         "(`eval_set_hash` checked), so the comparison is paired per episode: 'only bridge / only potential' count "
         "episodes solved by one model and not the other; p is an exact two-sided McNemar test. Planning times "
         "are only comparable when both GPUs are the same. Full table: `joined_results.csv`.\n",
         "| potential job | eval | potential | bridge (job) | Δ vs bridge | only bridge / only potential | p | "
         "baseline | Δ vs baseline | p | s/solve pot / bridge |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ev = f"ep{r['eval_epoch']} S{r['S']} K{r['K']} {r['mode']}"
        br = (f"{_f(r['bridge_success'])} ± {_f(r['bridge_se'])} (`{r['bridge_job']}`)" if r.get("bridge_job")
              else "– (no matching bridge eval)")
        if r.get("bridge_job") and not r.get("same_eval_set_bridge"):
            br += " **eval set differs**"
        L.append(f"| `{r['potential_job']}` | {ev} | {_f(r['potential_success'])} ± {_f(r['potential_se'])} | {br} | "
                 f"{_f(r.get('diff_vs_bridge'))} | {r.get('only_bridge', '–')} / {r.get('only_potential', '–')} | "
                 f"{_g(r.get('mcnemar_p_bridge'), 2)} | {_f(r.get('baseline_success'))} | "
                 f"{_f(r.get('diff_vs_baseline'))} | {_g(r.get('mcnemar_p_baseline'), 2)} | "
                 f"{_f(r.get('potential_solve_s'), 2)} / {_f(r.get('bridge_solve_s'), 2)} |")
    if not rows:
        L.append("| (no potential evals yet) |||||||||||")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="out")
    ap.add_argument("--bridge_out", action="append", default=[], help="bridge out/ dir or bundle .tar.gz (repeatable)")
    args = ap.parse_args()
    dirs = [C.materialise_out_dir(s, Path(args.out) / "bridge_src") for s in args.bridge_out]
    print(build_report(args.out, bridge_dirs=dirs))


if __name__ == "__main__":
    main()
