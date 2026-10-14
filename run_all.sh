#!/usr/bin/env bash
# One-shot, unattended entry point: setup.sh, then the job plan (potential/orchestrate.py).
# Re-launches itself under nohup (log: $OUT_DIR/run.log) so an SSH disconnect does not kill it.
#
#   bash run_all.sh [--tiers 1,2,3] [--epochs 20] [--seeds 0,1] [--max_hours H] [--price_per_hour P]
#                   [--benchmark] [--hold_after_done] [--resume bundle.tar.gz|dir|github] [--force]
#                   [--num_eval 200] [--extend_mode fresh|continue] [--sweep_grid] [--num_workers N]
#                   --bridge_out <bridge out dir or bundle.tar.gz> (repeatable; source of the baseline + calibration)
#
# Env: GITHUB_TOKEN (results -> GitHub release; or clone with https://<token>@github.com/...),
#      REPO_URL (clone first), HF_TOKEN + HF_REPO (optional HF upload), STABLEWM_HOME, OUT_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# WORKDIR = where venv/datasets/outputs live (~90 GB). On a shared SSH GPU box set it to a big disk, e.g.
# export WORKDIR=/scratch/$USER/lewm   (default: /workspace on RunPod, else $HOME)
if [[ -n "${WORKDIR:-}" ]]; then WS="$WORKDIR"; mkdir -p "$WS"; elif [[ -d /workspace ]]; then WS=/workspace; else WS="$HOME"; fi

if [[ -n "${REPO_URL:-}" && ! -f "$HERE/potential/orchestrate.py" ]]; then
  export OUT_DIR="${OUT_DIR:-$WS/pot_out}"      # repo not cloned yet: keep outputs outside the clone target
else
  export OUT_DIR="${OUT_DIR:-$HERE/out}"
fi
mkdir -p "$OUT_DIR"

ARGS=(); RESUME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME="$2"; shift 2 ;;
    --resume=*) RESUME="${1#*=}"; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ -n "$RESUME" && "$RESUME" != "github" ]]; then
  RESUME="$(cd "$(dirname "$RESUME")" && pwd)/$(basename "$RESUME")"
  [[ -e "$RESUME" ]] || { echo "resume source not found: $RESUME" >&2; exit 1; }
fi

# ------------------------------------------------------------------ foreground: relaunch under nohup
if [[ "${POT_NOHUP:-}" != "1" ]]; then
  export POT_NOHUP=1 POT_RUN_START="$(date +%s)" POT_STDOUT_IS_RUNLOG=1
  RA=(); [[ -n "$RESUME" ]] && RA=(--resume "$RESUME")
  # setsid: own session/process group, so Ctrl-C / closing the terminal never reaches the job
  DETACH=(nohup); command -v setsid >/dev/null && DETACH=(setsid nohup)
  "${DETACH[@]}" bash "$HERE/run_all.sh" "${RA[@]}" "${ARGS[@]+"${ARGS[@]}"}" >> "$OUT_DIR/run.log" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$OUT_DIR/run.pid"
  echo "started (pid $PID) under nohup; log: $OUT_DIR/run.log"
  echo "follow:  tail -f $OUT_DIR/run.log      stop cleanly:  kill -TERM $PID   (checkpoints + bundles)"
  if [[ -t 1 ]]; then
    echo "(following the log; Ctrl-C stops following, not the job)"
    tail -n +1 -f --pid="$PID" "$OUT_DIR/run.log" || true
  fi
  exit 0
fi

# ------------------------------------------------------------------ background part
echo "================ run_all.sh $(date '+%Y-%m-%d %H:%M:%S') args: ${ARGS[*]-} ${RESUME:+--resume $RESUME}"
trap 'echo "[run_all] signal received before the orchestrator started"; exit 143' TERM INT
bash "$HERE/setup.sh" || { echo "[run_all] setup.sh failed — nothing was run"; exit 1; }
# shellcheck disable=SC1091
source "$HOME/.pot_env"
trap - TERM INT

if [[ "$RESUME" == "github" ]]; then
  echo "[run_all] restoring state from the latest potential-results GitHub release into $OUT_DIR"
  (cd "$REPO_DIR" && "$VENV/bin/python" -m potential.github_sync download --dest "$OUT_DIR") || { echo "[run_all] GitHub restore failed"; exit 1; }
elif [[ -n "$RESUME" ]]; then
  echo "[run_all] restoring state from $RESUME into $OUT_DIR"
  TMP="$(mktemp -d "$WS/pot_resume.XXXX")"
  if [[ -d "$RESUME" ]]; then
    BASE="$RESUME"
  else
    tar -xzf "$RESUME" -C "$TMP"; BASE="$TMP"
  fi
  # the bundle's top-level dir is the out dir's name (out/ or pot_out/); accept the dir itself too
  MF="$(find "$BASE" -maxdepth 3 -name manifest.json -not -path '*/jobs/*' | head -1)"
  [[ -n "$MF" ]] || { echo "[run_all] no manifest.json in $RESUME"; exit 1; }
  SRC="$(dirname "$MF")"
  [[ -f "$SRC/run.log" ]] && cp "$SRC/run.log" "$OUT_DIR/run.log.previous"
  rsync -a --exclude run.log --exclude run.pid "$SRC/" "$OUT_DIR/" 2>/dev/null || \
    (cd "$SRC" && tar -cf - --exclude=./run.log --exclude=./run.pid . | (cd "$OUT_DIR" && tar -xf -))
  rm -rf "$TMP"
fi

cd "$REPO_DIR"
exec "$VENV/bin/python" -m potential.orchestrate --out "$OUT_DIR" "${ARGS[@]+"${ARGS[@]}"}"
