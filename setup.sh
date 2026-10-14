#!/usr/bin/env bash
# Idempotent setup for the PushT potential-field study on an (ephemeral) RunPod GPU pod.
#  - optional: clone $REPO_URL (else use the repo this script lives in)
#  - checks GPU (nvidia-smi), free disk, network
#  - Python 3.10 venv (uv) with torch matching the driver + pinned stable-worldmodel / stable-pretraining
#  - downloads the two PushT datasets into $STABLEWM_HOME/datasets/ (as-is, no conversion):
#      pusht_expert_train.lance  <- HF dataset galilai-group/lewm-pusht  (read by train.py / potential.train_fm)
#      pusht_expert_train.h5     <- HF dataset quentinll/lewm-pusht, file pusht_expert_train.h5.zst
#                                   (a raw zstd stream of the .h5, decompressed with zstd -d; read by eval.py)
#  - writes $OUT_DIR/env/{pip_freeze.txt,setup.json} and ~/.pot_env for run_all.sh
set -euo pipefail

log() { echo "[setup $(date +%H:%M:%S)] $*"; }
die() { echo "[setup] ERROR: $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# WORKDIR = where venv/datasets/outputs live (~90 GB). On a shared SSH GPU box set it to a big disk, e.g.
# export WORKDIR=/scratch/$USER/lewm   (default: /workspace on RunPod, else $HOME)
if [[ -n "${WORKDIR:-}" ]]; then WS="$WORKDIR"; mkdir -p "$WS"; elif [[ -d /workspace ]]; then WS=/workspace; else WS="$HOME"; fi

command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA GPU/driver on this machine"

# ---------------------------------------------------------------- 0. repo
if [[ -n "${REPO_URL:-}" ]]; then
  REPO_DIR="${REPO_DIR:-$WS/lewm-potential}"
  if [[ -d "$REPO_DIR/.git" ]]; then
    log "repo already cloned at $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD))"
  else
    log "cloning $REPO_URL -> $REPO_DIR"
    git clone ${REPO_BRANCH:+-b "$REPO_BRANCH"} "$REPO_URL" "$REPO_DIR"
  fi
else
  REPO_DIR="$HERE"
fi
[[ -f "$REPO_DIR/potential/orchestrate.py" && -f "$REPO_DIR/jepa.py" ]] || die "$REPO_DIR is not the lewm-potential repo with potential/"
OUT_DIR="${OUT_DIR:-$REPO_DIR/out}"
mkdir -p "$OUT_DIR/env"
export STABLEWM_HOME="${STABLEWM_HOME:-$WS/stablewm}"
DS="$STABLEWM_HOME/datasets"
mkdir -p "$DS" "$STABLEWM_HOME/.fm_markers"
log "repo $REPO_DIR | out $OUT_DIR | STABLEWM_HOME $STABLEWM_HOME"

# ---------------------------------------------------------------- 1. GPU / disk / network
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || die "nvidia-smi failed"
CUDA_DRV="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -1)"
log "driver supports CUDA $CUDA_DRV"

need_gb=12   # venv + outputs headroom
[[ -f "$STABLEWM_HOME/.fm_markers/lance.complete" ]] || need_gb=$((need_gb + 15))
[[ -f "$STABLEWM_HOME/.fm_markers/h5.complete" ]] || need_gb=$((need_gb + 61))   # 46.3 GB .h5 + 13.1 GB .zst transient
free_gb=$(df -Pk "$STABLEWM_HOME" | awk 'NR==2 {printf "%d", $4/1024/1024}')
df -h "$STABLEWM_HOME" "$OUT_DIR" | sed 's/^/[setup]   /'
(( free_gb >= need_gb )) || die "only ${free_gb} GB free under $STABLEWM_HOME, need ~${need_gb} GB (datasets + venv + outputs). Use a bigger container/volume disk."
log "disk ok: ${free_gb} GB free, ~${need_gb} GB needed"
for u in https://huggingface.co https://pypi.org https://download.pytorch.org; do
  curl -sS -o /dev/null --max-time 20 "$u" || die "network check failed for $u"
done
log "network ok"

# ---------------------------------------------------------------- 2. Python env
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
fi
VENV="${VENV:-$WS/pot-venv}"
[[ -x "$VENV/bin/python" ]] || uv venv --python 3.10 "$VENV"   # README: python 3.10
PY="$VENV/bin/python"

ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }
if ver_ge "$CUDA_DRV" 12.8; then IDX=cu128
elif ver_ge "$CUDA_DRV" 12.6; then IDX=cu126
elif ver_ge "$CUDA_DRV" 12.4; then IDX=cu124
elif ver_ge "$CUDA_DRV" 12.1; then IDX=cu121
else die "driver CUDA $CUDA_DRV too old (need >= 12.1)"; fi

if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "installing torch/torchvision from the $IDX index"
  uv pip install --python "$PY" torch torchvision --index-url "https://download.pytorch.org/whl/$IDX"
fi
TORCH_V="$("$PY" -c 'import torch; print(torch.__version__)')"
TV_V="$("$PY" -c 'import torchvision; print(torchvision.__version__)')"
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_V" "$TV_V" > "$OUT_DIR/env/constraints.txt"
# The repo pins only python 3.10 (README); the two libraries are pinned to the versions this code was written against.
uv pip install --python "$PY" -c "$OUT_DIR/env/constraints.txt" \
  --extra-index-url "https://download.pytorch.org/whl/$IDX" --index-strategy unsafe-best-match \
  "stable-worldmodel[train]==0.1.1" "stable-pretraining==0.1.8" \
  h5py hdf5plugin "imageio[ffmpeg]" pygame pymunk shapely opencv-python-headless \
  huggingface_hub zstandard matplotlib
"$PY" -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('[setup] torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------- 3. stable_worldmodel check
"$PY" - <<'EOF' || die "stable_worldmodel is not importable"
import os, warnings
warnings.filterwarnings("ignore")
import stable_worldmodel as swm, stable_pretraining as spt
from stable_worldmodel.data.utils import get_cache_dir
print(f"[setup] stable_worldmodel {getattr(swm, '__version__', '0.1.1')} importable; formats: {list(swm.data.FORMATS)}")
print(f"[setup] swm.data.load_dataset(name): resolves <STABLEWM_HOME>/datasets/<name> = {get_cache_dir(sub_folder='datasets')}/<name>;"
      " format is auto-detected by suffix ('.lance' dir -> LanceDataset, '.h5' -> HDF5Dataset);"
      " only '<user>/<repo>' names auto-download from the HF Hub.")
import gymnasium as gym
env = gym.make("swm/PushT-v1", render_mode="rgb_array"); env.reset(seed=0); env.close()
print("[setup] gym swm/PushT-v1 ok")
EOF

# ---------------------------------------------------------------- 4. data
export HF_HUB_DISABLE_PROGRESS_BARS=0
"$PY" - "$DS" "$STABLEWM_HOME/.fm_markers" <<'EOF' || die "dataset download failed (see above)"
import json, os, sys, time, urllib.request
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
ds, markers = Path(sys.argv[1]), Path(sys.argv[2])
api = HfApi()

def fail(msg):
    print(f"[setup] MISSING: {msg}", file=sys.stderr); sys.exit(1)

# --- Lance (training data, train.py / config/train/data/pusht.yaml: pusht_expert_train.lance)
LREPO, LDIR = "galilai-group/lewm-pusht", "pusht_expert_train.lance"
lance = ds / LDIR
if (markers / "lance.complete").exists() and lance.is_dir():
    print(f"[setup] lance present: {lance}")
else:
    try:
        files = [f for f in api.list_repo_tree(LREPO, repo_type="dataset", recursive=True, path_in_repo=LDIR)
                 if getattr(f, "size", None) is not None]
    except Exception as e:
        fail(f"cannot list HF dataset {LREPO}/{LDIR} ({e!r}). Place the Lance table at {lance}")
    expected = {f.path: f.size for f in files}
    print(f"[setup] downloading {LREPO}/{LDIR}: {len(expected)} files, {sum(expected.values())/1e9:.1f} GB -> {ds}")
    t = time.time()
    snapshot_download(LREPO, repo_type="dataset", allow_patterns=[f"{LDIR}/*", f"{LDIR}/**"], local_dir=str(ds),
                      max_workers=16)
    bad = [p for p, s in expected.items() if not (ds / p).is_file() or (ds / p).stat().st_size != s]
    if bad:
        fail(f"{len(bad)} Lance files missing/incomplete after download, e.g. {bad[:3]}")
    (markers / "lance.complete").write_text(json.dumps({"files": len(expected), "bytes": sum(expected.values())}))
    print(f"[setup] lance ok in {time.time()-t:.0f}s")

# --- HDF5 (eval data, eval.py / config/eval/pusht.yaml: pusht_expert_train -> pusht_expert_train.h5)
HREPO, HFILE = "quentinll/lewm-pusht", "pusht_expert_train.h5.zst"
h5 = ds / "pusht_expert_train.h5"
if (markers / "h5.complete").exists() and h5.is_file():
    print(f"[setup] hdf5 present: {h5}")
else:
    import zstandard
    dl = ds / ".download"
    try:
        t = time.time()
        zst = Path(hf_hub_download(HREPO, HFILE, repo_type="dataset", local_dir=str(dl)))
    except Exception as e:
        fail(f"cannot download {HREPO}/{HFILE} ({e!r}). Place the decompressed file at {h5}")
    print(f"[setup] downloaded {zst} ({zst.stat().st_size/1e9:.1f} GB) in {time.time()-t:.0f}s; decompressing (zstd -d)")
    # the HF file is a plain zstd stream of the .h5 (not a tar, despite the README's `tar --zstd -xvf`)
    with open(zst, "rb") as f:
        fp = zstandard.get_frame_parameters(f.read(18))
    part = h5.with_name(h5.name + ".part")
    t = time.time()
    with open(zst, "rb") as fi, open(part, "wb") as fo:
        zstandard.ZstdDecompressor(max_window_size=2**31).copy_stream(fi, fo, read_size=1 << 24, write_size=1 << 24)
    size = part.stat().st_size
    if fp.content_size not in (0, zstandard.CONTENTSIZE_UNKNOWN) and size != fp.content_size:
        fail(f"decompressed size {size} != frame header {fp.content_size}")
    with open(part, "rb") as f:
        if f.read(8) != b"\x89HDF\r\n\x1a\n":
            fail("decompressed file has no HDF5 signature")
    os.replace(part, h5)
    zst.unlink()
    (markers / "h5.complete").write_text(json.dumps({"bytes": size}))
    print(f"[setup] hdf5 ok ({size/1e9:.1f} GB) in {time.time()-t:.0f}s")

def du(p):
    p = Path(p)
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
print(f"[setup] DATASETS:\n[setup]   {lance}  {du(lance)/1e9:.2f} GB\n[setup]   {h5}  {du(h5)/1e9:.2f} GB")
EOF

# ---------------------------------------------------------------- 5. record env
"$PY" -m pip --version >/dev/null 2>&1 || true
uv pip freeze --python "$PY" > "$OUT_DIR/env/pip_freeze.txt"
"$PY" - "$OUT_DIR/env/setup.json" <<EOF
import json, shutil, subprocess, sys, torch
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
json.dump({"gpu": gpu, "cuda_driver": "$CUDA_DRV", "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "torch_index": "$IDX", "venv": "$VENV", "repo": "$REPO_DIR", "stablewm_home": "$STABLEWM_HOME",
           "free_gb": shutil.disk_usage("$STABLEWM_HOME").free / 1e9}, open(sys.argv[1], "w"), indent=2)
EOF
cat > "$HOME/.pot_env" <<EOF
export REPO_DIR="$REPO_DIR"
export VENV="$VENV"
export STABLEWM_HOME="$STABLEWM_HOME"
export OUT_DIR="$OUT_DIR"
EOF
log "done. env recorded in $OUT_DIR/env/ (pip_freeze.txt, setup.json); variables in ~/.pot_env"
