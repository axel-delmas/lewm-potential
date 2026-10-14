#!/usr/bin/env bash
# Quick pre-flight for a plain SSH GPU machine. Run:  bash ssh_check.sh
echo "== GPUs (index, name, used/total MiB, utilisation) =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || { echo "no nvidia-smi: not on a GPU machine"; exit 1; }
echo; echo "== Disk free (need ~90 GB where WORKDIR points) =="
df -h "${WORKDIR:-$HOME}" | tail -1
echo; echo "== Internet (needed for first-time setup) =="
for u in https://huggingface.co https://pypi.org https://download.pytorch.org; do
  curl -sS -o /dev/null --max-time 10 "$u" && echo "ok   $u" || echo "FAIL $u"
done
echo; echo "== Tools =="
for t in git curl tmux python3; do command -v $t >/dev/null && echo "ok   $t" || echo "MISSING $t"; done
echo; echo "Pick a GPU that shows ~0 MiB used, then: export CUDA_VISIBLE_DEVICES=<its index>"
