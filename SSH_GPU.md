# Running on an SSH GPU machine (baby steps)

1. VSCode terminal (Terminal > New Terminal) is already on the GPU machine. Get the code there
   (git clone your repo, or VSCode "File > Open Folder" on a copy you rsync'd).
2. `cd le-wm && bash ssh_check.sh`  -> shows free GPUs, disk, internet.
3. Choose a big-disk folder and a free GPU (index from step 2):
   ```
   export WORKDIR=/path/with/90GB/free/lewm      # e.g. /scratch/$USER/lewm  (check with df -h)
   export CUDA_VISIBLE_DEVICES=0                 # a GPU with ~0 MiB used in nvtop
   ```
4. Start inside tmux so closing VSCode never kills the job:
   ```
   tmux new -s lewm
   bash run_all.sh --tiers 1 --epochs 1      # tiny smoke test first
   ```
   Detach: Ctrl-b then d.  Reattach later: `tmux attach -t lewm`.
5. Watch: `tail -f out/run.log` (or `$OUT_DIR/run.log`) and `nvtop`.
6. Real run: same command with your tiers/epochs/seeds, e.g.
   `bash run_all.sh --tiers 1,2,3 --epochs 20 --seeds 0,1`.
7. Stop cleanly: `kill -TERM $(cat out/run.pid)`. Resume: re-run the same command (or `--resume out`).
Results end up in `out/` (plus `bundle.tar.gz` in the current dir). Download via VSCode (right-click > Download).
