## Potential variant (this repo) — what differs from the bridge notes below
- The field is v = −∇ₓU (PotentialHead: scalar output, same conditioning trunk as VelocityHead).
  P1/P2 use the A1/A2 losses and samplers unchanged. Parameters at W=256: head 3,450,881, total
  20,692,591 (bridge: 3,499,968 / 20,741,678; baseline 18,034,478).
- The baseline jobs and `calibration.json` are **imported from the bridge repo's results**
  (`--bridge_out`), not trained here. They come from the bridge's `fm/train_fm.py`, which is the same
  trainer as `potential/train_fm.py`. Provenance is in `jobs/<id>/imported_from.json`.
- The gate is the bridge's rule with P1 σ=0, S=1 as the control. No auto-diagnostic job runs on a
  gate failure (the bridge's BatchNorm-on-output fix would break v = −∇U).
- ∇U at planning time is computed outside `inference_mode` (the CEM solver runs inside it) on a
  detached copy of x. Its result is detached.
- Everywhere below, read "flow models" as including the potential models, and A1/A2 as P1/P2.

---

## Step 0 — verification (done before coding)

1. **stable_worldmodel / datasets.** `stable-worldmodel==0.1.1` and `stable-pretraining==0.1.8` (latest on PyPI) are installed by `setup.sh`. The `[format]` extra is replaced by explicit `h5py hdf5plugin imageio`, because its `decord` dependency has no wheel on some platforms. `swm.data.load_dataset(name)` resolves `name` under `$STABLEWM_HOME/datasets/`. The default `$STABLEWM_HOME` is `~/.stable_worldmodel`; the README's `~/.stable-wm` and "files directly under `$STABLEWM_HOME`" are stale. The format is then auto-detected from the suffix: a `*.lance` directory opens `LanceDataset`, a `*.h5` file opens `HDF5Dataset`. Only `<user>/<repo>` names auto-download from the HF Hub, so a bare `pusht_expert_train.lance` (train config) or `pusht_expert_train` (eval config, read with `HDF5Dataset` → `.h5`) has **no automatic download**. Both files are on the HF Hub, and `setup.sh` fetches them as-is:
   - Lance (train.py): HF dataset `galilai-group/lewm-pusht` → `pusht_expert_train.lance/` (14.3 GB).
   - HDF5 (eval.py): HF dataset `quentinll/lewm-pusht` → `pusht_expert_train.h5.zst` (13.1 GB). This is a **plain zstd stream of the .h5**, not a tar, so the README's `tar --zstd -xvf` does not apply. It is decompressed with `zstd -d` to `pusht_expert_train.h5` (46.3 GB, size checked against the zstd frame header).

   No conversion, and no loader changes. Pre-flight compares the episode/frame counts of the two files and prints whether they match.
2. **Parameter counts (repo model, PushT, action input 10):**

   | module | params |
   |---|---|
   | encoder (ViT-tiny) | 5,501,376 |
   | encoder projector | 792,768 |
   | predictor trunk | 10,791,360 |
   | pred_proj | 792,768 |
   | action embedder | 156,206 |
   | **total** | **18,034,478** |

   The flow models drop pred_proj and add the velocity head. At W=256 the head has 3,499,968 params, for a total of 20,741,678 (+2.71 M, +15 % vs baseline). At W=128 the head has 947,264 (total 18,188,974). At W=512 it has 13,422,272 (total 30,663,982).
3. **Embedder.smoothed_dim=10.** It is a Conv1d(k=1) from 10 to 10 channels, i.e. a per-timestep linear map, followed by Linear(10→768) → SiLU → Linear(768→192). For PushT, frameskip × action_dim = 5 × 2 = 10, so the 10-channel conv is square and **not a bottleneck**. Left unchanged.
4. **Known facts confirmed in code.**
   - Trunk: D=192, 16 heads × 64, MLP 2048, 6 layers.
   - Positional embedding: learned, (1,3,192), added to the state embeddings only, before dropout. Actions enter only via AdaLN.
   - A_EMB = 192.
   - pred_proj and the projector: Linear(192,2048) → BatchNorm1d → GELU → Linear(2048,192) on flattened (B·T,192).
   - Window: 4 frames, `ctx=emb[:, :3]`, `tgt=emb[:, 1:]`, action block 3 unused, num_preds=1.
5. **Mixed precision.** The repo trains with Lightning `precision: bf16`, which Lightning maps to `bf16-mixed` (autocast bf16, fp32 weights, no GradScaler). Baseline and flow models here both train under the same `torch.autocast(bfloat16)`. Evals for both run in fp32, as eval.py does.

### Facts found along the way
- **PushT data size.** The Lance table has 18,685 episodes and 2,336,736 frames (≈125 steps/episode, not the paper's 20k × 196). That gives 1,981,721 training windows (span 20 = 4 frames × frameskip 5, every start position) and 1,783,549 train / 198,172 val after the 0.9 split. With batch 128 and drop_last, that is **13,934 steps/epoch**.
- **LR schedule.** `spt.Module` uses manual optimization and calls `scheduler.step()` after every optimizer step. The `"interval": "epoch"` in train.py is therefore ignored. The schedule is `LinearWarmupCosineAnnealingLR` with warmup = 1 % of `trainer.estimated_stepping_batches` (= epochs × steps/epoch), cosine to 0 at the last step. The schedule therefore **depends on total epochs** (see the extension caveat).
- **Eval history.** At eval time `PlanConfig.history_len` defaults to 1, because config/eval does not set it. Planning therefore starts from one encoded frame and the window grows to 3 during the rollout; training uses 3 context frames.

## How the baseline is trained here (same numerics as the flow models)
The repo's `train.py` (Lightning + spt.Module) is left untouched and still runs. For checkpoint/resume with optimizer, scheduler and RNG state, **both** the baseline and the flow models are trained by `fm/train_fm.py`, a plain-PyTorch loop that reproduces train.py's computation:
- same composed config, dataset, transforms, normalizers and split (seed 3072);
- AdamW via `spt.optim.create_optimizer` (lr 5e-5, wd 1e-3);
- the same spt scheduler class with spt's default arguments, stepped every optimizer step;
- bf16 autocast, grad-norm clip 1.0, batch 128, drop_last;
- the baseline loss **is** `train.py::lejepa_forward`, imported and called through a tiny shim.

The job seed controls model init, SIGReg projections, flow noise and the shuffle order. The train/val split stays at the repo's seed 3072 for every run. train.py does not seed model init at all.

## Deviations from the spec / things to know
- **Validation.** Per-epoch validation runs on a fixed 100-batch val subset (12.8k windows), not the full val set Lightning would run. This is for speed and does not affect training. Flag: `--val_batches -1` for the full set.
- **Dataloader workers.** Default is max(6, min(cpus−2, 16)); the repo uses 6. Data loading (JPEG decode) is the likely bottleneck, and the worker count changes throughput only, not numerics or data order.
- **Eval.** Same sampling and normalisation as eval.py, but `eval.num_eval=200` (same 200 start states for every run; `eval_set_hash` in every result JSON) and no per-episode videos.
- **Gate rule.** Pass iff the control ≥ baseline − max(2·√(SE_b²+SE_c²), 2 points). A control that is *better* than the baseline passes.
- **Tier-4 planning sweep.** One factor at a time: S∈{1,2,4,8} at K=1, K∈{1,2,4} at S=4, plus `mean` mode at S=4. `--sweep_grid` runs the full S×K grid. The S=4,K=1 result is reused from the tier-2 job, which ran the identical evaluation.
- **Common random numbers.** One noise draw per `get_cost` call (= one CEM iteration), shared by all 300 candidates. Each particle k has its own noise. The draws come from a dedicated generator seeded by `--plan_seed` (default 0).
- **HF upload.** The incremental `out/` mirror is uploaded at most every 15 min, followed by `super_squash_history` to keep storage at the current size. `bundle.tar.gz` itself is uploaded at the end, on interrupt and on crash, not every 15 min. Uploading a multi-GB tarball every 15 min would multiply storage and bandwidth.
- **Checkpoints.** Each epoch writes a full state (model + optimizer + scheduler + RNG); the last 2 are kept. Milestones 5/10/(20)/50/100 keep a model-only file. When a job finishes, only the final full checkpoint is kept (needed for `--extend_mode continue`). The mid-epoch full checkpoint is written every 10 min (`full_mid.pt`) and removed at each epoch end.

## Code/paper mismatches kept as-is (baseline = repo defaults)
- SIGReg integration range [0,3] with 17 knots (paper App. A: [0.2,4]).
- λ = 0.09 (paper text: 0.1).
- The prediction loss averages over the 192 dims (Eq. 1 sums). The flow loss uses the same mean, so λ keeps its tuned scale.
- History 3 everywhere, `JEPA.rollout(history_size=3)` default.
- CEM 30 iterations (paper: 30 for PushT, matching here).
- `Embedder.smoothed_dim=10`.
- `eval.py` sampling skips the last valid start index (`len(valid_indices) - 1`).
- Normalisation uses sample std (torch, train) vs population std (sklearn, eval).
- Stale docstrings ("dinowm vs random", README `_object.ckpt` / `~/.stable-wm`).
- Projector "1-layer MLP" in the paper vs the 2-layer MLP in the code; predictor "ViT-S" vs width 192.
- `max_epochs: 100` in the config. The study uses `--epochs 20` for **both** baseline and flow.

## Epoch-extension caveat
The LR schedule depends on total epochs (warmup = 1 % of total steps, cosine reaching 0 at the end). Resuming a finished 20-epoch run with `--epochs 50` is therefore **not** equivalent to a fresh 50-epoch run: at epoch 20 the 20-epoch schedule is at LR ≈ 0, while a 50-epoch schedule would be at ≈ 0.65·peak.
- `--extend_mode fresh` (default): `--epochs 50` creates new `_e50` jobs trained from scratch. These are equivalent to fresh 50-epoch runs.
- `--extend_mode continue`: `_e50` jobs warm-start from the `_e20` job's final full checkpoint. The schedule is rebuilt for 50 epochs at the current step (an LR jump back up), and the job is marked `extended_from=20 (...)` in the manifest and report. Earlier milestone evals are copied over.

The same mode applies to baseline and flow jobs, so the baseline is always treated identically and never gets a smaller budget.
