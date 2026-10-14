# Potential-field predictor for LeWM (v = −∇U) — PushT phase

This is the conservative-field counterpart of the bridge variant in
[fm-jepa](https://github.com/axel-delmas/fm-jepa). The one-step latent transition
z_t → z_{t+1} is sampled by integrating a velocity field that is the negative gradient of a learned
scalar potential, **v = −∇ₓ U(x, τ | h_t, z_t, a_t)**. The bridge variant instead uses a free vector
field **v = head(x, τ | h_t, z_t, a_t)**.

Everything else is identical to the bridge repo: encoder, projector, action embedder, ARPredictor
trunk, SIGReg, optimizer, LR schedule, bf16, data, eval protocol, CEM planning, K-particle rollout
with common random numbers, σ calibration, σ=0 control and the gate. The original LeWM files
(`train.py`, `eval.py`, `jepa.py`, `module.py`, `utils.py`, `config/`) are unchanged. Everything new
lives in `potential/`, `tests/`, `bench/`, `setup.sh`, `run_all.sh`, `ssh_check.sh` and `SSH_GPU.md`.

## Variants
Per pair: τ ~ U[0,1], ε ~ N(0, I), and σ is a per-dimension multiple of the calibrated std of z_{t+1} − z_t.

| | loss (same as bridge) | sampler |
|---|---|---|
| **P1** (= A1 with v = −∇U) | x0 = z_t + σε, x_τ = (1−τ)x0 + τz1, mean‖v − (z1 − x0)‖² | Euler x ← x − dt ∇U, S steps |
| **P2** (= A2 with v = −∇U) | x_τ = (1−τ)z_t + τz1 + σ√(τ(1−τ))ε, x̂1 = x_τ + (1−τ)v, mean‖x̂1 − z1‖² | Euler–Maruyama, no noise on the last step |

Not implemented (TODO in `potential_jepa.py`): unrolled gradient descent on U trained end to end,
and denoising score matching with U as an energy.

## Head
`PotentialHead` has the same conditioning trunk as the bridge's `VelocityHead`:
- `x_in`, a sinusoidal τ embedding, a conditioning MLP on [h_t, z_t, a_t, τ], 4 AdaLN-zero blocks
  and an AdaLN final layer;
- the building blocks are copied verbatim;
- only smooth nonlinearities are used (LayerNorm, GELU, SiLU), so ∇U is continuous.

The output is a zero-initialised `Linear(W, 1)`, so U ≡ 0 and v ≡ 0 at init. U's output bias never
gets a gradient, since a constant does not change ∇U.

| module | params |
|---|---|
| baseline LeWM (with pred_proj) | 18,034,478 |
| bridge: VelocityHead W=256 | 3,499,968 (total 20,741,678) |
| **potential: PotentialHead W=256** | **3,450,881 (total 20,692,591)** |

## Gradients and precision
- **Training.** ∇ₓU is taken with `autograd.grad(..., create_graph=True)` (double backward). The
  loss reaches the head, h_t, z_t, a_t and, through x_τ, the encoder, along the same paths as the
  bridge's v = head(x_τ, …).
- **Sampling / no-grad loss.** stable_worldmodel's `CEMSolver.solve` runs under
  `@torch.inference_mode()`, and `train_fm.diagnostics` runs under `@torch.no_grad()`. In both
  cases ∇ₓU is computed:
  - inside `torch.inference_mode(False)` + `torch.enable_grad()`;
  - with respect to a detached copy of x only (inputs are cloned when they are inference tensors);
  - and the result is detached.

  h_t, z_t, a_t and the parameters never receive gradients from sampling. The tests cover the
  `no_grad`, `inference_mode` and grad-mode cases, plus a run through the real `CEMSolver`.
- **bf16.** The default is the bridge's numerics: head and ∇U under bf16 autocast. On CPU
  (`bench/bf16_check.py`, D=192, W=256, non-zero output layers), the error vs fp32 is:

  | | v field | training gradient |
  |---|---|---|
  | potential | 0.57 % | 0.45 % |
  | bridge head | 0.60 % | 0.49 % |

  This was not worse than the bridge, so no fp32 fallback is used by default. `--head_fp32` runs the
  head and the gradient call outside autocast; the job id gets `_fp32`. Preflight logs the same
  bf16-vs-fp32 field error on the GPU on a real batch.

## Cost (CPU micro-benchmark, `bench/timing.py`, fp32, Apple M5, real model sizes)
| | bridge | potential | ratio |
|---|---|---|---|
| full train step, batch 8 (encode+trunk+loss+SIGReg+bwd+AdamW) | 0.51 s | 0.50 s | ≈1.0× |
| head-only train step, 384 pairs (= batch 128 × 3) | 11.4 ms | 21.7 ms | 1.9× |
| CEM call, 300 candidates, P1/A1 σ=0 S=1 K=1 | 0.154 s | 0.162 s | 1.05× |
| CEM call, P1/A1 σ=1 S=4 K=1 | 0.201 s | 0.253 s | 1.26× |
| CEM call, P2/A2 σ=1 S=4 K=1 | 0.206 s | 0.264 s | 1.28× |
| CEM call, P2/A2 σ=1 S=4 K=4 | 0.651 s | 0.789 s | 1.21× |

On GPU the head is a larger share of the step, so the ratios may be larger. `run_all.sh --benchmark`
measures them on the pod.

## Run on a pod
Same requirements as the bridge repo (~110 GB disk). Results go to a private draft GitHub Release
of this repo, with tag prefix `potential-results-`.
```bash
git clone https://<GITHUB_TOKEN>@github.com/axel-delmas/lewm-potential.git lewm-potential && cd lewm-potential
# bridge results to compare against (out/ dir or bundle .tar.gz; repeatable), copied to the pod first
bash run_all.sh --tiers 1 --epochs 5 --bridge_out /path/to/bundle_e5_2026-10-08_tier1-complete.tar.gz
tail -f out/run.log
```
- **Flags.** Same as the bridge (see its `fm/README.md`), with these differences:
  - `--t2_variants` defaults to `P1,P2`;
  - `--head_fp32` is new;
  - `--bridge_out` is required, because the baseline is never trained here.
- **Environment.** `FM_*` variables are renamed `POT_*`, `~/.fm_env` becomes `~/.pot_env`, and the
  venv is `$WS/pot-venv`. The dataset cache (`$STABLEWM_HOME`, including its `.fm_markers`) is shared
  with the bridge repo, so a pod that already holds the PushT data does not download it again.
- **Local tests** (CPU): `python -m pytest tests -q`. The tests need torch and einops. The CEM test
  also needs stable-worldmodel and is skipped without it. The shape comparison against `FlowJEPA`
  reads `../le-wm` (override with `BRIDGE_REPO`).

## Fair comparison with the bridge repo (how results are joined)
- **No baseline is trained here.** Tier-1/3 baseline jobs and `calibration.json` are imported from
  `--bridge_out`:
  - the job ids are the bridge's (e.g. `baseline_seed0_e5`);
  - the import is refused if kind/seed/epochs/λ/num_eval differ, or if the calibration does not
    come from that baseline;
  - checkpoints are not copied;
  - provenance (source, bridge git commit, GPU) goes to `jobs/<id>/imported_from.json`.
- **The gate is unchanged:** P1 σ=0 S=1 vs the imported baseline, same rule. The bridge's
  auto-diagnostic (BatchNorm on the head output) has no conservative counterpart, so it is not
  scheduled.
- **Same eval start states.** `eval_fm.py` is the bridge's evaluation (same config, seed and
  sampling). Every result JSON carries `eval_set_hash`; the 5-epoch bridge bundle has `ca4bb61b1ec43790`.
- **Same epoch budget.** Use the same `--epochs` as the bridge results. Imports refuse a mismatch.
- **Join.** `report.md` and `joined_results.csv` (also `python -m potential.report out --bridge_out …`):
  - each potential eval is matched to the bridge eval with the same settings (P1↔A1, P2↔A2; same σ
    multiple, λ, head width, seed, epochs, eval epoch, S, K, mode) and to the imported baseline;
  - the pairing is per episode on (episode, start) after checking `eval_set_hash`;
  - it reports Δ success, the discordant counts and an exact McNemar p-value;
  - planning times are only comparable when both runs used the same GPU.

## Layout
- `potential/potential_head.py`: PotentialHead (scalar, AdaLN-zero blocks, sinusoidal τ embedding).
- `potential/potential_jepa.py`: PotentialJEPA(JEPA). Contains the field v = −∇U, the P1/P2 losses,
  the one-step sampler, the K-particle CRN rollout and the cost.
- `potential/{common,train_fm,eval_fm,calibrate,preflight,github_sync,orchestrate,report}.py`: the
  bridge's run infrastructure (copied verbatim in one commit, adapted in the next). The trainer
  still uses the bridge's per-epoch diagnostics, including the per-τ-bin loss.
- `tests/`: CPU unit tests. `bench/`: bf16 check and CPU timing.
- `potential/REPORT_NOTES.md`: protocol notes appended to every report.
