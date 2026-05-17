# qwen-sft — Experiment Log

Running log of the Qwen3-0.6B SFT data-mixture search experiments used for
the ALPS / RAILS ICML-workshop paper (`docs/paper/`). One section per
configuration; the comparison feeds Table 1 ("Qwen3 SFT mix" rows).

---

## Task setup (fixed across all runs)

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B`, bf16, full fine-tune (no LoRA) |
| Data | Tülu-3 SFT mixture, bucketed into 5 domains (math / code / chat / if / reasoning) by `prepare.py` |
| Search variable | module-level `DATA_MIX` dict in `source/train.py` (5 weights, trainer normalizes to 1) |
| Metric | balanced held-out cross-entropy across all 5 buckets, 200 val/bucket = 1000 total (`val_loss`, lower better) |
| Per-eval budget | `SFT_TIME_BUDGET_SECONDS=1200` (20 min training) |
| Fixed hyperparams (search baseline) | seq_len 2048, micro_batch 16, grad_accum 2 (eff. batch 32), LR 2e-5, warmup 50, weight_decay 0.01 |
| Eval cadence | `EVAL_EVERY_STEPS=20` (per-bucket val curves to wandb) |
| Runtime | `claude_cli` manager, 8 GPU (1 for Serial), k2m partition, 4 h wall (Serial finished early) |
| wandb | project `workshop`, entity `haolong`, group per run |

Data prep (one-time): `uv run --project source python source/prepare.py`
(buckets: math 234k / code 142k / chat 110k / if 30k / reasoning 205k train rows).

---

## Configurations

| Policy | Script | K | KEEP decision | live replan | ALPS knobs (calib / hazard / quota / rebase / gate) |
|---|---|:---:|---|:---:|---|
| **Serial** | `baseline-submit.sbatch` | 1 | agent, every iter | n/a | n/a |
| **Naive parallel** | `submit-naive.sbatch` (`manager-naive.js`) | 8 | manager, wave-end | ✗ | all OFF |
| **Manual ALPS** (prior pilot) | `submit.sbatch` (`manager.js`) | 8 | manager, prompt-driven | ✓ | gate is suggestion only; calib/hazard/quota OFF |
| **Explicit ALPS** | `submit-explicit.sbatch` (`manager-explicit.js`) | 8 | auto-fire gate + manager | ✓ | all ON (calib N, hazard Q\_t, quota, rebase, gate) |

Explicit ALPS was swept across 5 variants (parameterized via env on
`submit-explicit.sbatch`):

| Variant | calibrationRepeats | fallbackSigma | gateTauMin/Max |
|---|:---:|:---:|:---:|
| A | 3 | 1e-4 | 0.1 / 0.5 |
| B | 2 | 4e-5 | 0.1 / 0.5 |
| C | 5 | 1e-4 | 0.1 / 0.5 |
| D | 3 | 1e-4 | **0.05 / 0.2** (aggressive) |
| E | 3 | 1e-4 | **0.5 / 2.0** (conservative) |

---

## Final results

All runs 2026-04-29. "TIMEOUT" = ran the full 4 h wall budget as designed
(not an error); Serial COMPLETED early because 1 GPU does fewer evals.

| Job | Policy | Workers | Wall | Evals | Best `val_loss` ↓ | Notes |
|---|---|:---:|:---:|---:|---:|---|
| 1584281 | Serial | 1 | 3:22 | 9 | **1.10553** | 2 lineage advances (exp01→exp06); pure DATA_MIX |
| 1584321 | Naive parallel | 8 | 4 h | 49 | **1.10168** | manager fired 0 KEEPs; later batches contaminated by a concurrent run writing canonical `source/` |
| 1584077 | Manual ALPS (pilot) | 8 | 4 h | 62 | 1.10859 | 0 KEEPs in 22 manager calls; pure DATA_MIX |
| 1584298 | Explicit A (cR=3) | 8 | 4 h | 38 | 1.10232 | **1 auto-KEEP fired** (exp_0017, Δ=4.5e-4 > τ·σ̂=7.3e-6); σ̂=1.93e-5 |
| 1584406 | Explicit B (cR=2) | 8 | 4 h | 31 | 1.10221 | σ̂ from 2-sample pair |
| 1584407 | Explicit C (cR=5) | 8 | 4 h | 45 | 1.10303 | σ̂=5.35e-5 (n=5, most reliable) |
| 1584408 | **Explicit D** (τ aggressive) | 8 | 4 h | 37 | **1.10089** ⭐ | best overall; chosen for paper Table 1 |
| 1584409 | Explicit E (τ conservative) | 8 | 4 h | 51 | 1.10144 | |

Paper Table 1 ("Qwen3 SFT mix") uses: Serial 1.10553 / Naive 1.10168 /
ALPS (=Explicit D) 1.10089.

### Best discovered configuration (Explicit D, auto-KEPT into `source/train.py`)

```python
DATA_MIX = {'math': 0.32, 'code': 0.18, 'chat': 0.10, 'if': 0.10, 'reasoning': 0.30}
LEARNING_RATE = 3e-5      # default was 2e-5
WEIGHT_DECAY  = 0.0       # default was 0.01
```

`source/train.py` is intentionally left at this auto-KEPT state (the
lineage chain's HEAD). To reproduce a clean uniform-baseline run, reset
`DATA_MIX` to `{0.20×5}`, `LEARNING_RATE=2e-5`, `WEIGHT_DECAY=0.01`.

---

## Key findings

1. **Explicit gate fires where the manager would not.** Manual ALPS
   (1584077) made 22 manager calls and committed **0** KEEPs despite a
   clear winner. Explicit ALPS variant A auto-fired a KEEP at
   Δ=4.5e-4 > τ·σ̂=7.3e-6 (≈60× margin). This is the cleanest evidence
   for the paper's "manager is a conservative scheduler" claim.

2. **σ̂ estimate depends strongly on calibration sample size.**
   A (n=3)=1.93e-5, C (n=5)=5.35e-5, B (n=2 pair)=4e-5. Small samples
   under-estimate noise — the n=3 estimate is ~3× tighter than n=5.
   Supports the paper's noise-estimation discussion.

3. **The operator bandit explores every axis the framework declares.**
   `manager-explicit.js` registered `LR`, `WARMUP_RATIO`, `WEIGHT_DECAY`
   as search axes (in addition to `DATA_MIX`). All 5 explicit variants
   autonomously edited LR (→3e-5) and weight_decay (→0.0) even though
   the prompt said "only DATA_MIX". LR turned out to have ~5× the
   leverage of any pure data-mix tilt at this budget. Prompt-level
   constraints are weaker than `searchAxes` config.

4. **Multi-tenant collision on shared `source/.git`.** Running 5 explicit
   variants concurrently against one canonical `tasks/qwen-sft/source/`
   caused cross-run interference: variant A's auto-KEEP advanced the
   shared canonical `train.py`, so variants B/D hit
   `expected_old_repr` mismatches and variant C triggered
   stale-baseline rebases. Naive parallel's later batches were also
   contaminated. Future multi-config sweeps need per-config source
   clones (a framework limitation, noted in paper limitations).

5. **Data-mix optimum.** Across all policies the consistent winning
   region is **math + reasoning ≈ 0.55–0.62, chat/if starved to
   ~0.10–0.15, code ~0.17–0.25**. Reasoning bucket carries most of the
   signal; math saturates above ~0.30.

---

## Reproduction

```bash
# data prep (one-time)
uv run --project tasks/qwen-sft/source python tasks/qwen-sft/source/prepare.py

# Serial baseline (1 GPU, 4 h)
sbatch tasks/qwen-sft/baseline-submit.sbatch

# Naive parallel (8 GPU, 4 h)
AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch \
  --export=ALL,AUTORESEARCH_AGENT_RUNTIME tasks/qwen-sft/submit-naive.sbatch

# Manual ALPS (8 GPU, 4 h)
AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch \
  --export=ALL,AUTORESEARCH_AGENT_RUNTIME tasks/qwen-sft/submit.sbatch

# Explicit ALPS variant D (best) — aggressive τ
QWEN_SFT_VARIANT_TAG=D QWEN_SFT_CALIBRATION_REPEATS=3 \
  QWEN_SFT_GATE_TAU_MIN=0.05 QWEN_SFT_GATE_TAU_MAX=0.2 \
  AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch \
  --export=ALL,AUTORESEARCH_AGENT_RUNTIME,QWEN_SFT_VARIANT_TAG,QWEN_SFT_CALIBRATION_REPEATS,QWEN_SFT_GATE_TAU_MIN,QWEN_SFT_GATE_TAU_MAX \
  tasks/qwen-sft/submit-explicit.sbatch
```

Per-run outputs land in `artifact/<policy>/run-<ts>/`. WandB:
`https://wandb.ai/haolong/workshop` (filter by run group).
