# Experiment 0024_adambeta_09

## Experiment Details
- **id**: 0024_adambeta_09
- **hypothesis**: ADAM_BETAS first beta is unusually low (0.8); raising to the more common 0.9 may stabilize training.
- **parent**: shared baseline repo (/mnt/weka/home/junlin.chen/workspace/autoresearch)
- **exact_edit**: in train.py line 476, change `ADAM_BETAS = (0.8, 0.95)` to `ADAM_BETAS = (0.9, 0.95)`

## Resources
- **granted_gpu_ordinals**: 4
- **requested_resources**: gpu_tokens=gpu4, recommended_cpus_per_task=1

## Repository State
- **branch**: 0024_adambeta_09
- **commit**: 61b92058d01792bdc719769598bb0d3fa5112788
- **commit_short**: 61b9205
- **timestamp**: 2026-04-25T12:15:00Z

## Results
- **suggested_status**: keep — Training completed successfully with stable convergence and reasonable val_bpb.
- **val_bpb**: 1.0556
- **training_seconds**: 300.4
- **num_steps**: 451
- **peak_vram_mb**: 90632.2
- **depth**: 8
- **seed**: 42
- **config.adam_betas**: [0.9, 0.95] ✓ (edit confirmed)
