# HPC Agent

HPC Agent is a local AI agent runner focused on shared-allocation,
resource-aware orchestration.

Current state:

- runner-only architecture
- shared-allocation M1 baseline complete
- M2 single-manager, multi-worker orchestration baseline implemented
- DAG scheduler with `worker_class` pooled dispatch
- GPU token admission and graph-drain replanning
- default local agent runtime: `codex_cli`
- optional compatibility runtime: `api`

## Setup

```bash
bash setup.sh
```

This checkout is self-contained and no external upstream checkout is required.

## Running the autoresearch loop

The agent runner must execute on a GPU compute node with direct `/dev/nvidia*`
access. The old login-node + `srun --overlap` attach path is not usable on this
cluster (the codex sandbox cannot open slurm stream sockets).

Submit the runner as a sbatch job:

```bash
sbatch infra/hpc_agent/runner/scripts/submit-autoresearch-runner.sbatch
```

This allocates one 8-GPU node, materializes the dag-full project workspace
under `artifacts/autoresearch-runner-<timestamp>/`, and starts the agent
server in foreground for the duration of the job. Logs land in
`artifacts/autoresearch-runner-<timestamp>/server.log`.

Notes:

- local orchestration now defaults to `agentRuntime: codex_cli`
- provider API keys are only needed for projects that explicitly set `agentRuntime: api`
- scheduler design: `runner/docs/SCHEDULER.md`
- Codex runtime notes: `runner/docs/CODEX_LOCAL_RUNTIME.md`
- resource validation plan: `runner/docs/RESOURCE_ORCHESTRATION_TEST_PLAN.md`
