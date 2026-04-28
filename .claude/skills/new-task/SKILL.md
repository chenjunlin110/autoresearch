---
name: new-task
description: Scaffold a new task plugin under tasks/<name>/ for this concurrent search-task framework. Creates the runnable-code directory, manager prompt, worker prompt, wrapper script, sandbox cloner, workspace generator, submit sbatch, and a smoke test, using tasks/autoresearch/ as the reference template. Use when the user wants to add a new search task (data-mix search, prompt search, hyperparameter sweep on a different trainer, kernel autotuning, etc.) on top of the existing framework.
---

# new-task

Scaffold a fresh task plugin so the user can run a new kind of concurrent search on top of the framework. Output: a complete `tasks/<name>/` directory the user can iterate on, no stubs left disconnected.

## What you do when invoked

### 1. Gather inputs from the user

Ask for the following, in one batch (use AskUserQuestion):

- **task_name** — kebab-case identifier, e.g. `data-mix`, `prompt-search`, `kernel-tune`. Must match `^[a-z][a-z0-9-]*$`. Must not already exist under `tasks/`. Must not be `autoresearch` (existing).
- **one_line_description** — one sentence; goes into the README + manager prompt's "goal" section.
- **metric** — `{key, direction}`. The metric key the manager optimizes (e.g. `val_loss`, `accuracy`, `tokens_per_second`) and direction (`minimize` or `maximize`).
- **source_strategy** — one of:
  - `clone <git-url-or-path>` — clone an existing repo into `tasks/<name>/source/`. The framework's per-experiment sandbox will then clone again from there.
  - `empty` — leave `tasks/<name>/source/` empty (just `.gitkeep`); the user will fill it in later.
  - `copy <local-path>` — copy a directory into `tasks/<name>/source/` and `git init` it.
- **time_budget_seconds** — default 300.
- **gpus_per_worker** — default 1.
- **gpus_total** — default 8 (number of pooled workers).

### 2. Validate

- Reject name conflicts with existing `tasks/<name>/` directories.
- Reject names that don't match the regex.
- If `source_strategy` is `clone`, run `git ls-remote <url>` (or `ls <path>`) to confirm reachability before scaffolding.

### 3. Read the reference task

Read these files from `tasks/autoresearch/` to understand the template shape — do NOT copy them verbatim, copy the *structure*:

- `tasks/autoresearch/README.md` — task description shape
- `tasks/autoresearch/manager-full.js` — workspace generator + manager prompt + worker prompt + config renderer
- `tasks/autoresearch/create-full.js` — CLI wrapper
- `tasks/autoresearch/worker.sh` — bash wrapper that runs `source/` with proper env
- `tasks/autoresearch/sandbox.sh` — per-experiment git clone of `source/`
- `tasks/autoresearch/submit.sbatch` — sbatch entrypoint
- `tasks/autoresearch/tests/smoke.test.js` — basic smoke test for the workspace generator

### 4. Generate the new task

Create `tasks/<name>/` with this structure (mirroring autoresearch):

```
tasks/<name>/
  README.md                  - task description, search surface, metric definition
  source/                    - the runnable code
    (cloned/copied/.gitkeep depending on source_strategy)
  manager.js                 - workspace generator + manager prompt + worker prompt
  create.js                  - CLI wrapper
  worker.sh                  - bash wrapper
  sandbox.sh                 - per-experiment sandbox cloner
  submit.sbatch              - sbatch entrypoint
  tests/
    smoke.test.js            - smoke test for the workspace generator
```

Substitution rules when generating from the autoresearch template:

- Replace `autoresearch` (lowercase) → `<task_name>` everywhere it appears as an identifier (path component, project_id, worker_class). Leave `AUTORESEARCH_*` env vars only where they are framework-specific (e.g. `AUTORESEARCH_TIME_BUDGET_SECONDS` is the framework's wall-clock contract; rename it to `<TASK>_TIME_BUDGET_SECONDS` if the user wants — ask in step 1 if unsure).
- Replace `val_bpb` → `<metric.key>` everywhere. The result-validator path (`infra/hpc_agent/runner/src/result-validator.js`) reads `metrics.json[<metric.key>]`; the new task's wrapper must write that key.
- The manager prompt's "goal" line becomes: `Optimize <metric.key> (<direction>); under a fixed wall-clock budget of <time_budget> seconds per run.`
- The "search surface" section starts as: `<<<TODO: describe what changes between experiments. List the constants/configs/code regions the manager may edit. Examples for autoresearch: DEPTH, ASPECT_RATIO, WINDOW_PATTERN. >>>`. This is intentionally a TODO — the user is expected to fill in the actual search axes.
- `worker.sh` keeps the same env-export + result.txt + metrics.json structure, but the actual `cmd=(...)` line becomes `<<<TODO: invoke your runnable in source/. Must write metrics.json with key '<metric.key>' (a finite number) and result.txt with 'exit_code=<int>'. >>>`.
- `submit.sbatch` keeps the same shape but uses `tasks/<name>/create.js` and `artifact/<name>/run-<timestamp>/`.
- `sandbox.sh` is essentially identical to `tasks/autoresearch/sandbox.sh`; only the default source_root path changes.
- `manager.js` exports a single `create<TaskName>Workspace()` function (PascalCase task name), and a `render<TaskName>Config()`. The renderer functions for the manager prompt and worker prompt embed strict TODO placeholders the user must fill.

### 5. Source code handling

Based on `source_strategy`:

- **clone**: `git clone <url> tasks/<name>/source` (use `--no-hardlinks` for safety). Verify `tasks/<name>/source/.git` exists.
- **copy**: copy contents, then `cd tasks/<name>/source && git init && git add -A && git commit -m "import from <local-path>"`.
- **empty**: create `tasks/<name>/source/.gitkeep` so the directory is tracked, then `cd tasks/<name>/source && git init && git add .gitkeep && git commit -m "<name> task source placeholder"`. The user fills in real code later.

In all three cases, `tasks/<name>/source/` must end up as a git repo (the framework's sandbox script clones from it).

### 6. Verify

After scaffolding:

1. `cd infra/hpc_agent/runner && npm test` — must still pass (the new task's smoke test should be picked up by the existing glob `../../../tasks/*/tests/*.test.js`).
2. Render the workspace into a temp dir to confirm everything wires:
   ```
   TMP=$(mktemp -d)
   node tasks/<name>/create.js --artifact-root "$TMP" --gpu-count <gpus_total> --time-budget-seconds <time_budget>
   ls "$TMP/local/<name>/"
   ```
   Expect `config.yaml`, `projects.yaml`, `state.json` (with `isPaused: false`), `skills/workers/`.
3. Show the user the generated tree and a checklist of TODOs to fill in (mostly in `manager.js`'s prompt strings and `worker.sh`'s command line).

### 7. Commit (only if user explicitly asks)

Do NOT auto-commit the scaffold. The user typically wants to fill in the TODOs first. Tell them how to commit when they're ready:

```
git add tasks/<name>/
git commit -m "Add <name> task plugin"
```

## Conventions to enforce

- **Filenames**: kebab-case, lowercase. e.g. `manager.js`, `worker.sh`, `create.js`, `sandbox.sh`, `smoke.test.js`.
- **Function names**: camelCase. Exported entry point is `create<TaskName>Workspace`.
- **Module-level constants**: `ALL_CAPS_SNAKE`. Paths: `TASK_ROOT`, `REPO_ROOT`, `DEFAULT_WORKER_SCRIPT_PATH`, `DEFAULT_SANDBOX_SCRIPT_PATH`, `DEFAULT_WORKLOAD_ROOT`.
- **Path resolution in JS**: use `__dirname` (resolved from `import.meta.url`) and `path.resolve(__dirname, ...)`. Don't hardcode absolute paths.
- **Path resolution in bash**: `script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`. Default workspace_root via `cd "$script_dir/../.." && pwd`. No absolute repo paths embedded.
- **Output dirs**: `artifact/<task>/run-<timestamp>/` for orchestration state, `artifact/<task>/run-<ts>/local/<task>/repo/experiments/<exp_id>/` for per-run artifacts.
- **TODO markers**: use `<<<TODO: ...>>>` consistently so the user can grep for `<<<TODO` to find what needs filling in.
- **Tests**: a smoke test that calls `create<TaskName>Workspace({ artifactRoot: tmpDir })` and asserts the expected files appear. Mirror `tasks/autoresearch/tests/smoke.test.js`.

## What NOT to do

- Don't modify any framework code under `infra/hpc_agent/runner/`. New tasks plug in via files in `tasks/<name>/` only.
- Don't copy autoresearch-specific prompt content (e.g., `train.py` references, `DEPTH`/`WINDOW_PATTERN` knobs, `val_bpb`). The new task's manager prompt is a fresh canvas for the user.
- Don't try to be clever about the search surface — leave it as TODO placeholders. The user knows their domain better than the scaffolder does.
- Don't commit. Hand the scaffold to the user with a TODO checklist.
