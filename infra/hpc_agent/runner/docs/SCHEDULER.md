# Resource-Aware Scheduler

最近更新：2026-04-27

## 当前模型

Runner 用一个本地 control plane 管理一个 LLM manager 和一个 worker pool：

- 一个 **manager** 规划 `TASK_GRAPH`，每个 task 选择一种 `execution_mode`
- 多个 **worker**（通过 `worker_class` 组成 pool）作为 LLM 兜底执行路径
- 一个 **direct executor**（非 LLM）作为 `param_patch` 任务的默认执行路径
- 一个 **Slurm shared allocation** 提供固定 GPU token 池
- **Scheduler** 负责把 ready task 绑定到空闲资源（worker 或直接执行）和可用 GPU token

默认 autoresearch 配置在 8 张 GPU 上生成 8 个实验 worker（用于 `code_edit`
/ `llm_repair` 路径）：

- `exp_runner_0` 到 `exp_runner_7`
- `worker_class: experiment_runner`
- `maxConcurrentWorkers: 8`

`param_patch` 任务不占用 LLM worker name，只占用 GPU token，所以它和
worker-class 任务可以混合调度。

## 任务提交格式

Manager 提交 `TASK_GRAPH`，每个 task 显式选择 `execution_mode`：

```json
{
  "tasks": [
    {
      "id": "exp_lr_03",
      "worker_class": "experiment_runner",
      "execution_mode": "param_patch",
      "base_ref": "HEAD",
      "rationale": "测试 LR=3e-3，最近 baseline 在 5e-3 表现弱",
      "task": "sweep matrix_lr to 3e-3",
      "edits": [
        {"file": "train.py", "kind": "constant_replace",
         "name": "MATRIX_LR", "expected_old_repr": "0.005", "new_repr": "0.003"}
      ],
      "resources": {"gpus": 1, "cpus": 1},
      "priority": 2,
      "estimated_runtime_seconds": 320,
      "produces_tags": ["metrics:lr_03"],
      "early_stop": {"check_at_seconds": 90, "abort_if_loss_above": 4.0}
    },
    {
      "id": "exp_arch_attn",
      "worker_class": "experiment_runner",
      "execution_mode": "code_edit",
      "task": "在 train.py 中将 SDPA 替换为 sliding-window attention，window=128",
      "rationale": "局部 attention 在长序列上可能省内存换更大 batch",
      "resources": {"gpus": 1, "cpus": 1},
      "produces_tags": ["metrics:attn_local"]
    },
    {
      "id": "analyze_wave",
      "worker_class": "analyst",
      "task": "summarize the wave; recommend the next axis",
      "depends_on_tags": ["metrics:lr_03", "metrics:attn_local"],
      "replan_after": true
    }
  ]
}
```

`agent` 仍然可用，但只应该用于必须指定具体 worker 的任务（比如分析师 maya）。

### 字段速查

| 字段 | 含义 |
|---|---|
| `execution_mode` | `code_edit`（默认，走 LLM worker） / `param_patch`（直接执行器，没有 worker LLM） / `llm_repair`（手动触发的 LLM 修复任务） |
| `edits[]` | `execution_mode=param_patch` 必填。每条 edit 必须有 `file` + `kind`（`constant_replace` / `regex_replace` / `block_replace` / `unified_diff`）+ kind-specific 字段 |
| `base_ref` | param_patch 才用。基线源仓库的 ref（默认 `HEAD`）。**不要**用其他实验的 id —— 实验 sandbox 是独立 clone，子实验不能跨 sandbox fork |
| `rationale` | 必写，一句话说明假设和预期。下次 manager 唤醒时会在 ledger 里看到 `rationale → outcome`，这是它跨 cycle 的"记忆" |
| `early_stop` | 可选 `{check_at_seconds, abort_if_loss_above}`。wrapper 在 `check_at_seconds` 时观察 train.log 的最新 `loss: X.XX`，超过阈值就 SIGTERM 训练进程组 |
| `priority` / `utility` / `estimated_runtime_seconds` | scheduler 排序与 walltime admission 输入 |
| `depends_on` / `depends_on_tags` / `produces_tags` | DAG dependency gate |
| `replan_after` | 任务完成后强制 manager 立刻 replan（屏障） |

## 调度流程

1. Runner 进入 `single_manager` 模式，加载 manager 和 worker pool。
2. Runner 把 worker roster、GPU token 状态、target utilization、**实验 ledger**
   （top-K + recent + failed clusters，每条带 rationale 和 edit summary）
   注入 manager context。
3. Manager 输出 `TASK_GRAPH`、`KILL_TASKS`、或 `PROJECT_COMPLETE`。
4. Runner 解析 DAG，维护每个 task 的 runtime state。
5. Scheduler 找到 dependency 和 tag 都满足的 ready tasks。
6. Scheduler 只启动同时满足以下条件的 task：
   - 有空闲 worker 或匹配的 `worker_class`（`param_patch` 走直接执行器，跳过这一步）
   - GPU token 足够
   - 没有失败依赖
   - **walltime admission gate**：剩余 sbatch wall ≥ `estimated_runtime_seconds + 120s`
7. Worker 启动前，runner 在 DB 里创建 token request 并 grant 具体 GPU token。
8. 根据 `execution_mode` 分支：
   - **`param_patch`** → `_startDirectExecutorTask`：clone 源仓库到 sandbox →
     SHA-pin parent → 应用 `edits[]` → ast.parse 校验 → commit → spawn
     `tasks/<name>/worker.sh` (`detached: true`，独立 process group) → 等待并验证
     `result.txt` + `metrics.json`
   - **其他** → `_runWorkerStepWithRetries`：通过 LLM CLI（`claude_cli` /
     `codex_cli` / `api`）调起 worker LLM，由它去跑 sandbox.sh + worker.sh
9. 三层超时同时存在：
   - **inner**：`worker.sh` 用 `timeout --kill-after=15s` 包训练命令
   - **outer**：orchestrator 的 900s 硬上限通过 `setTimeout` + `abortController`
   - **early stop**（可选）：manager 在 task 上设阈值，wrapper 后台 watcher
     在 `check_at_seconds` 时观察 train.log 的 `loss: X.XX`
10. 任务结束后：runner release token；如果是 killed-by-timeout，先 sleep 5s
    并跑 `nvidia-smi --query-compute-apps` 探针，确认没有进程仍持有 CUDA 上下文。
11. **Watermark live replan**：每次 worker 完成都 push 进 replan-trigger 队列，
    但只有 `ready + running < ⌈1.5 × maxConcurrentWorkers⌉` 时才真叫 manager；
    否则 task id 累积到下次满足条件时一起带进去。配合
    `liveReplanMinIntervalSeconds`（默认 30s）做最小间隔保护。
12. DAG drain 后，如果没有失败且 `refillOnGraphDrain=true`，runner 让 manager 再规划下一轮。

## 排序策略

当前 ready queue 排序规则在 `src/scheduler.js`：

1. 更高 `priority`
2. 更长 critical path
3. 更高 `utility`
4. 更大的 GPU demand
5. 更短 `estimated_runtime_seconds`

这个策略的目标是：

- 长依赖链先启动，减少 tail latency
- GPU 能填满时尽量填满
- 小任务可在资源碎片里 backfill
- manager 可以用 `priority` 和 `utility` 表达策略偏好

## 资源语义

当前实现的资源是 GPU-first：

- GPU token 是独占的
- 一个 active grant 绑定一个或多个 `gpuN` token name
- token name 映射到 `CUDA_VISIBLE_DEVICES` 整数（`gpu3` → `3`）
- worker 结束后自动 release grant
- `grantRequiresLease: false`：GPU task 不依赖 Slurm allocation lease，
  runner 直接在 compute node 上跑

CPU/RAM 目前只作为 task metadata 和 command hint，还不是强约束 token。

## Orchestration 配置项

`config.yaml` 的 `orchestration` 块支持以下字段（在 `normalizeOrchestrationConfig` 里 normalize）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `mode` | `phase_managers` | `single_manager` 启用 DAG 模式 |
| `manager` | `manager` | manager agent 的名字 |
| `maxConcurrentWorkers` | `8` | 最大并发 worker 数；同时也是 watermark 的基数 |
| `maxManagerPasses` | `8` | 单次 cycle 最多调用 manager 次数 |
| `maxWallClockSeconds` | `null` | cycle 总时长上限（秒），超出则停止 manager loop |
| `refillOnGraphDrain` | `true` | DAG 跑完后是否让 manager 再规划 |
| `liveReplanOnTaskComplete` | `false`（autoresearch 默认 `true`） | worker 完成时尝试触发 manager replan |
| `liveReplanMinIntervalSeconds` | `0`（autoresearch 默认 `30`） | live replan 的最小触发间隔 |
| `targetGpuUtilization` | `1` | 目标 GPU 利用率，影响 backfill 策略 |

`directExecutor` 是 task 插件级别的 config 块（不在 `orchestration` 里）：

```yaml
directExecutor:
  enabled: true
  sourceRepoPath: <abs path to tasks/<name>/source>
  wrapperScript: <abs path to tasks/<name>/worker.sh>
  sandboxRoot: <abs path; per-experiment clones land at sandboxRoot/<id>>
  outputRoot: <abs path; per-experiment outputs land at outputRoot/<id>>
  metricKey: val_bpb
  timeBudgetSeconds: 300
  hardCapSeconds: 900
  envOverrides:
    AUTORESEARCH_RESULTS_PATH: <abs path>
    AUTORESEARCH_SHARED_CACHE_ROOT: <abs path>
```

由 `normalizeDirectExecutorConfig` 校验。`param_patch` 任务需要这个块；
否则 dispatch 会以 `directExecutor not configured` 阻塞。

## 当前边界

已经实现：

- 8-worker pool generation
- `worker_class` 动态绑定
- `execution_mode` 分支：`param_patch` 走直接执行器，其他走 LLM worker
- 4 种 structured edits + Python AST 归一化（`524288` 匹配 `2 ** 19`）
- 直接执行器：clone + SHA-pin + 原子 edits + ast.parse 校验 + commit + spawn
- DAG dependency 和 tag gate
- tag producer 唯一性校验（同一 tag 不能被两个 task 同时 produce）
- GPU token admission（`grantRequiresLease: false`）
- ready queue ranking（critical path 包含 tag 依赖）
- graph-drain replan
- watermark-gated live replan + 最小间隔
- TASK_GRAPH parse error reporting + per-edit-kind 校验
- `maxWallClockSeconds` wall-clock kill switch
- 三层超时（worker.sh inner / 900s outer / manager-decided early-stop）
- Slurm walltime admission gate
- failure.json 结构化失败记录
- canonical result validation + LLM-claim mismatch 日志
- 持久化 `torch.compile` 共享缓存（跨 sbatch）
- 实验 ledger（rationale + edit + 结果）注入 manager context
- 92 个单元 / 集成测试

还没实现：

- CPU/RAM token
- aging / starvation prevention
- preemption
- 多节点 scheduling（current goal: 单节点 duty cycle ≥75% 后再做）
