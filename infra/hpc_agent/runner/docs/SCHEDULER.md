# Resource-Aware Scheduler

日期：2026-04-25

## 当前模型

Runner 现在使用一个本地 control plane 管理多 agent：

- 一个 manager 负责规划 `TASK_GRAPH`
- 多个 worker 通过 `worker_class` 组成 worker pool
- 一个 Slurm shared allocation 提供固定 GPU token 池
- scheduler 负责把 ready task 绑定到空闲 worker 和可用 GPU token

默认 autoresearch smoke 会按 8 张 GPU 生成 8 个实验 worker：

- `exp_runner_0` 到 `exp_runner_7`
- `worker_class: experiment_runner`
- `maxConcurrentWorkers: 8`

## 任务提交格式

Manager 推荐提交 `TASK_GRAPH`，而不是把任务硬编码到某个 agent：

```json
{
  "tasks": [
    {
      "id": "exp_lr_1",
      "worker_class": "experiment_runner",
      "task": "run experiment",
      "resources": { "gpus": 1, "cpus": 1 },
      "priority": 2,
      "utility": 1.5,
      "estimated_runtime_seconds": 300,
      "produces_tags": ["metrics:lr_1"]
    },
    {
      "id": "analyze_wave",
      "worker_class": "analyst",
      "task": "summarize metrics and propose next wave",
      "depends_on_tags": ["metrics:lr_1"],
      "replan_after": true
    }
  ]
}
```

`agent` 仍然可用，但只应该用于必须指定某个具体 worker 的任务。

## 调度流程

一次 harness 任务的执行流程：

1. Runner 进入 `single_manager` 模式，加载 manager 和 worker pool。
2. Runner 把 worker roster、GPU token 状态、target utilization 注入 manager context。
3. Manager 输出 `TASK_GRAPH` 或 `PROJECT_COMPLETE`。
4. Runner 解析 DAG，维护每个 task 的 runtime state。
5. Scheduler 找到 dependency 和 tag 都满足的 ready tasks。
6. Scheduler 只启动同时满足以下条件的 task：
   - 有空闲 worker 或匹配的 `worker_class`
   - GPU token 足够
   - 没有失败依赖
7. Worker 启动前，runner 在 DB 里创建 token request 并 grant 具体 GPU token。
8. Worker 收到 grant context，使用指定 GPU 跑 `run-autoresearch-worker.sh` 或等价命令。
9. Worker 结束后，runner release token，记录 task completion / failure。
10. DAG drain 后，如果没有失败且 `refillOnGraphDrain=true`，runner 让 manager 再规划下一轮。

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
- 一个 active grant 绑定一个或多个 `gpuN`
- worker 结束后自动 release grant
- `grantRequiresLease: false`：GPU task 不依赖 Slurm allocation lease，runner 直接在 compute node 上跑

CPU/RAM 目前只作为 task metadata 和 command hint，还不是强约束 token。

## Orchestration 配置项

`config.yaml` 的 `orchestration` 块支持以下字段（在 `normalizeOrchestrationConfig` 里 normalize）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `mode` | `phase_managers` | `single_manager` 启用 DAG 模式 |
| `manager` | `manager` | manager agent 的名字 |
| `maxConcurrentWorkers` | `8` | 最大并发 worker 数 |
| `maxManagerPasses` | `8` | 单次 cycle 最多调用 manager 次数 |
| `maxWallClockSeconds` | `null` | cycle 总时长上限（秒），超出则停止 manager loop |
| `refillOnGraphDrain` | `true` | DAG 跑完后是否让 manager 再规划 |
| `liveReplanOnTaskComplete` | `false` | worker 完成时立即触发 manager replan |
| `liveReplanMinIntervalSeconds` | `0` | live replan 的最小触发间隔 |
| `targetGpuUtilization` | `1` | 目标 GPU 利用率，影响 backfill 策略 |

## 当前边界

已经实现：

- 8-worker pool generation
- `worker_class` 动态绑定
- DAG dependency 和 tag gate
- tag producer 唯一性校验（同一 tag 不能被两个 task 同时 produce）
- GPU token admission（`grantRequiresLease: false`，不需要 Slurm lease）
- ready queue ranking（critical path 计算包含 tag 依赖）
- graph-drain replan
- live replan queue（worker 完成时追加 replan，不丢失并发完成的 task）
- TASK_GRAPH parse error reporting（parse 失败时输出具体错误位置）
- `maxWallClockSeconds` wall-clock kill switch
- scheduler unit tests

还没实现：

- CPU/RAM token
- aging / starvation prevention
- preemption
- failure recovery 后的 orphan process/token sweep
