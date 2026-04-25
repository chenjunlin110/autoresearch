# Codex CLI Runtime

日期：2026-04-23

## 当前状态

`infra/hpc_agent/runner` 现在已经支持两种 agent runtime：

- `codex_cli`：默认本地运行时
- `api`：兼容旧的 provider/key pool/OAuth 路径

默认值已经改成：

```yaml
agentRuntime: codex_cli
```

## 为什么现在不再默认要求 API key

`train.py` 本来就不需要 API key。  
之前真正卡住的是 manager/worker agent 只能走 `runAgentWithAPI()`。

现在已经补了 `runAgentWithCodexCLI()`，所以本地项目可以直接起 `codex exec` 来跑 agent，而不是先解析 provider token。

结果是：

- `agentRuntime: codex_cli` 不再要求 key pool 非空
- `agentRuntime: api` 仍然保留原有 key 检查

## 实现内容

本次已经落地的改动：

1. `src/server.js`
   - 新增 `agentRuntime` 配置，默认 `codex_cli`
   - 新增 runtime 选择逻辑
   - 只有 `api` runtime 才做 “无 key 就 pause” 的 preflight
   - prompt 中加入了 `codex_cli` 运行时说明

2. `src/agent-runner.js`
   - 新增 `runAgentWithCodexCLI()`
   - 通过本地 `codex exec` 非交互执行 agent
   - 使用 `--output-last-message` 回收最终回复
   - 保留现有 report/log 生命周期

3. 文档和 setup
   - setup 不再默认提示“先填 API key”
   - `README` 和 `.env.example` 都改成 `codex_cli` 优先

## Codex CLI 调用方式

当前实现使用的核心形式是：

```bash
codex exec \
  --color never \
  --ignore-user-config \
  --ignore-rules \
  --dangerously-bypass-approvals-and-sandbox \
  --output-last-message <file> \
  -C <repo> \
  --add-dir <project_data_dir> \
  -
```

说明：

- 主工作目录是 repo 根目录
- project data 目录通过 `--add-dir` 加入可写集合
- prompt 通过 stdin 传入
- 最终 agent 回复通过 `--output-last-message` 读取

## 模型选择

`codex_cli` runtime 下：

- 如果配置是 OpenAI/Codex 模型名，会传给 `codex exec -m`
- 如果配置只是抽象 tier（如 `high` / `mid`），则不强行传模型，交给本地 Codex 默认配置

## 已验证

本机已确认：

```bash
codex --version
# codex-cli 0.123.0
```

并且已经做过直接烟测：

- `runAgentWithCodexCLI()` 成功返回
- 最终消息能正确落回 runner
- `npm test` 通过

还做过一轮真实 runner 项目烟测：

- `agentRuntime: codex_cli`
- 不配置任何 provider API key
- 在临时 `TBC_HOME` 下启动真实 `src/server.js`
- 通过项目 `state.json` 恢复一个待执行 worker schedule
- runner 日志确认进入 `Using Codex CLI runner for smoke (model: mid)`
- worker 在 repo 根目录成功写出 `runtime-smoke.txt`
- 文件内容为 `CODEX_CLI_PROJECT_OK`

## 保留的兼容路径

`api` runtime 仍然保留，适合：

- 需要 provider/key pool/OAuth 的旧项目
- 需要显式切换外部 provider 的场景

如果项目要继续走旧路径，只要在 `config.yaml` 里写：

```yaml
agentRuntime: api
```

## 已知边界

`codex_cli` runtime 已经可用，但它和旧 API runtime 不是同一种工具模型：

- API runtime 里的 `HpcSubmit` 是 Node 侧 JSON tool
- `codex_cli` runtime 下更偏向直接使用 shell

所以当前 prompt 已经补充说明：如果拿到了 `grant_id` / `lease_job_id` / `gpu_tokens`，就直接据此构造 `srun --overlap` 命令。

## 一句话

现在这套 runner 已经不是“必须先配 API key 才能跑 agent”了。  
本地默认就是 `codex_cli` runtime。
