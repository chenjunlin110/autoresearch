# Claude Code Runtime

日期：2026-04-25

## 当前状态

`infra/hpc_agent/runner` 现在支持三种 agent runtime：

- `codex_cli`：默认本地运行时（OpenAI Codex CLI）
- `claude_cli`：Claude Code CLI 本地运行时（本文档覆盖）
- `api`：兼容旧的 provider / key pool / OAuth 路径

只要本机装了 `claude`（Claude Code CLI），把项目的 `config.yaml` 改成：

```yaml
agentRuntime: claude_cli
```

或者在创建 autoresearch 工作区时传 `--agent-runtime claude_cli`，就会走 `claude -p` 非交互路径。

## 为什么加这个

- 现场可能只有 Claude Code CLI，没有 Codex CLI
- 同一套 manager / worker skill 在 Claude Code 下也要能跑，不想再起一个兼容层
- Anthropic 的缓存友好，长 manager context 走 Claude 代价更低

## Claude CLI 调用方式

当前实现的核心形式（见 `src/agent-runner.js:buildClaudeExecInvocation`）：

```bash
claude \
  --print \
  --dangerously-skip-permissions \
  --output-format text \
  --add-dir <repo> \
  --add-dir <project_data_dir> \
  [--model <claude-model>] \
  -
```

说明：

- 主工作目录是 repo 根目录（`cwd`）
- project data 目录通过 `--add-dir` 加入可写集合
- prompt 通过 stdin 传入（`child.stdin.end(prompt + '\n')`）
- 最终 agent 回复从 stdout 回收（`--print` 模式）
- `--dangerously-skip-permissions` 跳过 Claude Code 的交互式授权弹窗，契合 runner 的 headless 运行场景

## 模型选择

`claude_cli` runtime 下：

- 如果配置里是 Claude 模型名（`claude-*`、`anthropic/...`、`claude/...`），会传给 `claude --model`
- 如果配置只是抽象 tier（`high` / `mid` / `low`），则不强行传模型，交给本地 Claude Code 默认配置
- Codex 模型名（`gpt-*`、`openai/...`）会被忽略（`resolveClaudeCliModel` 返回 `null`）

## 使用方式

本地 dev：

```bash
claude --version  # 确认已安装

# 单个项目覆盖
cat > ~/.hpc_agent/local/my-project/config.yaml <<'EOF'
agentRuntime: claude_cli
EOF
```

autoresearch 整套工作区：

```bash
node infra/hpc_agent/runner/scripts/create-autoresearch-dag-full.js \
  --artifact-root ./artifacts/claude-run \
  --project-id autoresearch-dag-full \
  --gpu-count 8 \
  --experiment-worker-count 8 \
  --time-budget-seconds 300 \
  --agent-runtime claude_cli
```

sbatch 入口：

```bash
AUTORESEARCH_AGENT_RUNTIME=claude_cli sbatch infra/hpc_agent/runner/scripts/submit-autoresearch-runner.sbatch
```

## 和 Codex CLI 的区别

| 维度 | codex_cli | claude_cli |
|---|---|---|
| 二进制 | `codex` | `claude` |
| 非交互 flag | `exec --dangerously-bypass-approvals-and-sandbox` | `--print --dangerously-skip-permissions` |
| 输出回收 | `--output-last-message <file>` → 读文件回落 stdout | stdout（保留了 outputFile 钩子，便于未来切到 `--output-format json`） |
| 模型前缀 | `openai-codex/`, `openai/` | `anthropic/`, `claude/` |
| 对 git/gh 限制 | 由 runner 注入 sandbox policy | 由 Claude Code 自身 permission 体系处理；runner 通过 skill 提示补充规则 |

两个 runtime 共享同一批 manager/worker skill 文件、同一份 `TASK_GRAPH` 协议、同一套 resource grant 注入逻辑。切换只改 `agentRuntime` 一个字段。

## 已知边界

- `claude --model` 的可选值由本地 Claude Code 版本决定；抽象 tier 由 CLI 自己解析
- cost / usage 目前不会回填（和 `codex_cli` 一致；只有 `api` runtime 有真实 token 账目）
- prompt 里对运行时的提示语和 Codex 不同，以让 agent 知道要走 Claude Code 的 Bash / Read / Write / Edit 工具，不要尝试 JSON tool protocol

## 一句话

只要 PATH 里有 `claude`，改一行 `agentRuntime: claude_cli` 就能把整套 runner 跑在 Claude Code 上，无需其它改动。
