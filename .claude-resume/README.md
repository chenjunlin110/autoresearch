# Claude Code resume bundle

Files in this directory are imported into `~/.claude/` on a new machine so a
`claude --resume <session-id>` picks up the framework v1.5 redesign exactly
where it left off.

Contents:
- `redesign-plan.md` — the approved redesign plan (Phases 0–6)
- `session.jsonl` — the full Claude conversation transcript
- `restore.sh` — copies both into `~/.claude/` at the right paths

## Usage on the new machine

```bash
git clone git@github.com:chenjunlin110/autoresearch.git
cd autoresearch
git checkout framework
bash .claude-resume/restore.sh
claude --resume fd6465ef-7937-4ad1-8942-08b698990432
```

`restore.sh` writes:
- `redesign-plan.md` → `~/.claude/plans/autoresearch-hpc-agent-kind-zephyr.md`
- `session.jsonl` → `~/.claude/projects/<hashed-cwd>/fd6465ef-7937-4ad1-8942-08b698990432.jsonl`

The hashed-cwd is whatever directory you cloned into. The script auto-derives
it from the current path so it works anywhere, not just on
`/mnt/weka/home/junlin.chen/workspace/autoresearch`.
