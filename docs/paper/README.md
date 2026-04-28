# ALPS / RAILS Workshop Paper Draft

`icml-workshop.tex` describes **ALPS** (Adaptive Lineage-Aware Parallel
Search) — a scheduling policy for LLM-driven optimization — running on
top of **RAILS**, a runtime that exposes the structured edits, git
lineage, and ledger primitives ALPS needs.

The two are layered:

- **ALPS = method.** The policy: which lineage node to expand (UCT),
  which operator to apply (bandit), how much to speculate (commit
  hazard), when to commit (statistical gate), how to compose batches
  (diversity-aware acquisition), how to handle stale winners (rebase
  validation).
- **RAILS = runtime.** The substrate: structured `param_patch` edits
  with AST normalization, git-backed source lineage, watermark-gated
  replanning, rationale ledger, defense-in-depth timeouts, task-plugin
  contract.

## Build

```bash
cd docs/paper
pdflatex icml-workshop.tex
bibtex   icml-workshop
pdflatex icml-workshop.tex
pdflatex icml-workshop.tex
```

`icml2026.sty` must be in the same directory. If unavailable, swap the
preamble for vanilla `article` style; the document is otherwise
package-agnostic.

## Outline

1. **Introduction.** Problem framing (agent-led search), the
   serial/naive-parallel polarity, and the contribution: a scheduling
   policy, not richer primitives.
2. **Background and Related Work.** LLMs as optimizers, MCTS/UCT,
   parallel HPO (Hyperband, ASHA, PBT), batch BO, autonomous research,
   data-mixture work.
3. **Agent-Led Parallel Search as Scheduling.** Setup; the five
   scheduling decisions a parallel agent loop makes; **commit hazard**
   formalized.
4. **ALPS.** Lineage UCT (Eq. 2), operator bandit (Eq. 3), adaptive
   parallelism via commit hazard (Eqs. 1+4), diversity-aware batch
   acquisition (Eq. 5), statistical commit gate (Eq. 6), stale-winner
   rebase, full Algorithm 1.
5. **RAILS Runtime.** Structured edits, git-backed lineage, watermark,
   ledger, timeouts, plugin contract.
6. **Experiments.** Three policies (Serial, Naive parallel, Manual
   ALPS) on autoresearch; planned Qwen-SFT data-mix; planned ablations
   that swap each manager-driven ALPS decision for an explicit code
   path.
7. **Discussion.** Why the LLM is a conservative default scheduler;
   coverage as code, not prompt; noise estimation; cross-task operator
   priors.
8. **Limitations / Conclusion.**

## Factual notes

- **Manual ALPS** is the current pilot configuration. The runtime
  exposes the lineage, watermark, and rationale primitives; the LLM
  manager plays the scheduler role through prompt instructions. The
  paper explicitly distinguishes this from **Explicit ALPS** (operator
  posteriors, hazard-driven $Q_t$, statistical commit gate as code,
  diversity acquisition as code), which is partially implemented and
  marked as in progress in the experimental tables.
- **Pilot autoresearch table** (Table 1):
  - Serial: job 1580141, Karpathy single-agent, 53 min, 8 evals, best 0.9894.
  - Naive parallel: job 1579909, footnoted because it ran with an
    unrelated sandbox-isolation bug since fixed; 27 evals, best 0.9928.
    Conservative as evidence — a clean naive-parallel run cannot
    produce cumulative wins by construction.
  - Manual ALPS: job 1581726, post-fix, 65 evals, best 0.9920, one
    KEEP at $t \approx 55$ min.
  - Explicit ALPS: TBD, in progress.
- **Pilot Qwen-SFT** (Table 2): in flight (job 1581949 queued behind
  cluster QOS). All rows TBD.
- **Ablation table** (Table 3): only "Manual ALPS full" and "−lineage"
  rows populated; the rest are dedicated 1-h runs that swap the
  manager's role in each ALPS decision for explicit code.

## Open work before camera-ready

1. **Implement explicit operator posteriors.** Maintain $\{\mu_o,
   \sigma_o\}$ in code; update from `picked` events; surface in the
   manager prompt.
2. **Implement statistical commit gate** (Eq. 6) as a hard rule in the
   `KEEP_EXPERIMENT` handler, with $\hat\sigma$ estimated from baseline
   repeats.
3. **Implement hazard-driven $Q_t$** (Eq. 4) replacing the fixed
   watermark $\alpha$.
4. **Implement diversity-aware batch acquisition** (Eq. 5) by
   penalizing same-operator candidates within a batch.
5. **Run noise-estimation reps** so $\hat\sigma$ in the gate is
   empirically grounded.
6. **Run the planned ablations** (Tables 2--3).
7. **Replace anonymous author block** for camera-ready.
