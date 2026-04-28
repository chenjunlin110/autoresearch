# RAILS Workshop Paper Draft

`icml-workshop.tex` describes the system + method for **RAILS**
(Resource-Aware Information Lineage for parallel LLM agent Search).
Compiled with `icml2026.sty` (which the user must obtain from the ICML
2026 workshop submission package and place next to the `.tex`).

```bash
cd docs/paper
pdflatex icml-workshop.tex
bibtex   icml-workshop
pdflatex icml-workshop.tex   # second pass for refs
pdflatex icml-workshop.tex   # final pass
```

If `icml2026.sty` is not yet available, change the line
`\usepackage{icml2026}` and the corresponding `\twocolumn[…]` /
`\icmltitle` / `\icmlauthorlist` block to a vanilla article preamble;
the rest of the document is style-package-agnostic.

## Outline

1. **Introduction** — agent-led search as a class of optimization problems;
   serial vs. naive-parallel as the two endpoints; three contributions
   (debt formalization, RAILS framework, two-task experimental protocol).
2. **Agent-Led Search and Information Debt** — formalization
   ($b_t, u_t, s_t, y_t, \pi_\theta, H_t$); three measurable debts
   (state, lineage, coverage).
3. **RAILS** — three mechanisms paying down each debt independently:
   watermark-gated replanning ($\alpha$ knob), cumulative lineage
   (`KEEP_EXPERIMENT` + rebase validation + statistical commit gate),
   rationale/coverage ledger; `param_patch` vs `code_edit`
   execution modes; task-plugin contract; Algorithm 1.
4. **Experiments** — three policies (Serial, Naive parallel, RAILS)
   on two tasks (autoresearch, Qwen-SFT data-mix) at matched 1-h wall.
   Pilot table populated; SFT and full ablation tables marked TBD.
5. **Related Work** — LLM-as-optimizer, autonomous research,
   parallel HPO, data-mix optimization.
6. **Discussion / Limitations / Conclusion**.

## Factual notes

- **Naive-parallel data point** in Table 1 is from a configuration that
  also exhibited a sandbox-isolation bug (later fixed). A footnote in
  the table acknowledges this and notes a clean re-run is planned for
  camera-ready. The conclusion (no cumulative wins → no compounding
  improvement) is robust to the bug.
- **RAILS pilot** (Table 1, last row) is from job 1581726, post-fix,
  with KEEP_EXPERIMENT enabled. 65 evals, best 0.9920, exactly one
  KEEP issued at $t \approx 55$ min.
- **Karpathy serial baseline** (Table 1, first row) is job 1580141,
  53-min wall (early termination), 8 evals, 4 keeps, best 0.9894.
- **Qwen-SFT row** (Table 2) is currently in flight (job 1581949,
  4-h sbatch with Qwen3-0.6B, queued behind the QOS limit). Manifest
  on disk, prep done.
- **Ablation** (Table 3): only the "remove cumulative lineage" row is
  filled (= naive parallel). The other rows require dedicated 1-h runs
  per ablated channel; these are pending.

## Open items before camera-ready

- Implement the **statistical commit gate** (Eq. 2) as a tool
  suggestion in the manager prompt + log gate violations.
- Implement the **coverage manifest** (axis, values tested) as an
  injection into the manager's per-cycle context.
- Implement **automatic rebase validation** for stale winners.
- Run **N=3 seed reps** of the pilot autoresearch comparison so
  numbers come with error bars.
- Replace the `[Anonymous Authors]` block with the real author list.
