# ICML Workshop Paper Draft

`icml-workshop.tex` describes the system design of the concurrent
search-task framework, with autoresearch as the running example. It
compiles standalone with vanilla LaTeX (no ICML style file needed):

```bash
cd docs/paper
pdflatex icml-workshop.tex
pdflatex icml-workshop.tex   # second pass for refs
```

For the actual ICML submission:
1. Drop `icml2024.sty` (or whichever year) into this directory.
2. Switch the `\documentclass` line per the comment at the top of the
   `.tex` file.
3. Replace the inline `thebibliography` block with a real `references.bib`.

## Outline (matches the .tex)

1. **Introduction** — autoresearch context; naive parallelization fails to
   convert throughput into research output.
2. **Background** — the serial loop's three structural properties (cumulative
   branching, per-result feedback, code reading) that naive fan-out breaks.
3. **System Design** — manager + worker pool + scheduler; two execution
   modes (`param_patch` deterministic / `code_edit` LLM); cumulative
   `KEEP_EXPERIMENT` lineage; watermark-gated live replan; manager context
   composition; defense-in-depth timeout rings.
4. **Empirical Evaluation** — 1-hour matched wall-time comparison among
   Karpathy serial / naive parallel / cumulative parallel.
5. **Lessons Learned** — six concrete bug-and-fix vignettes (sandbox
   isolation, weka filesystem latency, watermark tuning, principle-driven
   vs threshold-driven prompts, code injection, manager-decided race
   handling).
6. **Related Work** — autoresearch, multi-agent code editors, tree-search
   planners, data-mix search literature.
7. **Conclusion and Future Work** — multi-seed averaging, axis-fixation
   detection, cross-task lineage portability, Qwen-SFT data-mix plugin.

## Open items

- Author + affiliation block is a placeholder (`<email>`, `Affiliation TBD`).
- `references.bib` is not yet split out; the inline `thebibliography` block
  has 5 placeholder entries.
- Acknowledgements section is empty.
- The numeric table in §4 reflects three actual runs (job 1580141 serial,
  job 1579909 naive parallel, job 1581726 cumulative parallel) but should
  ideally include error bars from re-running each config 2-3 times.
