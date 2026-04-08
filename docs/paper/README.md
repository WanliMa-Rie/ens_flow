# EnsFlow — NeurIPS paper draft

## Files
- `main.tex`  — the full paper draft (intro, method, experiments, appendix).
- `refs.bib`  — bibliography (placeholders for several references; verify before submission).
- `README.md` — this file.

## Building
```bash
cd docs/paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
or with `latexmk`:
```bash
latexmk -pdf main.tex
```

The draft currently uses the `article` class with `natbib` (author-year citations).
Replace `\documentclass{article}` and drop in `neurips_2026.sty` when the
official style file is available.

## Status of the draft (2026-04-08)
- **Story (Intro + Related Work)**: complete, ready for co-author read.
- **Method**: complete. Propositions 1–3 and Corollary 1 are stated and
  proved in-line and in Appendix A.
- **Experiments**: protocol fully specified; result tables are empty shells
  with `\todo{}` markers. Baselines and metrics are locked to the existing
  `rna_backbone_design/analysis/metrics.py`.
- **Appendix**: proofs, architecture details, B-factor normalization,
  hyperparameters, and a milestone checklist for when compute returns.
- **Remaining work**: (1) run experiments once GPUs return and fill in
  `\todo{}` markers; (2) verify every `.bib` entry; (3) add a qualitative
  figure (α-heatmap + B-factor overlay) once trained.

## Narrative cheat sheet (for quickly re-pitching the paper)
1. **Problem**: RNA ensembles are an open ML-for-bio problem; every existing
   model outputs single structures or uses global-in-time noise.
2. **Insight**: Flexibility is local. We need a learned heteroscedastic
   uncertainty field on SE(3)^L.
3. **Technical core**: Heteroscedastic NLL on drift residuals → closed-form
   optimum σ\* = ‖r‖ (Prop. 1). Under Bayes-optimal drift, this equals the
   trace of the clean-state posterior covariance (Cor. 1).
4. **Physical anchor**: B-factors (crystallographic ADPs) give an independent
   physical target for the learned α (Level 3).
5. **Clean hierarchy**: Level 1 (ODE) and fixed-width FoldFlow-SFM both arise
   as strict degenerate limits of Level 2 (Prop. 3).
6. **Falsifiable qualitative claim**: Even in Level 2 (no B-factor
   supervision), the learned α should correlate with held-out B-factors.
   This is the headline ablation.
