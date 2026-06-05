# REVIEW V4 SUMMARY

## Raw Baseline

Oracle/full-truth diagnostic from V2:
- Top200 = 16
- Top500 = 28
- Top1000 = 48

Held-out fixed raw baseline (`inverted_base_motif_structure_score + top3_mean_score`):
- Top200: mean overlap = 4.90, mean precision = 0.0245, mean fold_enrichment = 15.96
- Top500: mean overlap = 8.70, mean precision = 0.0174, mean fold_enrichment = 11.34
- Top1000: mean overlap = 15.10, mean precision = 0.0151, mean fold_enrichment = 9.84

## Multi-objective Calibration

Held-out Top200 comparison by objective:
- top1000_hypergeom: mean overlap = 4.20, precision = 0.0210, fold = 13.68, median p = 1.405e-04
- top500_hypergeom: mean overlap = 4.10, precision = 0.0205, fold = 13.36, median p = 1.405e-04
- weighted_top200_500_1000: mean overlap = 4.10, precision = 0.0205, fold = 13.36, median p = 2.659e-04
- top200_overlap: mean overlap = 3.80, precision = 0.0190, fold = 12.38, median p = 1.976e-03
- top200_hypergeom: mean overlap = 3.80, precision = 0.0190, fold = 12.38, median p = 1.976e-03
- top200_fold: mean overlap = 3.80, precision = 0.0190, fold = 12.38, median p = 1.976e-03

## Supervised Calibration

Best held-out Top200 model: logistic_l2, mean overlap = 3.50, interval = [2, 3.0, 7], mean fold = 11.40
- LightGBM: unavailable in current environment, not run.
- Logistic models do not exceed the held-out raw baseline Top200 mean (4.90).

## Two-branch Score

Best held-out Top200 variant: A_branch_score, mean overlap = 4.90, interval = [2, 5.0, 8], mean fold = 15.96
- Two-branch does not exceed the held-out raw baseline Top200 mean (4.90).

## Current Ceiling

- Held-out raw baseline Top200 stable interval: [2, 5.0, 8]
- Existing features currently stabilize around Top200 overlap 5.0 on held-out splits.
- Main conclusions should still be based on held-out validation, not oracle/full-truth diagnostics.
- If V4 still cannot materially push held-out Top200 above the raw baseline, region/expression modules are still justified.
