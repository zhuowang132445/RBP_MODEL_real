# RBP_TRACE_V2_1 CONTEXT DIAGNOSIS REPORT

## Baseline

- baseline Top200: 16
- baseline Top500: 28
- baseline Top1000: 48

## Current V2 Held-out

- current V2 selected all_context held-out Top200 mean overlap: 6.40

## Oracle Upper Bound

- best region Top200: 18 via `rbp_trace_v2_score_region__explicit__k_10__beta_0.25`
- best self-complementarity Top200: 16 via `rbp_trace_v2_score_repeat__self_complementarity__k_3__beta_0.0`
- best paired-architecture Top200: 27 via `rbp_trace_v2_score_paired_architecture__k_10__beta_0.25`
- best all_context Top200: 31 via `rbp_trace_v2_score_all_context__k_20__beta_region_0.5__beta_repeat_0.1__beta_hairpin_0.25__repeat_self_complementarity_minus_simple_repeat_penalty`
- best overall Top200: 31 via `rbp_trace_v2_score_all_context__k_20__beta_region_0.5__beta_repeat_0.1__beta_hairpin_0.25__repeat_self_complementarity_minus_simple_repeat_penalty`

## Region Diagnostic

- original region annotation is effectively missing UTR signal; inferred UTR from CDS boundaries partially restores it.

## Module Contribution

- self-complementarity module oracle Top200 upper bound: 16
- paired-architecture module oracle Top200 upper bound: 27
- paired-architecture remains the dominant single context module.

## Top-window-k Diagnostic

- all_context: best top-window-k = 20, Top200 = 31
- paired_architecture: best top-window-k = 10, Top200 = 27
- region: best top-window-k = 10, Top200 = 18
- self_complementarity: best top-window-k = 3, Top200 = 16

## Reality Check

- Full-truth oracle results are upper-bound diagnostics, not generalization estimates.
- Realistic current ceiling should be interpreted between the current held-out all_context result and the oracle best candidate.
- Current three-module ceiling estimate for OsDRB1 is roughly held-out Top200 6.4 versus oracle Top200 31.
