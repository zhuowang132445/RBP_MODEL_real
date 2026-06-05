# Review Manifest

1. Current project path
/public/home/wz/workplace/cursor/modle/unified_rbp_model_v1

2. Bundle generation time
2026-06-05 14:46:35 +0800

3. Modification summary
- Synced repaired predictor/trainer/auto-teacher code into the project before bundling.
- Added lightweight diagnostic scripts for coordinate offset audit and OsDRB1 false-positive audit.
- Generated trainable parameter, coordinate alignment, false-positive, and gene-ranking-bias reports.

4. Fixed issues
- auto_teacher kingdom label: default corrected to plant=1 and per-query kingdom resolution added.
- trainability / freeze binding backbone: default late_fusion/binding/joint changed to head-only unless explicit flags are set.
- coordinate offset: added key-based +1 inference audit rather than row-order merge.
- gene ranking aggregation: repaired predictor now supports multiple aggregations and correlation reporting.
- score ablation: repaired predictor now emits multiple ablation score columns and summary metrics when labels exist.

5. Still unresolved
- Existing transcriptome result directories in this bundle were produced before a full rerun with the repaired predictor, so they do not yet contain refreshed gene_scores_by_* outputs.
- score_ablation_summary for transcriptome-wide runs is absent because those runs do not carry labels.
- RBP specificity / target de-duplication issues remain a modeling problem and are not solved by this bundling step.

6. Report file purposes
- alignment_audit.tsv / alignment_audit_report.json: motif cache, embedding alignment, structure cache, k-mer normalization, trainability, and prior audit summary.
- trainable_parameter_report.tsv / .json: confirms default head-only trainability for late_fusion after the fix.
- coordinate_alignment_report.json / .tsv: direct vs +1 validation-window coordinate matching against transcriptome windows.
- gene_ranking_bias_report.tsv / .json: correlation between gene score and number of windows.
- false_positive_audit.tsv / .json: Top1000 OsDRB1 TP/FP/outside-top1000 gene-level audit.
- unified_project_audit.json: broader project audit snapshot from the in-project audit script.

7. Prediction result directories and commands
- rice_transcriptome_osdrb_posthoc_struct: rice transcriptome OsDRB1 posthoc motif+structure prediction. See run_commands_used.sh.
- arabidopsis_transcriptome_atgrp7_atgrp8_osdrb_posthoc_struct: Arabidopsis transcriptome AtGRP7/AtGRP8/OsDRB1 posthoc motif+structure prediction. See run_commands_used.sh.

8. Large files intentionally omitted
- *.pt / *.pth checkpoints
- *.npy / large *.npz
- full structure caches
- full transcriptome window tables
- full window_scores.tsv.gz
- any large gene_scores tables are represented by preview_*.tsv plus wc summaries
