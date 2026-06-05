# Review Bundle V2

1. Project path
/public/home/wz/workplace/cursor/modle/unified_rbp_model_v1

2. Bundle generation time
2026-06-05 15:26:31 +0800

3. What was rerun
- Motif baseline with repaired predictor:
  - results/review_v2_motif_direct
  - results/review_v2_motif_auto_teacher
- Rice OsDRB1 transcriptome prediction with repaired predictor:
  - results/review_v2_rice_osdrb_posthoc_struct

4. Fixed logic exercised in this rerun
- auto_teacher kingdom label default plant=1
- binding/late_fusion/joint default head-only trainability
- gene ranking aggregation outputs
- score ablation window-level outputs
- coordinate audit available in reports

5. Key observations
- AtGRP7 direct and auto_teacher are still G-rich in this minimal rerun.
- AtGRP8 direct and auto_teacher are still G-rich in this minimal rerun.
- OsDRB1 direct is U-rich-like; auto_teacher is UC-rich.
- Best TRIBE overlap in this rerun came from:
  - score_name = inverted_base_motif_structure_score
  - aggregation = top3_mean_score
  - Top1000 overlap = 48 / 283 comparable truth genes

6. Included files
- repaired core code
- motif baseline summaries and known_motif_rank.tsv
- rice prediction_report.json / motif_summary / motif_top_kmers
- score_ablation_summary.tsv
- overlap_evaluation_summary.tsv / .json
- false_positive_audit.tsv / .json
- preview_gene_scores*.tsv + wc summaries
- run_commands_used.sh

7. Excluded large files
- checkpoints
- *.npy / *.npz
- full window_scores.tsv.gz
- full structure cache
- full transcriptome window table
- full gene_scores_by_*.tsv (preview only)
