#!/usr/bin/env bash
set -euo pipefail
cd /public/home/wz/workplace/cursor/modle/unified_rbp_model_v1
PY=/public/home/wz/.conda/envs/rbp_model/bin/python

# motif baseline: direct
CUDA_VISIBLE_DEVICES=3 $PY predict_unified_original_logic.py \
  --prediction-mode single \
  --checkpoint checkpoints/unified_original_logic_model_motif.pt \
  --motif-profile-mode direct \
  --query-rbp-ids AtGRP7,AtGRP8,LOC_Os05g24160.1 \
  --out-dir results/review_v2_motif_direct \
  --device cuda

# motif baseline: auto_teacher
CUDA_VISIBLE_DEVICES=3 $PY predict_unified_original_logic.py \
  --prediction-mode single \
  --checkpoint checkpoints/unified_original_logic_model_motif.pt \
  --motif-profile-mode auto_teacher \
  --query-rbp-ids AtGRP7,AtGRP8,LOC_Os05g24160.1 \
  --out-dir results/review_v2_motif_auto_teacher \
  --device cuda

$PY scripts/motif_known_rank.py \
  --motif-top-kmers results/review_v2_motif_direct/motif_top_kmers.tsv \
  --out results/review_v2_motif_direct/known_motif_rank.tsv
$PY scripts/motif_known_rank.py \
  --motif-top-kmers results/review_v2_motif_auto_teacher/motif_top_kmers.tsv \
  --out results/review_v2_motif_auto_teacher/known_motif_rank.tsv

# rice OsDRB1 transcriptome prediction
CUDA_VISIBLE_DEVICES=3 $PY predict_unified_original_logic.py \
  --prediction-mode posthoc \
  --motif-checkpoint checkpoints/unified_original_logic_model_motif.pt \
  --binding-checkpoint checkpoints/unified_original_logic_model_binding.pt \
  --window-tsv data/prediction_assets/rice_v7/input/rice_v7_windows_w200_s50.tsv.gz \
  --query-rbp-ids LOC_Os05g24160.1 \
  --motif-profile-mode auto_teacher \
  --structure-features-npy data/prediction_assets/rice_v7/structure_cache/transcriptome.paired_probability.npy \
  --structure-meta-json data/prediction_assets/rice_v7/structure_cache/transcriptome.structure_meta.json \
  --structure-alpha 0.5 \
  --structure-score-mode combined \
  --out-dir results/review_v2_rice_osdrb_posthoc_struct \
  --device cuda

# gene-level aggregation + TRIBE overlap
$PY scripts/gene_score_overlap_eval.py \
  --window-scores results/review_v2_rice_osdrb_posthoc_struct/window_scores.tsv.gz \
  --truth-gene-list /public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/11_rice_prediction/validation_tribe_osdrb1/results/tribe_msu_genes.tsv \
  --rbp-id LOC_Os05g24160.1 \
  --out-dir results/review_v2_rice_osdrb_posthoc_struct

# false-positive audit based on best overlap combo
$PY scripts/osdrb1_false_positive_audit.py \
  --gene-scores results/review_v2_rice_osdrb_posthoc_struct/gene_scores_by_inverted_base_motif_structure_score.with_rbp.tsv \
  --truth-gene-list /public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/11_rice_prediction/validation_tribe_osdrb1/results/tribe_msu_genes.tsv \
  --rbp-id LOC_Os05g24160.1 \
  --ranking-score-col top3_mean_score \
  --top-k 1000 \
  --out-dir results/review_v2_rice_osdrb_posthoc_struct/false_positive_audit_best
