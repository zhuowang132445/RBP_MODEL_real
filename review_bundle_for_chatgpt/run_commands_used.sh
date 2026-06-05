#!/usr/bin/env bash
set -euo pipefail
cd /public/home/wz/workplace/cursor/modle/unified_rbp_model_v1
PY=/public/home/wz/.conda/envs/rbp_model/bin/python

# alignment audit
$PY audit_unified_project.py
$PY scripts/validation_coordinate_alignment_audit.py \
  --window-tsv data/prediction_assets/rice_v7/input/rice_v7_windows_w200_s50.tsv.gz \
  --observed-tsv /public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/11_rice_prediction/validation_hice_window_osdrb1_matched/hice_centered_observed_scores.tsv \
  --background-tsv /public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/11_rice_prediction/validation_hice_window_osdrb1_matched/hice_same_transcript_background_scores.tsv.gz \
  --out-dir results/coordinate_alignment_tmp

# kingdom-label fixed motif baseline
CUDA_VISIBLE_DEVICES=3 $PY predict_unified_original_logic.py \
  --prediction-mode single \
  --checkpoint checkpoints/unified_original_logic_model_motif.pt \
  --motif-profile-mode direct \
  --query-rbp-ids AtGRP7,AtGRP8,LOC_Os05g24160.1 \
  --out-dir results/atgrp7_atgrp8_osdrb_direct_motif_ckpt

# OsDRB1 transcriptome prediction
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
  --out-dir results/rice_transcriptome_osdrb_posthoc_struct \
  --device cuda

# Arabidopsis transcriptome prediction
CUDA_VISIBLE_DEVICES=3 $PY predict_unified_original_logic.py \
  --prediction-mode posthoc \
  --motif-checkpoint checkpoints/unified_original_logic_model_motif.pt \
  --binding-checkpoint checkpoints/unified_original_logic_model_binding.pt \
  --window-tsv data/prediction_assets/arabidopsis_tair10/input/tair10_windows_w200_s50.tsv.gz \
  --query-rbp-ids AtGRP7,AtGRP8,LOC_Os05g24160.1 \
  --motif-profile-mode auto_teacher \
  --structure-features-npy data/prediction_assets/arabidopsis_tair10/structure_cache/transcriptome.paired_probability.npy \
  --structure-meta-json data/prediction_assets/arabidopsis_tair10/structure_cache/transcriptome.structure_meta.json \
  --structure-alpha 0.5 \
  --structure-score-mode combined \
  --out-dir results/arabidopsis_transcriptome_atgrp7_atgrp8_osdrb_posthoc_struct \
  --device cuda

# score ablation (written automatically when labels exist)
CUDA_VISIBLE_DEVICES=3 $PY predict_unified_original_logic.py \
  --prediction-mode posthoc \
  --motif-checkpoint checkpoints/unified_original_logic_model_motif.pt \
  --binding-checkpoint checkpoints/unified_original_logic_model_binding.pt \
  --window-tsv /public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/results/plant_rbp_validation/atgrp7_atgrp8_targeted_windows.tsv.gz \
  --query-rbp-ids AtGRP7,AtGRP8 \
  --motif-profile-mode auto_teacher \
  --out-dir results/predict_atgrp7_atgrp8_posthoc_motif_binding \
  --device cuda

# false-positive audit
$PY scripts/osdrb1_false_positive_audit.py \
  --gene-scores results/rice_transcriptome_osdrb_posthoc_struct/gene_scores.tsv \
  --truth-gene-list /public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/11_rice_prediction/validation_tribe_osdrb1/results/tribe_msu_genes.tsv \
  --rbp-id LOC_Os05g24160.1 \
  --top-k 1000 \
  --out-dir results/false_positive_audit_tmp
