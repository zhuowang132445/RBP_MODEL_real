# unified_rbp_model_v1

This folder contains a first unified RBP model that combines:

- Motif backbone: `rbp_motif_model_v6_no_test_time_prior_prototype_20260527`
- Binding backbone: MuSIC-style `RBPBindingCNN`
- Trainable bridge: `UnifiedFusionHead`

The two old backbones are frozen in stage 1. Stage 1 trains only the fusion head from cached features.

## What This Model Does

Inputs:

- RBP protein sequence / cached protein features
- 200 nt RNA window
- cached binding ESM embedding for the RBP when using the binding branch
- window-level features such as motif match, RNA structure, composition, and base binding score

Outputs:

- motif prediction outputs through the motif backbone
- base binding score through the binding backbone
- unified motif-aware binding score through the fusion head
- gene-level ranking from window-level scores

## Important Limitation

The original binding CNN expects a 1280-d ESM-2 t33 protein embedding. Raw protein sequence alone is not enough for that branch unless the embedding has already been computed. This project does not rerun large-scale ESM extraction.

## Data Policy

Large original and intermediate datasets are included as symlinks under:

- `data_sources/`
- `data/`

See:

- `DATA_MANIFEST.json`
- `DATA_MANIFEST.tsv`

## Setup

```bash
cd /public/home/wz/workplace/cursor/modle/unified_rbp_model_v1
PY=/public/home/wz/.conda/envs/rbp_model/bin/python

$PY prepare_project_data.py
$PY audit_unified_project.py
```

## Stage 1 Training

Dry run:

```bash
$PY train_unified_stage1.py --dry-run-check --device cpu --batch-size 32
```

Train fusion head only:

```bash
CUDA_VISIBLE_DEVICES=3 $PY train_unified_stage1.py \
  --epochs 50 \
  --batch-size 512 \
  --lr 1e-3 \
  --device cuda \
  2>&1 | tee results/stage1_train.log
```

Outputs:

- `checkpoints/unified_stage1_fusion_head.pt`
- `results/stage1_training_history.tsv`
- `results/stage1_test_metrics.tsv`
- `results/stage1_train_report.json`

## Prediction From Precomputed Window Features

```bash
CUDA_VISIBLE_DEVICES=3 $PY predict_motif_and_binding.py score-table \
  --checkpoint checkpoints/unified_stage1_fusion_head.pt \
  --window-features data/stage1_validation/fusion_v2_window_features.tsv.gz \
  --out-dir results/stage1_score_table \
  --device cuda
```

Outputs:

- `unified_window_scores.tsv.gz`
- `unified_gene_scores.tsv`
- `unified_prediction_report.json`

## Current Scope

This is not yet full multi-task end-to-end training on RNAcompete + POSTAR3. It is a first unified system that makes the two existing models work inside one model class and trains a motif-aware binding fusion head.

The next stage would train on both:

- motif task: RNAcompete z-score reconstruction
- binding task: POSTAR3/CLIP window BCE

with a shared protein representation.
