# Unified RBP Model v1: original-logic route

This is the route matching the original project goal.

It keeps both original model logics:

1. Motif branch: original V6.1 `ESMRNA_Predictor`
   - ESM2 protein encoder
   - phys_feats
   - kingdom labels
   - subtype / seed conditioned decoder
   - SVD reconstruction to 16,384 7-mer z-score motif profile

2. Binding branch: original MuSIC `RBPBindingCNN`
   - original RNA CNN
   - original protein embedding projector
   - original binding classifier logic
   - plus a trainable motif-aware bridge/classifier using V6.1 motif latent/subtype/seed features

Training data:

- RNAcompete motif profiles: 348 protein profiles
- MuSIC CLIP windows: experiment-holdout train/val/test cache
- External validation proteins are not main training data: AtGRP7, AtGRP8, OsDRB1

## Build multitask resources

```bash
cd /public/home/wz/workplace/cursor/modle/unified_rbp_model_v1
PY=/public/home/wz/.conda/envs/rbp_model/bin/python
$PY build_multitask_resources.py
```

## Cache motif features for MuSIC CLIP proteins

```bash
CUDA_VISIBLE_DEVICES=3 $PY cache_original_logic_motif_features.py \
  --device cuda \
  --batch-size 4
```

## Dry-run

```bash
CUDA_VISIBLE_DEVICES=3 $PY train_unified_original_logic.py \
  --dry-run-check \
  --device cuda \
  --batch-size-motif 2 \
  --batch-size-binding 32
```

## Train original-logic unified model

### Stage 1: motif

```bash
CUDA_VISIBLE_DEVICES=3 $PY train_unified_original_logic.py \
  --stage motif \
  --epochs 10 \
  --motif-steps-per-epoch 30 \
  --batch-size-motif 2 \
  --lr 5e-5 \
  --device cuda \
  2>&1 | tee results/original_logic_motif_train.log
```

Checkpoint:

```text
checkpoints/unified_original_logic_model_motif.pt
```

### Stage 2: binding

```bash
CUDA_VISIBLE_DEVICES=3 $PY train_unified_original_logic.py \
  --stage binding \
  --resume checkpoints/unified_original_logic_model_motif.pt \
  --epochs 10 \
  --binding-steps-per-epoch 500 \
  --batch-size-binding 512 \
  --lr 5e-5 \
  --device cuda \
  2>&1 | tee results/original_logic_binding_train.log
```

Checkpoint:

```text
checkpoints/unified_original_logic_model_binding.pt
```

### Stage 3: joint

```bash
CUDA_VISIBLE_DEVICES=3 $PY train_unified_original_logic.py \
  --stage joint \
  --resume checkpoints/unified_original_logic_model_binding.pt \
  --batch-size-motif 2 \
  --epochs 10 \
  --binding-steps-per-epoch 500 \
  --batch-size-binding 512 \
  --lr 1e-5 \
  --device cuda \
  2>&1 | tee results/original_logic_joint_train.log
```

Checkpoint:

```text
checkpoints/unified_original_logic_model_joint.pt
```

Do not use `train_unified_multitask.py` as the main model if the goal is preserving both original model logics. That script is only a simplified baseline.
