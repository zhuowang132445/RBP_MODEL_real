# RBP-TRACE

RBP-conditioned transcriptome-wide RNA binding potential prediction and candidate target ranking framework.

## Included

- `RBP-TRACE-Core`: baseline ranking using existing `inverted_base/motif/structure` scores
- `RBP-TRACE-ContextV2`: optional region, repeat/inverted-repeat, hairpin/stem-loop context modules

## Not Included

- expression module
- ADAR/TRIBE editability module
- deep model retraining

## Baseline vs ContextV2

- `baseline` uses only the existing `inverted_base_motif_structure_score` gene ranking logic
- `context_v2` adds optional transcript context features that do not depend on RNA-seq

## Run

Baseline:

```bash
python scripts/run_rbp_trace_baseline.py \
  --config configs/rbp_trace_osdrb1_v2.yaml
```

ContextV2:

```bash
python scripts/run_rbp_trace_context_v2.py \
  --config configs/rbp_trace_osdrb1_v2.yaml \
  --enable-region \
  --enable-repeat \
  --enable-hairpin
```

Evaluation helper:

```bash
python scripts/evaluate_rbp_trace.py \
  --config configs/rbp_trace_osdrb1_v2.yaml
```

## Output Interpretation

- `binding_potential_score`: baseline transcriptome ranking score from the existing model output
- `context_enhanced_score`: baseline score plus optional context bonuses selected on calibration truth only
- `candidate target ranking`: prioritization output for follow-up analysis

These outputs are ranking scores for candidate targets. They are not validated target probabilities.
