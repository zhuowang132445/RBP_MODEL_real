# Examples

Baseline:

```bash
python scripts/run_rbp_trace_baseline.py --config configs/rbp_trace_osdrb1_v2.yaml
```

ContextV2:

```bash
python scripts/run_rbp_trace_context_v2.py \
  --config configs/rbp_trace_osdrb1_v2.yaml \
  --enable-region \
  --enable-repeat \
  --enable-hairpin
```
