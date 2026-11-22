#!/bin/bash
# Project 2a: WMT14 guidance sweep, full 3003 sources, greedy config sampler (top_k=1). Deterministic.
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python; cd $E/p02_guidance_sweep
W="--config $N/config/eval/wmt14.yaml --checkpoint-dir $E/artifacts/wmt14 --data-dir $E/data/wmt14"
for c in ${CFGS:-0 0.5 1.0 1.5 2.0 3.0}; do
  $E/common/run.sh run_wmt_cfg.log $PY $N/evaluate.py $W --cfg $c --bs 16 --out $E/p02_guidance_sweep/wmt_bs16_cfg$c
done
