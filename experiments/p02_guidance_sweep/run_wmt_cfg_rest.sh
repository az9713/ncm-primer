#!/bin/bash
# Project 2a (rest): cfg 1.0 on all 3003 sources; cfg 0/1.5/2.0/3.0 on the first 1000 sources (batch 16).
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python; cd $E/p02_guidance_sweep
W="--config $N/config/eval/wmt14.yaml --checkpoint-dir $E/artifacts/wmt14 --data-dir $E/data/wmt14 --bs 16"
$E/common/run.sh run_wmt_cfg.log $PY $N/evaluate.py $W --cfg 1.0 --out $E/p02_guidance_sweep/wmt_bs16_cfg1.0
for c in 0 0.5 1.0 1.5 2.0 3.0; do
  $E/common/run.sh run_wmt_cfg1k.log $PY $N/evaluate.py $W --n 1000 --cfg $c --out $E/p02_guidance_sweep/wmt1k_bs16_cfg$c
done
