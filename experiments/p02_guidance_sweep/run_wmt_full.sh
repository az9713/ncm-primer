#!/bin/bash
# Project 2 (full test set): cfg 1.5/2.0/3.0 on all 3003 sources, then the sampler sweep on all 3003 sources
# (cfg 1.0, greedy + T 0.6/0.8/1.0 with top_p 0.9, 3 repeats each). Batch 16.
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python; cd $E/p02_guidance_sweep
W="--config $N/config/eval/wmt14.yaml --checkpoint-dir $E/artifacts/wmt14 --data-dir $E/data/wmt14 --bs 16"
for c in 1.5 2.0 3.0; do
  $E/common/run.sh run_wmt_full.log $PY $N/evaluate.py $W --cfg $c --out $E/p02_guidance_sweep/wmt_bs16_cfg$c
done
$E/common/run.sh run_wmt_full.log $PY $N/evaluate.py $W --cfg 1.0 --out $E/p02_guidance_sweep/wmtfull_cfg1.0_greedy
for t in 0.6 0.8 1.0; do for r in 1 2 3; do
  $E/common/run.sh run_wmt_full.log $PY $N/evaluate.py $W --cfg 1.0 --temp $t --top_k 0 --top_p 0.9 --out $E/p02_guidance_sweep/wmtfull_cfg1.0_t${t}_p0.9_r$r
done; done
