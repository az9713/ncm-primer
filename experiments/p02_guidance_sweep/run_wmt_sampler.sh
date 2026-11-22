#!/bin/bash
# Project 2b: WMT14 sampler sweep at cfg 1.0, first 500 sources, batch 16, 3 repeats each (conditional path sets no seed).
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python; cd $E/p02_guidance_sweep
W="--config $N/config/eval/wmt14.yaml --checkpoint-dir $E/artifacts/wmt14 --data-dir $E/data/wmt14 --n 500 --bs 16"
$E/common/run.sh run_wmt_sampler.log $PY $N/evaluate.py $W --cfg 1.0 --out $E/p02_guidance_sweep/wmt500_cfg1.0_greedy
for t in 0.6 0.8 1.0; do for r in 1 2 3; do
  $E/common/run.sh run_wmt_sampler.log $PY $N/evaluate.py $W --cfg 1.0 --temp $t --top_k 0 --top_p 0.9 --out $E/p02_guidance_sweep/wmt500_cfg1.0_t${t}_p0.9_r$r
done; done
