#!/bin/bash
# XSum, first 1000 test docs: penalty arms (project 1) and guidance w=0.5 (project 2, paper Fig.14 / Table 2 setting).
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python; cd $E/p01_penalties
X="--config $N/config/eval/xsum.yaml --checkpoint-dir $E/artifacts/xsum --data-dir $E/data/xsum --n 1000 --bs 16"
$E/common/run.sh run_xsum.log $PY $N/evaluate.py $X --out $E/p01_penalties/xsum1k_default
$E/common/run.sh run_xsum.log $PY $N/evaluate.py $X --freq 0 --mu 0 --adj "" --out $E/p01_penalties/xsum1k_argmax
$E/common/run.sh run_xsum.log $PY $N/evaluate.py $X --freq 0 --mu 0 --out $E/p01_penalties/xsum1k_adjonly
$E/common/run.sh run_xsum.log $PY $N/evaluate.py $X --adj "" --out $E/p01_penalties/xsum1k_freqonly
cd $E/p02_guidance_sweep
$E/common/run.sh run_xsum_cfg.log $PY $N/evaluate.py $X --cfg 0 --bs 8 --out $E/p02_guidance_sweep/xsum1k_cfg0
$E/common/run.sh run_xsum_cfg.log $PY $N/evaluate.py $X --cfg 0.5 --bs 8 --out $E/p02_guidance_sweep/xsum1k_cfg0.5
$E/common/run.sh run_xsum_cfg.log $PY $N/evaluate.py $X --cfg 1.0 --bs 8 --out $E/p02_guidance_sweep/xsum1k_cfg1.0
