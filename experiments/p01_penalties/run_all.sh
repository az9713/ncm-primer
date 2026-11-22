#!/bin/bash
# Project 1: LM1B small, 4 penalty arms (paired: same seeds, pick_tokens draws no random numbers); XSum on first 1000 docs.
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python; cd $E/p01_penalties
LM="--config $N/config/eval/lm1b-small.yaml --checkpoint-dir $E/artifacts/lm1b/small"
$E/common/run.sh run.log $PY $N/evaluate.py $LM --out $E/p01_penalties/lm1b_default
$E/common/run.sh run.log $PY $N/evaluate.py $LM --freq 0 --mu 0 --adj "" --out $E/p01_penalties/lm1b_argmax
$E/common/run.sh run.log $PY $N/evaluate.py $LM --freq 0 --mu 0 --out $E/p01_penalties/lm1b_adjonly
$E/common/run.sh run.log $PY $N/evaluate.py $LM --adj "" --out $E/p01_penalties/lm1b_freqonly
