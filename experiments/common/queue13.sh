#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE12-DONE" $E/common/queue12.out; do sleep 20; done
cd $E/p09_ar_baseline
P9_NSEQ="1:64,4:256,8:512" P9_PEN_B=8 P9_OUT=results_small_batches.json $E/common/run.sh run3.log $PY ar_vs_ncm_small.py
echo QUEUE13-DONE
