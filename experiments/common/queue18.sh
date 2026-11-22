#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE17-DONE" $E/common/queue17.out; do sleep 15; done
cd $E/p09_ar_baseline
P9_NSEQ="4:64,8:64,16:64" P9_PEN_B=16 P9_OUT=results_short_runs.json $E/common/run.sh run5.log $PY ar_vs_ncm_small.py
echo QUEUE18-DONE
