#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE14-DONE" $E/common/queue14.out; do sleep 20; done
cd $E/p09_ar_baseline
P9_NSEQ="4:256" P9_PEN_B=4 P9_OUT=results_b4_fresh.json $E/common/run.sh run4.log $PY ar_vs_ncm_small.py
P9_NSEQ="8:512" P9_PEN_B=8 P9_OUT=results_b8_fresh.json $E/common/run.sh run4.log $PY ar_vs_ncm_small.py
echo QUEUE15-DONE
