#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
cd $E/p09_ar_baseline
P9_NSEQ="16:1024,64:1024" P9_PEN_B=16 P9_OUT=results_gcfix.json $E/common/run.sh run6.log $PY ar_vs_ncm_small.py
cd $E/p06_wallclock
for t in wmt14 tinystories lm1b-small xsum; do $E/common/run.sh run4.log $PY timing.py $t; done
echo QUEUE20-DONE
