#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE11-DONE" $E/common/queue11.out; do sleep 20; done
cd $E/p06_wallclock
for t in wmt14 tinystories lm1b-small xsum; do $E/common/run.sh run3.log $PY timing.py $t; done
echo QUEUE12-DONE
