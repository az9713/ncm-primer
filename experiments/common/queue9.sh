#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE8-DONE" $E/common/queue8.out; do sleep 20; done
(cd $E/p09_ar_baseline && $E/common/run.sh run2.log $PY ar_vs_ncm.py); echo P9-RERUN-DONE
echo QUEUE9-DONE
