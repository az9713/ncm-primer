#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE18-DONE" $E/common/queue18.out; do sleep 15; done
cd $E/p09_ar_baseline && $E/common/run.sh leak_test.log $PY leak_test.py
echo QUEUE19-DONE
