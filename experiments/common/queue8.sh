#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE7-DONE" $E/common/queue7.out; do sleep 20; done
(cd $E/p03_level1_probe && NCM_MEM_FRACTION=0.92 $E/common/run.sh probe2.log $PY probe.py); echo P3-PROBE-DONE
echo QUEUE8-DONE
