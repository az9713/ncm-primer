#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE9-DONE" $E/common/queue9.out; do sleep 20; done
(cd $E/p04_coarse_to_fine && $E/common/run.sh bf16_hard_full2.log $PY bf16_hard_full.py); echo BF16-DONE
echo QUEUE10-DONE
