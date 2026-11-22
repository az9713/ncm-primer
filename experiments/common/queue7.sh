#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE6-DONE" $E/common/queue6.out; do sleep 20; done
(cd $E/p06_wallclock && $E/common/run.sh run2.log $PY timing.py); echo P6-RERUN-DONE
S=$(cat $E/p10_train_tokenizer/STEPS.txt 2>/dev/null || echo 10000)
(cd $E/p10_train_tokenizer && $E/common/run.sh train_real.log $PY train.py --steps $S --bs 1024 --tag real --eval_every 1000); echo P10-REAL-DONE
(cd $E/p10_train_tokenizer && $E/common/run.sh train_aug.log $PY train.py --steps $S --bs 1024 --tag aug --aug --eval_every 1000); echo P10-AUG-DONE
echo QUEUE7-DONE
