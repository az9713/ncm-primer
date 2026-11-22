#!/bin/bash
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE1-DONE" $E/common/queue1.out; do sleep 10; done
(cd $E/p06_wallclock && $E/common/run.sh run.log $PY timing.py)
bash $E/p03_level1_probe/run_all.sh
(cd $E/p09_ar_baseline && $E/common/run.sh run.log $PY ar_vs_ncm.py)
(cd $E/p10_train_tokenizer && $E/common/run.sh timing100.log $PY train.py --steps 100 --bs 1024 --tag timing100 --eval_every 100)
echo QUEUE2-DONE
