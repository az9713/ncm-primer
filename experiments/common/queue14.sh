#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE13-DONE" $E/common/queue13.out; do sleep 20; done
cd $E/p10_train_tokenizer
$E/common/run.sh timing100.log $PY train.py --steps 100 --bs 1024 --accum 4 --tag timing100 --eval_every 100
S=$(cat $E/p10_train_tokenizer/STEPS.txt 2>/dev/null || echo 10000)
$E/common/run.sh train_real.log $PY train.py --steps $S --bs 1024 --accum 4 --tag real --eval_every 1000
$E/common/run.sh train_aug.log $PY train.py --steps $S --bs 1024 --accum 4 --tag aug --aug --eval_every 1000
echo QUEUE14-DONE
