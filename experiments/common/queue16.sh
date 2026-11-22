#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE15-DONE" $E/common/queue15.out; do sleep 10; done
echo "old queues drained $(date -Iseconds)"
cd $E/p10_train_tokenizer
$E/common/run.sh train_real_2500.log $PY train.py --steps 2500 --bs 1024 --accum 4 --tag real --eval_every 500
$E/common/run.sh train_aug_2500.log $PY train.py --steps 2500 --bs 1024 --accum 4 --tag aug --aug --eval_every 500
cd $E/p09_ar_baseline
P9_NSEQ="4:256" P9_PEN_B=4 P9_OUT=results_b4_fresh.json $E/common/run.sh run4.log $PY ar_vs_ncm_small.py
P9_NSEQ="8:512" P9_PEN_B=8 P9_OUT=results_b8_fresh.json $E/common/run.sh run4.log $PY ar_vs_ncm_small.py
echo QUEUE16-DONE
