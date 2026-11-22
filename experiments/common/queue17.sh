#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; PY=$E/.venv/Scripts/python
until grep -q "QUEUE16-DONE" $E/common/queue16.out; do sleep 20; done
cd $E/p09_ar_baseline
P9_NSEQ="16:1024" P9_PEN_B=16 P9_OUT=results_b16_fresh.json $E/common/run.sh run4.log $PY ar_vs_ncm_small.py
cd $E/p05_tokenizer_recon && $E/common/run.sh wmt_fit_check.log $PY wmt_fit_check.py
cd $E/p03_level1_probe && NCM_MEM_FRACTION=0.92 $E/common/run.sh probe3.log $PY probe.py && $E/common/run.sh seed_effect.log $PY seed_effect.py
echo QUEUE17-DONE
