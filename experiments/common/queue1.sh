#!/bin/bash
E=$HOME/Downloads/stanford_ncm_turing/experiments
until grep -q "### END" $E/p05_tokenizer_recon/run.log; do sleep 10; done
bash $E/p01_penalties/run_all.sh
bash $E/p02_guidance_sweep/run_wmt_cfg.sh
echo QUEUE1-DONE
