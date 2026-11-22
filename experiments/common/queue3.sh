#!/bin/bash
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python; SE=$E/common/shard_eval.sh
while tasklist //FI "PID eq 21716" | grep -q 21716; do sleep 10; done
echo "merge pid 21716 gone $(date -Iseconds)"
LMC=$N/config/eval/lm1b-small.yaml; LMK=$E/artifacts/lm1b/small; P1=$E/p01_penalties
# 1. fp16 GPU score of the default shard (pair with the CPU fp32 score for validation)
$E/common/run.sh $P1/score_validation.log $PY $E/common/score_genppl.py --config $LMC --glob "$P1/lm1b_default/shard_*.pkl" --dtype fp16
# 2. remaining LM1B arms
bash $SE $P1/run_arms.log $P1/lm1b_argmax $LMC $LMK fp16 --freq 0 --mu 0 --adj ""
bash $SE $P1/run_arms.log $P1/lm1b_adjonly $LMC $LMK fp16 --freq 0 --mu 0
bash $SE $P1/run_arms.log $P1/lm1b_freqonly $LMC $LMK fp16 --adj ""
echo P1-LM1B-DONE
# 3. WMT14 guidance sweep (BLEU merge loads no LM)
bash $E/p02_guidance_sweep/run_wmt_cfg.sh
echo P2-WMT-CFG-DONE
# 4. timing
(cd $E/p06_wallclock && $E/common/run.sh run.log $PY timing.py)
echo P6-DONE
# 5. TinyStories
TSC=$N/config/eval/tinystories.yaml; TSK=$E/artifacts/tinystories; P3=$E/p03_level1_probe
bash $SE $P3/run.log $P3/ts_default $TSC $TSK fp16
bash $SE $P3/run.log $P3/ts_uniform $TSC $TSK fp16 --l0 uniform
bash $SE $P3/run.log $P3/ts_smooth0.1 $TSC $TSK fp16 --l0 smooth:0.1
(cd $P3 && $E/common/run.sh probe.log $PY probe.py)
echo P3-DONE
# 6. AR baseline
(cd $E/p09_ar_baseline && $E/common/run.sh run.log $PY ar_vs_ncm.py)
echo P9-DONE
# 7. tokenizer training timing
(cd $E/p10_train_tokenizer && $E/common/run.sh timing100.log $PY train.py --steps 100 --bs 1024 --tag timing100 --eval_every 100)
echo P10-TIMING-DONE
# 8. XSum first 1000 (penalty arms + guidance)
bash $E/p01_penalties/run_xsum.sh
echo QUEUE3-DONE
