#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python
until grep -q "QUEUE10-DONE" $E/common/queue10.out; do sleep 15; done
echo "old queues drained $(date -Iseconds)"
(cd $E/p06_wallclock && $E/common/run.sh run2.log $PY timing.py); echo P6-RERUN-DONE
(cd $E/p09_ar_baseline && $E/common/run.sh run2.log $PY ar_vs_ncm.py); echo P9-RERUN-DONE
(cd $E/p03_level1_probe && NCM_MEM_FRACTION=0.92 $E/common/run.sh probe2.log $PY probe.py); echo P3-PROBE-DONE
(cd $E/p04_coarse_to_fine && $E/common/run.sh bf16_hard_full2.log $PY bf16_hard_full.py); echo BF16-DONE
S=$(cat $E/p10_train_tokenizer/STEPS.txt 2>/dev/null || echo 10000)
(cd $E/p10_train_tokenizer && $E/common/run.sh train_real.log $PY train.py --steps $S --bs 1024 --tag real --eval_every 1000); echo P10-REAL-DONE
bash $E/p01_penalties/run_xsum.sh; echo XSUM-DONE
(cd $E/p08_mbr && $E/common/run.sh run.log $PY mbr.py --n $(cat $E/p08_mbr/N.txt 2>/dev/null || echo 3003) --bs 16 --temp 0.8 --top_p 0.9); echo P8-DONE
bash $E/p02_guidance_sweep/run_wmt_sampler.sh; echo P2-SAMPLER-DONE
(cd $E/p10_train_tokenizer && $E/common/run.sh train_aug.log $PY train.py --steps $S --bs 1024 --tag aug --aug --eval_every 1000); echo P10-AUG-DONE
TSC=$N/config/eval/tinystories.yaml; P3=$E/p03_level1_probe
$E/common/run.sh $P3/score_validation.log $PY $E/common/score_genppl.py --config $TSC --glob "$P3/ts_default/shard_*.pkl" --dtype fp16 --limit 64
$E/common/run.sh $P3/score_validation.log $PY $E/common/score_genppl.py --config $TSC --glob "$P3/ts_default/shard_*.pkl" --dtype fp32 --device cpu --limit 64
echo QUEUE11-DONE
