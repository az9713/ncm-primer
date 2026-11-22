#!/bin/bash
export NCM_GEN=$(cat $HOME/Downloads/stanford_ncm_turing/experiments/common/GEN)
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python; SE=$E/common/shard_eval.sh
until grep -q "### END" $E/p02_guidance_sweep/run_wmt_cfg.log; do sleep 15; done
echo "wmt cfg0 bs64 finished $(date -Iseconds)"
bash $E/p02_guidance_sweep/run_wmt_cfg.sh; echo P2-WMT-CFG-DONE
(cd $E/p06_wallclock && $E/common/run.sh run.log $PY timing.py); echo P6-DONE
(cd $E/p05_tokenizer_recon && $E/common/run.sh pad_check.log $PY pad_check.py); echo PADCHECK-DONE
TSC=$N/config/eval/tinystories.yaml; TSK=$E/artifacts/tinystories; P3=$E/p03_level1_probe
bash $SE $P3/run.log $P3/ts_default $TSC $TSK fp16
bash $SE $P3/run.log $P3/ts_uniform $TSC $TSK fp16 --l0 uniform
bash $SE $P3/run.log $P3/ts_smooth0.1 $TSC $TSK fp16 --l0 smooth:0.1
(cd $P3 && $E/common/run.sh probe.log $PY probe.py); echo P3-DONE
(cd $E/p09_ar_baseline && $E/common/run.sh run.log $PY ar_vs_ncm.py); echo P9-DONE
(cd $E/p10_train_tokenizer && $E/common/run.sh timing100.log $PY train.py --steps 100 --bs 1024 --tag timing100 --eval_every 100); echo P10-TIMING-DONE
(cd $E/p04_coarse_to_fine && $E/common/run.sh bf16_hard_full.log $PY bf16_hard_full.py); echo BF16-DONE
bash $E/p01_penalties/run_xsum.sh; echo XSUM-DONE
(cd $E/p08_mbr && $E/common/run.sh run.log $PY mbr.py --n $(cat $E/p08_mbr/N.txt 2>/dev/null || echo 3003) --bs 16 --temp 0.8 --top_p 0.9); echo P8-DONE
bash $E/p02_guidance_sweep/run_wmt_sampler.sh; echo P2-SAMPLER-DONE
$E/common/run.sh $E/p03_level1_probe/score_validation.log $PY $E/common/score_genppl.py --config $TSC --glob "$P3/ts_default/shard_*.pkl" --dtype fp16 --limit 64
$E/common/run.sh $E/p03_level1_probe/score_validation.log $PY $E/common/score_genppl.py --config $TSC --glob "$P3/ts_default/shard_*.pkl" --dtype fp32 --device cpu --limit 64
echo QUEUE5-DONE
