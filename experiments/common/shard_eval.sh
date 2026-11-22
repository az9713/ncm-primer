#!/bin/bash
# usage: shard_eval.sh LOG OUTDIR CONFIG CKPT DTYPE [eval_shard args...]
# Runs scripts/eval_shard.py exactly as evaluate.py:29-35 builds it (1 shard, cwd = ncm-main), then scores the shard
# with common/score_genppl.py (copy of eval_merge.py genppl/genppl_flm) in DTYPE (fp16 or fp32) on the GPU.
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python
LOG=$1; OUT=$2; CFG=$3; CK=$4; DT=$5; shift 5
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
cd $N && $E/common/run.sh $LOG $PY $N/scripts/eval_shard.py --config $CFG --checkpoint-dir $CK --nshards 1 "$@" --shard 0 --out $OUT/shard_0.pkl
cd $E && $E/common/run.sh $LOG $PY $E/common/score_genppl.py --config $CFG --glob "$OUT/shard_*.pkl" --dtype $DT
