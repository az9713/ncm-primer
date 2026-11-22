#!/bin/bash
E=$HOME/Downloads/stanford_ncm_turing/experiments; N=$HOME/Downloads/stanford_ncm_turing/ncm-main
PY=$E/.venv/Scripts/python; cd $E/p03_level1_probe
T="--config $N/config/eval/tinystories.yaml --checkpoint-dir $E/artifacts/tinystories"
$E/common/run.sh run.log $PY $N/evaluate.py $T --out $E/p03_level1_probe/ts_default
$E/common/run.sh run.log $PY $N/evaluate.py $T --l0 uniform --out $E/p03_level1_probe/ts_uniform
$E/common/run.sh run.log $PY $N/evaluate.py $T --l0 smooth:0.1 --out $E/p03_level1_probe/ts_smooth0.1
$E/common/run.sh probe.log $PY probe.py
