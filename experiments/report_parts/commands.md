## 5. Appendix: exact command lines

Every job was started by `experiments/common/run.sh`, which wrote these lines into the logs. Paths are as logged. Jobs that a stale queue skipped (`common/skipped.log`) are not listed. rc = exit code.

### p01_penalties

- `run.log` 2026-11-21T18:30:39-08:00 rc=0 5279 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/lm1b-small.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/lm1b/small --out ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/lm1b_default`
- `score_validation.log` 2026-11-21T19:11:44-08:00 rc=? ? s:
  `.venv/Scripts/python common/score_genppl.py --config ../ncm-main/config/eval/lm1b-small.yaml --glob p01_penalties/lm1b_default/shard_*.pkl --dtype fp32 --device cpu`
- `score_validation.log` 2026-11-21T20:00:08-08:00 rc=0 872 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/experiments/common/score_genppl.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/lm1b-small.yaml --glob ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/lm1b_default/shard_*.pkl --dtype fp16`
- `run_arms.log` 2026-11-21T20:15:33-08:00 rc=0 445 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/scripts/eval_shard.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/lm1b-small.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/lm1b/small --nshards 1 --freq 0 --mu 0 --adj  --shard 0 --out ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/lm1b_argmax/shard_0.pkl`
- `run_arms.log` 2026-11-21T20:23:04-08:00 rc=0 331 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/experiments/common/score_genppl.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/lm1b-small.yaml --glob ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/lm1b_argmax/shard_*.pkl --dtype fp16`
- `run_arms.log` 2026-11-21T20:29:13-08:00 rc=0 545 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/scripts/eval_shard.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/lm1b-small.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/lm1b/small --nshards 1 --freq 0 --mu 0 --shard 0 --out ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/lm1b_adjonly/shard_0.pkl`
- `run_arms.log` 2026-11-21T20:38:22-08:00 rc=0 176 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/experiments/common/score_genppl.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/lm1b-small.yaml --glob ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/lm1b_adjonly/shard_*.pkl --dtype fp16`
- `run_arms.log` 2026-11-21T20:41:42-08:00 rc=0 372 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/scripts/eval_shard.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/lm1b-small.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/lm1b/small --nshards 1 --adj  --shard 0 --out ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/lm1b_freqonly/shard_0.pkl`
- `run_arms.log` 2026-11-21T20:47:56-08:00 rc=0 166 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/experiments/common/score_genppl.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/lm1b-small.yaml --glob ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/lm1b_freqonly/shard_*.pkl --dtype fp16`
- `run_xsum.log` 2026-11-22T01:00:45-08:00 rc=1 519 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/xsum.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/xsum --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/xsum --n 1000 --out ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/xsum1k_default`
- `run_xsum.log` 2026-11-22T01:42:08-08:00 rc=0 456 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/xsum.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/xsum --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/xsum --n 1000 --bs 16 --out ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/xsum1k_default`
- `run_xsum.log` 2026-11-22T01:49:48-08:00 rc=0 463 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/xsum.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/xsum --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/xsum --n 1000 --bs 16 --freq 0 --mu 0 --adj  --out ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/xsum1k_argmax`
- `run_xsum.log` 2026-11-22T01:57:32-08:00 rc=0 456 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/xsum.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/xsum --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/xsum --n 1000 --bs 16 --freq 0 --mu 0 --out ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/xsum1k_adjonly`
- `run_xsum.log` 2026-11-22T02:05:09-08:00 rc=0 455 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/xsum.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/xsum --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/xsum --n 1000 --bs 16 --adj  --out ~/Downloads/stanford_ncm_turing/experiments/p01_penalties/xsum1k_freqonly`

### p02_guidance_sweep

- `run_wmt_cfg.log` 2026-11-21T20:51:37-08:00 rc=0 1978 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --cfg 0 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt_cfg0`
- `run_wmt_cfg.log` 2026-11-21T21:25:12-08:00 rc=0 475 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --cfg 0 --bs 16 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt_bs16_cfg0`
- `run_wmt_cfg.log` 2026-11-21T21:33:16-08:00 rc=0 1597 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --cfg 0.5 --bs 16 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt_bs16_cfg0.5`
- `run_wmt_cfg.log` 2026-11-22T00:28:39-08:00 rc=0 638 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --bs 16 --cfg 1.0 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt_bs16_cfg1.0`
- `run_wmt_cfg1k.log` 2026-11-22T00:39:18-08:00 rc=0 132 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --bs 16 --n 1000 --cfg 0 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt1k_bs16_cfg0`
- `run_wmt_cfg1k.log` 2026-11-22T00:41:31-08:00 rc=0 229 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --bs 16 --n 1000 --cfg 0.5 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt1k_bs16_cfg0.5`
- `run_wmt_cfg1k.log` 2026-11-22T00:45:21-08:00 rc=0 229 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --bs 16 --n 1000 --cfg 1.0 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt1k_bs16_cfg1.0`
- `run_wmt_cfg1k.log` 2026-11-22T00:49:12-08:00 rc=0 230 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --bs 16 --n 1000 --cfg 1.5 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt1k_bs16_cfg1.5`
- `run_wmt_cfg1k.log` 2026-11-22T00:53:03-08:00 rc=0 230 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --bs 16 --n 1000 --cfg 2.0 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt1k_bs16_cfg2.0`
- `run_wmt_cfg1k.log` 2026-11-22T00:56:54-08:00 rc=0 230 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --bs 16 --n 1000 --cfg 3.0 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt1k_bs16_cfg3.0`
- `run_xsum_cfg.log` 2026-11-22T02:12:45-08:00 rc=0 459 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/xsum.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/xsum --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/xsum --n 1000 --bs 16 --cfg 0 --bs 8 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/xsum1k_cfg0`
- `run_xsum_cfg.log` 2026-11-22T02:20:25-08:00 rc=0 1934 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/xsum.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/xsum --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/xsum --n 1000 --bs 16 --cfg 0.5 --bs 8 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/xsum1k_cfg0.5`
- `run_xsum_cfg.log` 2026-11-22T02:52:40-08:00 rc=0 1927 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/xsum.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/xsum --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/xsum --n 1000 --bs 16 --cfg 1.0 --bs 8 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/xsum1k_cfg1.0`
- `run_wmt_sampler.log` 2026-11-22T04:39:49-08:00 rc=0 123 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_greedy`
- `run_wmt_sampler.log` 2026-11-22T04:41:53-08:00 rc=0 123 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --temp 0.6 --top_k 0 --top_p 0.9 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_t0.6_p0.9_r1`
- `run_wmt_sampler.log` 2026-11-22T04:43:58-08:00 rc=0 123 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --temp 0.6 --top_k 0 --top_p 0.9 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_t0.6_p0.9_r2`
- `run_wmt_sampler.log` 2026-11-22T04:46:02-08:00 rc=0 123 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --temp 0.6 --top_k 0 --top_p 0.9 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_t0.6_p0.9_r3`
- `run_wmt_sampler.log` 2026-11-22T04:48:08-08:00 rc=0 127 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --temp 0.8 --top_k 0 --top_p 0.9 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_t0.8_p0.9_r1`
- `run_wmt_sampler.log` 2026-11-22T04:50:16-08:00 rc=0 126 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --temp 0.8 --top_k 0 --top_p 0.9 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_t0.8_p0.9_r2`
- `run_wmt_sampler.log` 2026-11-22T04:52:23-08:00 rc=0 126 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --temp 0.8 --top_k 0 --top_p 0.9 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_t0.8_p0.9_r3`
- `run_wmt_sampler.log` 2026-11-22T04:54:31-08:00 rc=0 125 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --temp 1.0 --top_k 0 --top_p 0.9 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_t1.0_p0.9_r1`
- `run_wmt_sampler.log` 2026-11-22T04:56:38-08:00 rc=0 127 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --temp 1.0 --top_k 0 --top_p 0.9 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_t1.0_p0.9_r2`
- `run_wmt_sampler.log` 2026-11-22T04:58:46-08:00 rc=0 126 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/wmt14.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/wmt14 --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/wmt14 --n 500 --bs 16 --cfg 1.0 --temp 1.0 --top_k 0 --top_p 0.9 --out ~/Downloads/stanford_ncm_turing/experiments/p02_guidance_sweep/wmt500_cfg1.0_t1.0_p0.9_r3`

### p03_level1_probe

- `run.log` 2026-11-21T23:43:43-08:00 rc=0 635 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/scripts/eval_shard.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/tinystories.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/tinystories --nshards 1 --shard 0 --out ~/Downloads/stanford_ncm_turing/experiments/p03_level1_probe/ts_default/shard_0.pkl`
- `run.log` 2026-11-21T23:54:20-08:00 rc=0 60 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/experiments/common/score_genppl.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/tinystories.yaml --glob ~/Downloads/stanford_ncm_turing/experiments/p03_level1_probe/ts_default/shard_*.pkl --dtype fp16`
- `run.log` 2026-11-21T23:55:21-08:00 rc=0 602 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/scripts/eval_shard.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/tinystories.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/tinystories --nshards 1 --l0 uniform --shard 0 --out ~/Downloads/stanford_ncm_turing/experiments/p03_level1_probe/ts_uniform/shard_0.pkl`
- `run.log` 2026-11-22T00:05:25-08:00 rc=0 58 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/experiments/common/score_genppl.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/tinystories.yaml --glob ~/Downloads/stanford_ncm_turing/experiments/p03_level1_probe/ts_uniform/shard_*.pkl --dtype fp16`
- `run.log` 2026-11-22T00:06:24-08:00 rc=0 598 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/scripts/eval_shard.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/tinystories.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/tinystories --nshards 1 --l0 smooth:0.1 --shard 0 --out ~/Downloads/stanford_ncm_turing/experiments/p03_level1_probe/ts_smooth0.1/shard_0.pkl`
- `run.log` 2026-11-22T00:16:24-08:00 rc=0 57 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/experiments/common/score_genppl.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/tinystories.yaml --glob ~/Downloads/stanford_ncm_turing/experiments/p03_level1_probe/ts_smooth0.1/shard_*.pkl --dtype fp16`
- `probe.log` 2026-11-22T00:17:22-08:00 rc=1 58 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python probe.py`
- `probe2.log` 2026-11-22T01:37:05-08:00 rc=0 185 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python probe.py`
- `seed_effect.log` 2026-11-22T01:40:52-08:00 rc=0 2 s:
  `../.venv/Scripts/python seed_effect.py`
- `score_validation.log` 2026-11-22T05:01:08-08:00 rc=0 35 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/experiments/common/score_genppl.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/tinystories.yaml --glob ~/Downloads/stanford_ncm_turing/experiments/p03_level1_probe/ts_default/shard_*.pkl --dtype fp16 --limit 64`
- `score_validation.log` 2026-11-22T05:01:44-08:00 rc=0 353 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ~/Downloads/stanford_ncm_turing/experiments/common/score_genppl.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/tinystories.yaml --glob ~/Downloads/stanford_ncm_turing/experiments/p03_level1_probe/ts_default/shard_*.pkl --dtype fp32 --device cpu --limit 64`
- `probe3.log` 2026-11-22T07:52:56-08:00 rc=0 155 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python probe.py`
- `seed_effect.log` 2026-11-22T07:55:31-08:00 rc=1 2 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python seed_effect.py`
- `seed_effect.log` 2026-11-22T08:00:08-08:00 rc=0 3 s:
  `../.venv/Scripts/python -B seed_effect.py`

### p04_coarse_to_fine

- `run.log` 2026-11-21T17:09:34-08:00 rc=0 241 s:
  `../.venv/Scripts/python viewer.py`
- `fig20_variants.log` 2026-11-21T17:14:08-08:00 rc=0 10 s:
  `../.venv/Scripts/python fig20_variants.py`
- `bf16_hard_full.log` 2026-11-22T00:27:12-08:00 rc=1 85 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python bf16_hard_full.py`
- `bf16_hard_full2.log` 2026-11-22T01:40:11-08:00 rc=0 99 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python bf16_hard_full.py`

### p05_tokenizer_recon

- `run.log` 2026-11-21T17:14:20-08:00 rc=0 4362 s:
  `../.venv/Scripts/python recon.py`
- `pad_check.log` 2026-11-21T23:39:15-08:00 rc=0 267 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python pad_check.py`
- `wmt_fit_check.log` 2026-11-22T07:51:37-08:00 rc=0 79 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python wmt_fit_check.py`

### p06_wallclock

- `run.log` 2026-11-21T22:02:34-08:00 rc=1 5788 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py`
- `run2.log` 2026-11-22T01:10:25-08:00 rc=1 226 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py`
- `run3.log` 2026-11-22T05:07:40-08:00 rc=0 150 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py wmt14`
- `run3.log` 2026-11-22T05:10:11-08:00 rc=0 49 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py tinystories`
- `run3.log` 2026-11-22T05:11:01-08:00 rc=0 35 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py lm1b-small`
- `run3.log` 2026-11-22T05:11:37-08:00 rc=0 188 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py xsum`
- `run4.log` 2026-11-22T08:26:55-08:00 rc=0 208 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py wmt14`
- `run4.log` 2026-11-22T08:30:24-08:00 rc=0 92 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py tinystories`
- `run4.log` 2026-11-22T08:31:57-08:00 rc=0 29 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py lm1b-small`
- `run4.log` 2026-11-22T08:32:26-08:00 rc=0 178 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python timing.py xsum`

### p07_sudoku_trace

- `run.log` 2026-11-21T16:52:07-08:00 rc=0 250 s:
  `../.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/sudoku-hard.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/sudoku/hard --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/sudoku/hard --out ~/Downloads/stanford_ncm_turing/experiments/p07_sudoku_trace/eval_hard`
- `run.log` 2026-11-21T16:56:18-08:00 rc=0 117 s:
  `../.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/sudoku-easy.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/sudoku/easy --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/sudoku/easy --out ~/Downloads/stanford_ncm_turing/experiments/p07_sudoku_trace/eval_easy`
- `run.log` 2026-11-21T16:58:17-08:00 rc=0 130 s:
  `../.venv/Scripts/python ~/Downloads/stanford_ncm_turing/ncm-main/evaluate.py --config ~/Downloads/stanford_ncm_turing/ncm-main/config/eval/sudoku-medium.yaml --checkpoint-dir ~/Downloads/stanford_ncm_turing/experiments/artifacts/sudoku/medium --data-dir ~/Downloads/stanford_ncm_turing/experiments/data/sudoku/medium --out ~/Downloads/stanford_ncm_turing/experiments/p07_sudoku_trace/eval_medium`
- `trace.log` 2026-11-21T17:00:53-08:00 rc=0 502 s:
  `../.venv/Scripts/python trace.py`

### p08_mbr

- `run.log` 2026-11-22T03:24:50-08:00 rc=0 4498 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python mbr.py --n 3003 --bs 16 --temp 0.8 --top_p 0.9`
- `lc_bleu.log` 2026-11-22T05:57:09-08:00 rc=0 84 s:
  `../.venv/Scripts/python -B lc_bleu.py`

### p09_ar_baseline

- `run.log` 2026-11-22T00:18:20-08:00 rc=1 514 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ar_vs_ncm.py`
- `run2.log` 2026-11-22T01:14:13-08:00 rc=0 1371 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ar_vs_ncm.py`
- `run3.log` 2026-11-22T05:14:59-08:00 rc=0 1373 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ar_vs_ncm_small.py`
- `run4.log` 2026-11-22T07:36:30-08:00 rc=0 306 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ar_vs_ncm_small.py`
- `run4.log` 2026-11-22T07:41:37-08:00 rc=0 294 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ar_vs_ncm_small.py`
- `run4.log` 2026-11-22T07:46:39-08:00 rc=0 298 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ar_vs_ncm_small.py`
- `run5.log` 2026-11-22T07:55:34-08:00 rc=0 174 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ar_vs_ncm_small.py`
- `leak_test.log` 2026-11-22T07:58:42-08:00 rc=0 41 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python leak_test.py`
- `leak_test2.log` 2026-11-22T08:09:28-08:00 rc=0 72 s:
  `../.venv/Scripts/python -B leak_test2.py`
- `run6.log` 2026-11-22T08:11:47-08:00 rc=0 908 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python ar_vs_ncm_small.py`

### p10_train_tokenizer

- `timing100.log` 2026-11-22T00:26:56-08:00 rc=1 13 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python train.py --steps 100 --bs 1024 --tag timing100 --eval_every 100`
- `train_real.log` 2026-11-22T01:41:52-08:00 rc=1 15 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python train.py --steps 10000 --bs 1024 --tag real --eval_every 1000`
- `train_aug.log` 2026-11-22T05:00:53-08:00 rc=1 13 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python train.py --steps 10000 --bs 1024 --tag aug --aug --eval_every 1000`
- `timing100.log` 2026-11-22T05:37:53-08:00 rc=0 153 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python train.py --steps 100 --bs 1024 --accum 4 --tag timing100 --eval_every 100`
- `train_real.log` 2026-11-22T05:40:27-08:00 rc=1 100 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python train.py --steps 10000 --bs 1024 --accum 4 --tag real --eval_every 1000`
- `train_real_2500.log` 2026-11-22T05:42:31-08:00 rc=0 3473 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python train.py --steps 2500 --bs 1024 --accum 4 --tag real --eval_every 500`
- `train_aug_2500.log` 2026-11-22T06:40:24-08:00 rc=0 3365 s:
  `~/Downloads/stanford_ncm_turing/experiments/.venv/Scripts/python train.py --steps 2500 --bs 1024 --accum 4 --tag aug --aug --eval_every 500`
