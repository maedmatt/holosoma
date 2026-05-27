#!/usr/bin/env bash
# Train one policy, run the policy_eval benchmark on the trained checkpoint,
# log the delta-vs-baseline back into the training's own wandb run.
# Usage: scripts/train_and_bench.sh <bench_tag> <train_command...>
# Note: no -e since IsaacGym segfaults during interpreter teardown even on
# successful runs, so success is signalled by the checkpoint existing
set -uo pipefail

bench_tag="$1"; shift

# 1. Train. --training.name=${bench_tag} puts the tag in the log dir name so
# parallel runs (different bench tags) don't collide on the checkpoint glob
# below. Wandb run path is written into the checkpoint metadata so we can
# find the run again after this process ends. start_ts is used below to
# reject stale ckpts from previous runs if training crashes before saving.
start_ts=$(date +%s)
"$@" --training.name="${bench_tag}" || true

# 2. Most recent checkpoint whose log dir contains the bench tag, written
# after this script started (guards against picking a stale ckpt from a
# prior run if training crashed before saving). Assumes you never run two
# trainings with the same bench_tag in parallel.
ckpt=$(find logs -path "*${bench_tag}*/model_*.pt" -newermt "@${start_ts}" 2>/dev/null \
       | xargs -r ls -t 2>/dev/null | head -1 || true)
if [[ -z "${ckpt}" ]]; then
  echo "error: no checkpoint written after ${start_ts} under logs/*/*${bench_tag}*/model_*.pt; did training crash before saving?"
  exit 1
fi
echo "checkpoint: ${ckpt}"

# 3. Headless benchmark eval with the policy_eval callback. 64 envs with
# per-env startup randomization (mass/friction/COM) so the metrics carry
# a mean ± std across the physics distribution.
python -m holosoma.eval_agent \
  --checkpoint="${ckpt}" \
  --policy-eval.config.enabled \
  --training.name="${bench_tag}" \
  --training.max-eval-steps=16000 \
  --training.headless=True \
  --training.num-envs=64 \
  --simulator.config.highrate-logging-enabled=True \
  randomization:g1-benchmark \
  terrain:terrain-locomotion-plane || true

# 4. Compute metrics on the just-created policy_eval dir, resume the
# training's wandb run (path read from ckpt metadata), log the heatmap +
# per-(scenario, metric) scalars. Same start_ts guard as the ckpt picker
# so a stale eval dir from a prior crashed run can't shadow this one.
eval_dir=$(find logs -type d -name policy_eval -path "*${bench_tag}*" -newermt "@${start_ts}" 2>/dev/null \
           | xargs -r ls -dt 2>/dev/null | head -1 || true)
if [[ -z "${eval_dir}" ]]; then
  echo "error: no policy_eval dir written after ${start_ts} under logs/*/*${bench_tag}*/policy_eval; did eval crash before saving the first scenario?"
  exit 1
fi
python scripts/policy_metrics.py --policy-dir "${eval_dir}" --resume-from-ckpt "${ckpt}"
