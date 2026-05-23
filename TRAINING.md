# Run book

How to train + auto-benchmark policies.

Each command wraps `python -m holosoma.train_agent ...` in `scripts/train_and_bench.sh <tag>`. The wrapper trains the policy, then runs the `policy_eval` benchmark on the resulting checkpoint and logs the delta-vs-baseline heatmap back into the same wandb run under `bench/delta_heatmap`.

Common to all runs:
- flat terrain, symmetry disabled, wandb project `IDSIA`
- noise predictor ckpt for the quiet variant: `models/v11_sim_no_cmd_all36/best.pt`, weight `-600` (both can be overridden on the CLI)

## Workstation (video logged to wandb)

### 1. PPO baseline

```bash
scripts/train_and_bench.sh ppo_baseline \
  python -m holosoma.train_agent exp:g1-23dof \
    simulator:isaacgym terrain:terrain-locomotion-plane logger:wandb \
    --training.project IDSIA \
    --algo.config.use-symmetry False \
    --logger.tags "('baseline', 'ppo', 'no-symmetry', 'flat-terrain')"
```

### 2. FastSAC baseline

```bash
scripts/train_and_bench.sh fastsac_baseline \
  python -m holosoma.train_agent exp:g1-23dof-fast-sac \
    simulator:isaacgym terrain:terrain-locomotion-plane logger:wandb \
    --training.project IDSIA \
    --algo.config.use-symmetry False \
    --logger.tags "('baseline', 'fastsac', 'no-symmetry', 'flat-terrain')"
```

### 3. FastSAC + noise predictor

```bash
scripts/train_and_bench.sh fastsac_noise_pred_w600 \
  python -m holosoma.train_agent exp:g1-23dof-quiet-fast-sac \
    simulator:isaacgym terrain:terrain-locomotion-plane logger:wandb \
    --training.project IDSIA \
    --algo.config.use-symmetry False \
    --reward.terms.penalty_noise.params.noise_predictor_ckpt=models/v11_sim_no_cmd_all36/best.pt \
    --reward.terms.penalty_noise.weight=-600.0 \
    --logger.tags "('noise-predictor', 'fastsac', 'no-symmetry', 'flat-terrain')"
```

## Cluster (no video)

Same commands, plus `--logger.video.enabled False` to skip video recording and upload.

### 1. PPO baseline

```bash
scripts/train_and_bench.sh ppo_baseline \
  python -m holosoma.train_agent exp:g1-23dof \
    simulator:isaacgym terrain:terrain-locomotion-plane logger:wandb \
    --training.project IDSIA \
    --algo.config.use-symmetry False \
    --logger.video.enabled False \
    --logger.tags "('baseline', 'ppo', 'no-symmetry', 'flat-terrain', 'cluster')"
```

### 2. FastSAC baseline

```bash
scripts/train_and_bench.sh fastsac_baseline \
  python -m holosoma.train_agent exp:g1-23dof-fast-sac \
    simulator:isaacgym terrain:terrain-locomotion-plane logger:wandb \
    --training.project IDSIA \
    --algo.config.use-symmetry False \
    --logger.video.enabled False \
    --logger.tags "('baseline', 'fastsac', 'no-symmetry', 'flat-terrain', 'cluster')"
```

### 3. FastSAC + noise predictor

```bash
scripts/train_and_bench.sh fastsac_noise_pred_w600 \
  python -m holosoma.train_agent exp:g1-23dof-quiet-fast-sac \
    simulator:isaacgym terrain:terrain-locomotion-plane logger:wandb \
    --training.project IDSIA \
    --algo.config.use-symmetry False \
    --logger.video.enabled False \
    --reward.terms.penalty_noise.params.noise_predictor_ckpt=models/v11_sim_no_cmd_all36/best.pt \
    --reward.terms.penalty_noise.weight=-600.0 \
    --logger.tags "('noise-predictor', 'fastsac', 'no-symmetry', 'flat-terrain', 'cluster')"
```
