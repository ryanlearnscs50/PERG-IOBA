#!/bin/bash
#SBATCH --job-name=perg-w6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
# Session 9: plain-ResNet candidates for mac_opt (scripts/w6_cfgs.txt), 10 CV repeats. Array element i ->
# config i/3, seed chunk i%3. Resumable (skips seeds whose oof exists).
CHUNKS=("0 1 2 3" "4 5 6" "7 8 9")
export SEEDS="${CHUNKS[$((SLURM_ARRAY_TASK_ID % 3))]}"
export SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID / 3))
bash "$SLURM_SUBMIT_DIR/scripts/run_cfgs.sh" scripts/w6_cfgs.txt 1
