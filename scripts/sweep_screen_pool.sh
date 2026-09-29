#!/bin/bash
#SBATCH --job-name=perg-scp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/logs/sweep_screen_pool_%j.out
# Screening TCN trained on all records outside the test subjects (label = abnormal), evaluated per task.
cd "$SLURM_SUBMIT_DIR/src"
PY=../.venv/bin/python
i=0
for t in norm_mac norm_opt norm_macopt; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) $PY train.py --task $t --pool_all 1 --n_ens 5 --mixup 0.4 --out ../runs/screen/$t \
    --name tcn_mix_ens5@all > ../runs/logs/scr_tcn_mix_ens5@all_$t.log 2>&1 & i=$((i+1))
done
wait
grep -h SUMMARY ../runs/logs/scr_*@all*.log
