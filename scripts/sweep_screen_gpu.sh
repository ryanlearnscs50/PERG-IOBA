#!/bin/bash
#SBATCH --job-name=perg-scr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/logs/sweep_screen_gpu_%j.out
# Screening tasks: best week-2 TCN configs (5-model ensemble per fold + mixup), full and small TCN.
cd "$SLURM_SUBMIT_DIR/src"
nvidia-smi -L
PY=../.venv/bin/python
i=0
run() { CUDA_VISIBLE_DEVICES=$((i % 4)) $PY train.py "$@" > ../runs/logs/scr_$(echo "$@" | sed 's/.*--name //')_$t.log 2>&1 & i=$((i+1)); }
SMALL="--ch 32 --levels 5 --dropout 0.2 --wd 5e-2"
for t in norm_abn norm_mac norm_opt norm_macopt mac_opt; do
  run --task $t --n_ens 5 --mixup 0.4 --out ../runs/screen/$t --name tcn_mix_ens5
  run --task $t --n_ens 5 --mixup 0.4 $SMALL --out ../runs/screen/$t --name tcnS_mix_ens5
done
wait
grep -h SUMMARY ../runs/logs/scr_*.log
