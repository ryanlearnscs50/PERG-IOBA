#!/bin/bash
#SBATCH --job-name=perg-w2g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/logs/sweep_w2_gpu_%j.out
# Week 2: variance reduction (5-model ensembles per fold), smaller/regularised TCN, mixup.
cd "$SLURM_SUBMIT_DIR/src"
nvidia-smi -L
PY=../.venv/bin/python
i=0
run() { CUDA_VISIBLE_DEVICES=$((i % 4)) $PY train.py "$@" > ../runs/logs/$(echo "$@" | sed 's/.*--name //').log 2>&1 & i=$((i+1)); }
SMALL="--ch 32 --levels 5 --dropout 0.2 --wd 5e-2"
for t in bin grp; do
  run --task $t --n_ens 5                          --name w2_${t}_tcn_ens5
  run --task $t --n_ens 5 --mixup 0.4              --name w2_${t}_tcn_mix_ens5
  run --task $t --n_ens 5 $SMALL                   --name w2_${t}_tcnS_ens5
  run --task $t --n_ens 5 $SMALL --mixup 0.4       --name w2_${t}_tcnS_mix_ens5
done
wait
grep -h SUMMARY ../runs/logs/w2_*.log
