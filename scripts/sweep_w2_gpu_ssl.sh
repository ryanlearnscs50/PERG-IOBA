#!/bin/bash
#SBATCH --job-name=perg-w2s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=runs/logs/sweep_w2_ssl_%j.out
# Week 2: best week-1 4-group config (SSL + age/sex FiLM+concat) with 5-model ensembles.
cd "$SLURM_SUBMIT_DIR/src"
nvidia-smi -L
PY=../.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $PY train.py --task grp --meta both --ssl 200 --n_ens 5 --name w2_grp_tcn_ssl_both_ens5 > ../runs/logs/w2_grp_tcn_ssl_both_ens5.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY train.py --task bin --ssl 200 --n_ens 5 --mixup 0.4 --name w2_bin_tcn_ssl_mix_ens5 > ../runs/logs/w2_bin_tcn_ssl_mix_ens5.log 2>&1 &
wait
