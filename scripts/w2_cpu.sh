#!/bin/bash
#SBATCH --job-name=perg-w2c
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/logs/w2_cpu_%j.out
# Week 2 CPU work: literature-protocol leakage check, then MiniRocket / DWT / FPCA feature models (nested-tuned LR).
cd "$SLURM_SUBMIT_DIR/src"
PY=../.venv/bin/python
$PY exp_w2.py leak
$PY exp_w2.py feats
