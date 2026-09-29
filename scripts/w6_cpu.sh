#!/bin/bash
#SBATCH --job-name=perg-w6c
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Session 9: classical baseline + classical ensemble members on mac_opt for 10 CV repeats (seeds 0-9).
cd "$SLURM_SUBMIT_DIR/src"
export PERG_SEEDS="0 1 2 3 4 5 6 7 8 9"
PY=../.venv/bin/python
$PY exp_screen.py mac_opt pca_base,lr_mrocket,lr_perg_fpca,lr_new 2>&1 | grep -v Warn
$PY exp_screen.py mac_opt cls_ens 2>&1 | grep -v Warn
