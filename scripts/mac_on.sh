#!/bin/bash
#SBATCH --job-name=perg-macon
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/logs/mac_on_%j.out
# Classical models on mac_on (macular vs optic neuropathies proper), same list as used for mac_opt.
cd "$SLURM_SUBMIT_DIR/src"
../.venv/bin/python exp_screen.py mac_on pca_base,pca_tuned,lr_perg,lr_perg_fpca,lr_new,lr_new_fft,lr_mrocket,cls_ens 2>&1 | grep -v Warn
