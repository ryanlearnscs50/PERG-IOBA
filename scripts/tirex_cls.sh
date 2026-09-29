#!/bin/bash
#SBATCH --job-name=perg-tirex
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/logs/tirex_cls_%j.out
# Frozen TiRex embeddings (runs/fm_tirex*.npy) + nested-tuned LR/SVM, same protocol as the other FM models.
cd "$SLURM_SUBMIT_DIR/src"
for t in mac_opt norm_mac mac_on; do
  ../.venv/bin/python exp_screen.py $t lr_fm_tirex,lr_fm_tirex_perg,svm_fm_tirex,lr_fm_tirexaug,lr_fm_tirexaug_perg 2>&1 | grep -v Warn
done
