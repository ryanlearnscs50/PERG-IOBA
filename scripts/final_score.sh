#!/bin/bash
#SBATCH --job-name=perg-final
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
# Session 7: score pre-registered ensembles for mac_on (order matters: grand_ens uses the others) + rebuild summary table.
cd "$SLURM_SUBMIT_DIR/src"
PY=../.venv/bin/python
$PY exp_screen.py mac_on dl_cnn_ens,dl_big_ens,dl_w4_ens,grand_ens 2>&1 | grep -v Warn
$PY exp_screen.py table 2>/dev/null | grep -E "task|mac_opt |norm_mac |norm_imd |mac_on "
