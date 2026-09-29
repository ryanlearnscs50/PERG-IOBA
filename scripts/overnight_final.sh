#!/bin/bash
#SBATCH --job-name=perg-fin
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=runs/logs/overnight_final_%j.out
# After the overnight sweeps: score pre-registered ensembles and write runs/screen/summary.csv.
cd "$SLURM_SUBMIT_DIR/src"
PY=../.venv/bin/python
for t in mac_opt norm_mac norm_imd; do $PY exp_screen.py $t dl_cnn_ens,dl_big_ens,dl_big_cwt 2>&1 | grep -v Warn; done
$PY exp_screen.py table 2>/dev/null | grep -E "task|mac_opt |norm_mac |norm_imd "
echo "incomplete configs (missing seeds):"
while read t a m name rest; do for s in 0 1 2; do [ -f ../runs/screen/$t/$name/oof_seed$s.npy ] || echo "  $t $name seed$s"; done; done < ../scripts/overnight_cfgs.txt
