#!/bin/bash
#SBATCH --job-name=perg-w4fin
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=runs/logs/w4_final_%j.out
# After wave 4: score pre-registered session-5 ensembles (order matters: grand_ens uses the others' OOFs).
cd "$SLURM_SUBMIT_DIR/src"
PY=../.venv/bin/python
for t in mac_opt norm_mac norm_imd; do $PY exp_screen.py $t dl_w4_ens,cls_ens,grand_ens 2>&1 | grep -v Warn; done
$PY exp_screen.py table 2>/dev/null | grep -E "task|mac_opt |norm_mac |norm_imd "
echo "incomplete configs (missing seeds):"
while read t a m name rest; do for s in 0 1 2; do [ -f ../runs/screen/$t/$name/oof_seed$s.npy ] || echo "  $t $name seed$s"; done; done < ../scripts/w4_cfgs.txt
