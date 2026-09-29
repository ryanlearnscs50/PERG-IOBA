#!/bin/bash
# Generic array runner: each array element runs PER consecutive lines of $CFG (task arch mode name [args...]),
# 3 CV seeds each, all in parallel on its one GPU. Usage: sbatch --array=0-N%K [partition opts] scripts/run_cfgs.sh CFG PER
#SBATCH --job-name=perg-cfg
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=runs/logs/cfg_%A_%a.out
CFG=$1; PER=${2:-2}
cd "$SLURM_SUBMIT_DIR/src"
export OMP_NUM_THREADS=2
PY=../.venv/bin/python
nvidia-smi -L
for k in $(seq 0 $((PER - 1))); do
  line=$(sed -n "$((SLURM_ARRAY_TASK_ID * PER + k + 1))p" ../$CFG)
  [ -z "$line" ] && continue
  set -- $line; t=$1; a=$2; m=$3; name=$4; shift 4
  for s in ${SEEDS:-0 1 2}; do
    [ -f ../runs/screen/$t/$name/oof_seed$s.npy ] && continue   # resumable
    $PY train_dl.py --task $t --arch $a --mode $m "$@" --seed $s --name $name 2>&1 | grep -v Warn | sed "s/^/[$t $name s$s] /" &
  done
done
wait
