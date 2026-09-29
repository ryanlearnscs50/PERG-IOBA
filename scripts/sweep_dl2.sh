#!/bin/bash
#SBATCH --job-name=perg-dl2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/logs/dl2_%A_%a.out
# Wave 3 (fixed before seeing wave-2 pool results): CNNs (resnet, inception) with auxiliary per-eye VA regression,
# and wider (ch 64) variants. 12 configs, 3 per GPU, 3 seeds each.
cd "$SLURM_SUBMIT_DIR/src"
export OMP_NUM_THREADS=2
PY=../.venv/bin/python
CFGS=(
 "norm_mac resnet direct --va_w 0.5" "norm_mac inception direct --va_w 0.5" "norm_mac resnet pool --va_w 0.5"
 "norm_mac inception pool --va_w 0.5" "mac_opt resnet direct --va_w 0.5" "mac_opt inception direct --va_w 0.5"
 "mac_opt resnet pool --va_w 0.5" "mac_opt inception pool --va_w 0.5" "norm_mac resnet direct --ch 64"
 "norm_mac inception direct --ch 64" "mac_opt resnet direct --ch 64" "mac_opt inception direct --ch 64")
for i in $((SLURM_ARRAY_TASK_ID * 3)) $((SLURM_ARRAY_TASK_ID * 3 + 1)) $((SLURM_ARRAY_TASK_ID * 3 + 2)); do
  set -- ${CFGS[$i]}; t=$1; a=$2; m=$3; shift 3
  name=dl_${a}_${m}$(echo "$@" | sed 's/--va_w 0.5/_va/; s/--ch 64/_ch64/; s/ //g')
  for s in 0 1 2; do
    $PY train_dl.py --task $t --arch $a --mode $m "$@" --seed $s --name $name 2>&1 | grep -v Warn | sed "s/^/[$t $name s$s] /" &
  done
done
wait
