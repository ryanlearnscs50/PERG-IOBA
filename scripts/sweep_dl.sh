#!/bin/bash
#SBATCH --job-name=perg-dl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=runs/logs/dl_%A_%a.out
# Deep screening sweep (fixed a priori, all configs reported): task x arch x mode x hybrid, 3 configs x 3 seeds in parallel on 1 GPU.
cd "$SLURM_SUBMIT_DIR/src"
export OMP_NUM_THREADS=2
PY=../.venv/bin/python
for i in $((SLURM_ARRAY_TASK_ID * 3)) $((SLURM_ARRAY_TASK_ID * 3 + 1)) $((SLURM_ARRAY_TASK_ID * 3 + 2)); do
tasks=(norm_mac mac_opt); archs=(tcn inception resnet); modes=(direct pool); hybs=(0 1)
t=${tasks[$((i % 2))]}; a=${archs[$(((i / 2) % 3))]}; m=${modes[$(((i / 6) % 2))]}; h=${hybs[$(((i / 12) % 2))]}
name=dl_${a}_${m}$([ $h = 1 ] && echo _hyb)
nvidia-smi -L; echo $t $name
for s in 0 1 2; do
  $PY train_dl.py --task $t --arch $a --mode $m --hybrid $h --seed $s --name $name 2>&1 | grep -v Warn | sed "s/^/[$t $name s$s] /" &
done
done
wait
