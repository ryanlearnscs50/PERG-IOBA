#!/bin/bash
#SBATCH --job-name=perg-imd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=runs/logs/dl_imd_%A_%a.out
# Normal vs inherited macular dystrophy: the 4 dl_cnn_ens members, 2 configs x 3 seeds per GPU.
cd "$SLURM_SUBMIT_DIR/src"
export OMP_NUM_THREADS=2
PY=../.venv/bin/python
m=$([ $SLURM_ARRAY_TASK_ID = 0 ] && echo direct || echo pool)
for a in resnet inception; do for s in 0 1 2; do
  $PY train_dl.py --task norm_imd --arch $a --mode $m --seed $s --name dl_${a}_${m} 2>&1 | grep -v Warn | sed "s/^/[$a $m s$s] /" &
done; done
wait
