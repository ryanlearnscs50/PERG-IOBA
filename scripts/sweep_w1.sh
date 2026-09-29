#!/bin/bash
#SBATCH --job-name=perg-w1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/logs/sweep_w1_%j.out
cd "$SLURM_SUBMIT_DIR/src"
nvidia-smi -L
PY=../.venv/bin/python
i=0
run() { CUDA_VISIBLE_DEVICES=$((i % 4)) $PY train.py "$@" > ../runs/logs/$(echo "$@" | sed 's/.*--name //').log 2>&1 & i=$((i+1)); }
for t in bin grp; do
  run --task $t --meta none              --name ${t}_tcn
  run --task $t --meta none --causal 1   --name ${t}_tcn_causal
  run --task $t --meta concat            --name ${t}_tcn_meta_concat
  run --task $t --meta film              --name ${t}_tcn_meta_film
  run --task $t --meta both              --name ${t}_tcn_meta_both
  run --task $t --meta none --ssl 200    --name ${t}_tcn_ssl
  run --task $t --meta film --ssl 200    --name ${t}_tcn_ssl_film
  run --task $t --meta both --ssl 200    --name ${t}_tcn_ssl_both
done
wait
grep -h SUMMARY ../runs/logs/*.log
