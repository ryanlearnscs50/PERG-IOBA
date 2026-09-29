#!/bin/bash
# Show nvidia-smi for a running slurm GPU job, plus which run each GPU process is.
# Usage: scripts/gpuwatch.sh <jobid>
J=${1:?jobid}
srun --jobid=$J --overlap --cpu-bind=none bash -c '
nvidia-smi
echo; echo "GPU(slot)  PID     RUN"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null; pgrep -u $USER -f "python train[.]py"); do
  [ -r /proc/$p/cmdline ] || continue
  dev=$(tr "\0" "\n" < /proc/$p/environ | grep ^CUDA_VISIBLE_DEVICES= | cut -d= -f2)
  echo "   $dev       $p  $(tr "\0" " " < /proc/$p/cmdline | sed "s/.*--name //")"
done | sort -u'
