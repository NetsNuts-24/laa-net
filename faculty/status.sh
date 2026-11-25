#!/usr/bin/env bash
# Faculty machine: where are the two R1 runs?   bash status.sh [smoke|full]
set -uo pipefail
WORK="${WORK:-$HOME/laa_visual}"
OUT="$WORK/runs/r1_${1:-full}"
[ -d "$OUT" ] || { echo "no runs in $OUT"; exit 1; }
echo "started: $(cat "$OUT/started.txt" 2>/dev/null)   now: $(date)"
for v in control fixed; do
  pid=$(cat "$OUT/$v.pid" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state="RUNNING (pid $pid)"; else state="NOT RUNNING"; fi
  n=$(tr '\r' '\n' < "$OUT/$v.log" | grep -c "R1_EPOCH")
  echo "== R1-$v: $state, $n epochs finished =="
  tr '\r' '\n' < "$OUT/$v.log" | grep "R1_EPOCH" | tail -n 3
  tr '\r' '\n' < "$OUT/$v.log" | grep "Saved best model" | tail -n 1
  tr '\r' '\n' < "$OUT/$v.log" | grep "Epoch: \[" | tail -n 1 | cut -c1-150
  tr '\r' '\n' < "$OUT/$v.log" | grep -i -E "Traceback|Error|out of memory" | tail -n 3
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
df -h "$WORK" | tail -n 1
