#!/usr/bin/env bash
# Faculty machine: where are the R1 runs?   bash status.sh [smoke|full]
set -uo pipefail
WORK="${WORK:-$HOME/laa_visual}"
OUT="$WORK/runs/r1_${1:-full}"
echo "now: $(date)"
[ -d "$OUT" ] || { echo "no runs yet in $OUT"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv; exit 0; }
for v in control fixed; do
  log="$OUT/$v.log"
  if [ ! -f "$log" ]; then echo "== R1-$v: not started =="; continue; fi
  pid=$(cat "$OUT/$v.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state="RUNNING (pid $pid)"; else state="NOT RUNNING"; fi
  n=$(tr '\r' '\n' < "$log" | grep -c "R1_EPOCH" || true)
  echo "== R1-$v: $state, $n epochs finished, started $(cat "$OUT/$v.started.txt" 2>/dev/null) =="
  tr '\r' '\n' < "$log" | grep "R1_EPOCH" | tail -n 3
  tr '\r' '\n' < "$log" | grep "Saved best model" | tail -n 1
  tr '\r' '\n' < "$log" | grep "Epoch: \[" | tail -n 1 | cut -c1-150
  tr '\r' '\n' < "$log" | grep -i -E "Traceback|Error|out of memory" | tail -n 3
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
df -h "$WORK" | tail -n 1
