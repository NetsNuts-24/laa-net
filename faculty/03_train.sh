#!/usr/bin/env bash
# Faculty machine: R1 = two training runs side by side (DECISIONS.md D029, D031).
#   R1-control  official method                         -> GPU 0
#   R1-fixed    official method + frame-resampling fix  -> GPU 1
# Same data, seed (317), epochs, loss weights (10 / 100), float32, environment.
#
#   bash 03_train.sh smoke 2>&1 | tee train_smoke.log    2 short epochs of each, waits, prints timings
#   bash 03_train.sh full  2>&1 | tee train_full.log     100 epochs of each, returns at once (runs in background)
#
# Download on first start (approved by the user first): EfficientNet-B4 AdvProp weights, 78 MB,
# github.com/lukemelas/EfficientNet-PyTorch, fetched by the official code.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODE="${1:?say smoke or full}"
case "$MODE" in
  smoke) EPOCHS=2; WARM=1 ;;       # epoch 0 backbone frozen, epoch 1 unfrozen and frames redrawn
  full)  EPOCHS=100; WARM=6 ;;     # as in the paper and the released config
  *) echo "say smoke or full"; exit 1 ;;
esac
TRAIN_ANNO="$REPO/processed_data/c23/train_FaceForensics_original_81.json"
VAL_ANNO="$REPO/processed_data/c23/val_FaceForensics_original_81.json"
need "$TRAIN_ANNO" "run 01_prepare_data.sh first"; need "$VAL_ANNO" "run 01_prepare_data.sh first"
need "$SAVE/test/frames/original" "run 01_prepare_data.sh first"
[ "$(nvidia-smi -L | wc -l)" -ge 2 ] || { echo "ERROR: two GPUs expected"; exit 1; }

OUT="$RUNS/r1_$MODE"
mkdir -p "$OUT"
GPU_OF() { [ "$1" = control ] && echo 0 || echo 1; }

for v in control fixed; do
  D="$OUT/$v"
  if [ -f "$OUT/$v.log" ]; then echo "ERROR: $OUT/$v.log exists; this run was already started. Move $OUT away to start again."; exit 1; fi
  if [ ! -d "$D" ]; then
    clean_copy "$D"
    python "$TOOL" patch "$D" "$v"
    git -C "$D" diff > "$OUT/$v.patch"                      # exact code difference from the official commit
  fi
  python "$TOOL" config-train "$D" "r1${MODE}_$v" --data-root "$SAVE" --train-anno "$TRAIN_ANNO" --val-anno "$VAL_ANNO" \
    --epochs "$EPOCHS" --warm-up "$WARM" --hm-lmda 10 --cstency-lmda 100
  cp "$D/configs/r1${MODE}_$v.yaml" "$OUT/$v.yaml"
  ( cd "$D" && python "$TOOL" probe "configs/r1${MODE}_$v.yaml" "$v" ) 2>&1 | grep -v -i "warn" | tail -n 12
done
if diff <(grep -v '^TASK:' "$OUT/control.yaml") <(grep -v '^TASK:' "$OUT/fixed.yaml"); then
  echo "configs identical apart from the task name"
else
  echo "ERROR: the two configs differ in more than the task name"; exit 1
fi
pip freeze > "$OUT/env_freeze.txt"
cp "$WORK"/split_*.json "$OUT/" 2>/dev/null || true

for v in control fixed; do
  D="$OUT/$v"; G=$(GPU_OF $v)
  ( cd "$D" && CUDA_VISIBLE_DEVICES=$G nohup python scripts/train.py --cfg "configs/r1${MODE}_$v.yaml" > "$OUT/$v.log" 2>&1 &
    echo $! > "$OUT/$v.pid" )
  echo "started R1-$v on GPU $G, pid $(cat "$OUT/$v.pid"), log $OUT/$v.log"
done
date > "$OUT/started.txt"

if [ "$MODE" = full ]; then
  echo
  echo "Both runs are training in the background and keep going if this window is closed."
  echo "Progress:  bash $HERE/status.sh        Stop both:  kill \$(cat $OUT/control.pid $OUT/fixed.pid)"
  exit 0
fi

echo "waiting for the two smoke runs..."
while kill -0 "$(cat "$OUT/control.pid")" 2>/dev/null || kill -0 "$(cat "$OUT/fixed.pid")" 2>/dev/null; do sleep 20; done
FAIL=0
for v in control fixed; do
  echo "== R1-$v =="
  tr '\r' '\n' < "$OUT/$v.log" | grep -E "R1_EPOCH|Saved best model|Reloading data" || true
  python "$TOOL" timing "$OUT/$v.log" || FAIL=1
  ls "$OUT/$v"/logs/*/*_model_last.pth > /dev/null 2>&1 || { echo "ERROR: no checkpoint written"; FAIL=1; }
done
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
[ "$FAIL" -eq 0 ] || { echo "SMOKE RUN FAILED: last lines of the logs follow"; for v in control fixed; do tr '\r' '\n' < "$OUT/$v.log" | grep -v 'it/s\|s/it' | tail -n 15; done; exit 1; }
echo "SMOKE RUN OK. Send train_smoke.log (or everything printed above) back before starting the full run."
