#!/usr/bin/env bash
# Faculty machine: the R1 training runs (DECISIONS.md D029, D031, D033).
#   R1-control  official method
#   R1-fixed    official method + frame-resampling fix
# Same data, seed (317), epochs, loss weights (10 / 100), float32, environment. The two runs are
# independent, so they may run side by side, one after the other, or on different GPUs.
#
#   bash 03_train.sh <smoke|full> [control|fixed|both]
#     smoke  2 short epochs, waits, prints timings
#     full   100 epochs, returns at once (training continues in the background)
#   Which GPUs may be used is set with GPUS (default "0 1"):
#     bash 03_train.sh smoke                    both runs, control on GPU 0 and fixed on GPU 1
#     GPUS=1 bash 03_train.sh smoke             both runs on GPU 1, one after the other
#     GPUS=1 bash 03_train.sh full control      control alone on GPU 1
#     GPUS=0 bash 03_train.sh full fixed        fixed alone on GPU 0
#
# Download on first start (approved, D032): EfficientNet-B4 AdvProp weights, 78 MB,
# github.com/lukemelas/EfficientNet-PyTorch, fetched by the official code.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODE="${1:?say smoke or full}"
WHICH="${2:-both}"
case "$MODE" in
  smoke) EPOCHS=2; WARM=1 ;;       # epoch 0 backbone frozen, epoch 1 unfrozen and frames redrawn
  full)  EPOCHS=100; WARM=6 ;;     # as in the paper and the released config
  *) echo "say smoke or full"; exit 1 ;;
esac
case "$WHICH" in
  control|fixed) VARIANTS="$WHICH" ;;
  both) VARIANTS="control fixed" ;;
  *) echo "say control, fixed or both"; exit 1 ;;
esac
if [ "$WHICH" = both ] && [ "$MODE" = full ] && [ "${#GPU_LIST[@]}" -lt 2 ]; then
  echo "ERROR: one GPU given (GPUS=$GPUS). Start the full runs one at a time, for example:"
  echo "  GPUS=$GPUS bash 03_train.sh full control"
  exit 1
fi
TRAIN_ANNO="$REPO/processed_data/c23/train_FaceForensics_original_81.json"
VAL_ANNO="$REPO/processed_data/c23/val_FaceForensics_original_81.json"
need "$TRAIN_ANNO" "run 01_prepare_data.sh first"; need "$VAL_ANNO" "run 01_prepare_data.sh first"
need "$SAVE/test/frames/original" "run 01_prepare_data.sh first"

OUT="$RUNS/r1_$MODE"
mkdir -p "$OUT"
# control takes the first GPU in GPUS; fixed takes the second if there is one, else the first
gpu_of() { if [ "$1" = fixed ] && [ "$WHICH" = both ]; then echo "$G_B"; else echo "$G_A"; fi; }
running() { [ -f "$OUT/$1.pid" ] && kill -0 "$(cat "$OUT/$1.pid")" 2>/dev/null; }

for v in $VARIANTS; do
  D="$OUT/$v"
  if [ -f "$OUT/$v.log" ]; then echo "ERROR: $OUT/$v.log exists; R1-$v ($MODE) was already started. Ask before starting it again."; exit 1; fi
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
if [ -f "$OUT/control.yaml" ] && [ -f "$OUT/fixed.yaml" ]; then
  if diff <(grep -v '^TASK:' "$OUT/control.yaml") <(grep -v '^TASK:' "$OUT/fixed.yaml"); then
    echo "configs of control and fixed are identical apart from the task name"
  else
    echo "ERROR: the two configs differ in more than the task name"; exit 1
  fi
fi
pip freeze > "$OUT/env_freeze.txt"
cp "$WORK"/split_*.json "$OUT/" 2>/dev/null || true

start() {   # variant
  local v="$1" D="$OUT/$1" G
  G=$(gpu_of "$1")
  ( cd "$D" && CUDA_VISIBLE_DEVICES=$G nohup python scripts/train.py --cfg "configs/r1${MODE}_$v.yaml" > "$OUT/$v.log" 2>&1 &
    echo $! > "$OUT/$v.pid" )
  date > "$OUT/$v.started.txt"
  echo "started R1-$v on GPU $G, pid $(cat "$OUT/$v.pid"), log $OUT/$v.log"
}

if [ "$MODE" = full ]; then
  for v in $VARIANTS; do start "$v"; done
  echo
  echo "Training continues in the background and keeps going if this window is closed."
  echo "Progress:  bash $HERE/status.sh        Stop a run:  kill \$(cat $OUT/<control|fixed>.pid)"
  exit 0
fi

if [ "${#GPU_LIST[@]}" -ge 2 ]; then          # smoke, two GPUs: side by side
  for v in $VARIANTS; do start "$v"; done
  echo "waiting for the smoke run(s)..."
  for v in $VARIANTS; do while running "$v"; do sleep 20; done; done
else                                          # smoke, one GPU: one after the other
  for v in $VARIANTS; do
    start "$v"
    echo "waiting for the smoke run of R1-$v..."
    while running "$v"; do sleep 20; done
  done
fi
FAIL=0
for v in $VARIANTS; do
  echo "== R1-$v =="
  tr '\r' '\n' < "$OUT/$v.log" | grep -E "R1_EPOCH|Saved best model|Reloading data" || true
  python "$TOOL" timing "$OUT/$v.log" || FAIL=1
  [ "$(tr '\r' '\n' < "$OUT/$v.log" | grep -c "R1_EPOCH")" -eq 2 ] || { echo "ERROR: R1-$v did not finish 2 epochs"; FAIL=1; }
  ls "$OUT/$v"/logs/*/*_model_last.pth > /dev/null 2>&1 || { echo "ERROR: no checkpoint written"; FAIL=1; }
done
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
[ "$FAIL" -eq 0 ] || { echo "SMOKE RUN FAILED: last lines of the logs follow"; for v in $VARIANTS; do tr '\r' '\n' < "$OUT/$v.log" | grep -v 'it/s\|s/it' | tail -n 15; done; exit 1; }
echo "SMOKE RUN OK ($VARIANTS). Send the log back before starting the full run."
