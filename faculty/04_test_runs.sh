#!/usr/bin/env bash
# Faculty machine: after the full R1 runs, test both models on the FF++ c23 test crops.
# For each run two checkpoints are tested:
#   best  chosen by the official rule on the validation split (validation accuracy up AND loss down)
#   last  the model after the final epoch (our addition, for information)
# Run:  bash 04_test_runs.sh 2>&1 | tee test_runs.log
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$RUNS/r1_full"
need "$OUT/control.log" "the full runs were not started"
for v in control fixed; do
  if kill -0 "$(cat "$OUT/$v.pid")" 2>/dev/null; then echo "ERROR: R1-$v is still training (see status.sh)"; exit 1; fi
  n=$(tr '\r' '\n' < "$OUT/$v.log" | grep -c "R1_EPOCH" || true)
  echo "R1-$v: $n epochs finished"
  [ "$n" -eq 100 ] || echo "WARNING: R1-$v did not finish 100 epochs; results below are for an unfinished run."
done

T="$RUNS/tester"
if [ ! -d "$T" ]; then clean_copy "$T"; python "$TOOL" patch "$T" test; git -C "$T" diff > "$RUNS/tester.patch"; fi

LOGS="$OUT/test"
for v in control fixed; do
  for kind in best last; do
    W=$(ls "$OUT/$v"/logs/*/PoseEfficientNet_r1full_"$v"_model_"$kind".pth 2>/dev/null | head -n 1 || true)
    [ -n "$W" ] || { echo "ERROR: no $kind checkpoint for R1-$v"; exit 1; }
    python - "$W" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print(sys.argv[1].split('/')[-1], '| saved at epoch', c['epoch'])
PY
    test_weights "${v}_$kind" "$W" "$T" "$LOGS"
  done
done
python "$TOOL" summarize "$LOGS" "$LOGS/results.json"
for v in control fixed; do
  echo "== validation history R1-$v (first 3, last 3 epochs) =="
  tr '\r' '\n' < "$OUT/$v.log" | grep "R1_EPOCH" | sed -n '1,3p;98,100p'
  tr '\r' '\n' < "$OUT/$v.log" | grep "R1_EPOCH" > "$OUT/$v.epochs.txt"
done
echo "TESTS FINISHED. Send test_runs.log back. All numbers are 'our run' (float32, PyTorch 2.7, c23)."
