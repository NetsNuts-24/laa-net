#!/usr/bin/env bash
# Faculty machine: check the new environment before any training is trusted.
# Tests the AUTHORS' released SBI weights with the official test code (float32, PyTorch 2.7) on the
# FF++ c23 test crops made by 01_prepare_data.sh, and compares with our Kaggle result R0 (D022,
# official environment, float64). If the numbers agree, the environment, the crops and float32 are fine.
# Run:  bash 02_check_port.sh 2>&1 | tee check_port.log
# Uses weights/released_sbi_model_best.pth next to this script (from the USB bundle), or a path given
# as the first argument. It is the SBI file inside the authors' "LAA-Net pretrained.zip" (435 MB).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

W_IN="${1:-$HERE/weights/released_sbi_model_best.pth}"
need "$W_IN" "the released SBI weights file"
SHA=fde726610e778045f65c3fbe839bc463c4d3a9970f72ca109d5911917305c991
need "$SAVE/test/frames/original" "run 01_prepare_data.sh first"
mkdir -p "$WORK/weights" "$RUNS"
W="$WORK/weights/released_sbi_model_best.pth"
[ -f "$W" ] || cp "$W_IN" "$W"
echo "$SHA  $W" | sha256sum -c -

T="$RUNS/tester"
if [ ! -d "$T" ]; then clean_copy "$T"; python "$TOOL" patch "$T" test; git -C "$T" diff > "$RUNS/tester.patch"; fi

LOGS="$RUNS/port_check"
test_weights released "$W" "$T" "$LOGS"
# Stops here (exit code 1) if any AUC is more than 1.0 away from R0: then the crops, the environment
# or float32 changed the model's behaviour and that must be understood before training.
python "$TOOL" summarize "$LOGS" "$LOGS/results.json" --compare-r0 --max-diff 1.0
echo "PORT CHECK FINISHED. Send check_port.log (or the table above) back."
