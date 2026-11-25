# Shared settings for the faculty-machine scripts. Sourced, not run.
WORK="${WORK:-$HOME/laa_visual}"
ENV_NAME="${ENV_NAME:-laa310}"
REPO="$WORK/LAA-Net"                                   # official clone, used for crops and landmarks
SAVE="$WORK/datasets_df/FaceForensics++/c23"           # face crops: <split>/frames/<type>/<video>/<frame>.png
RUNS="$WORK/runs"
COMMIT=a33446ab94b3bc0d176a879d43e0700adc52c817
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$HERE/laa_r1.py"
TYPES="Deepfakes Face2Face FaceSwap NeuralTextures"
ALL_TYPES="original,Deepfakes,Face2Face,FaceSwap,NeuralTextures"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

need() { [ -e "$1" ] || { echo "ERROR: missing $1 ($2)"; exit 1; }; }

# clean_copy <dst>: untouched copy of the official code at the pinned commit
clean_copy() {
  git clone -q "$REPO" "$1"
  git -C "$1" checkout -q "$COMMIT"
  [ -z "$(git -C "$1" status --porcelain)" ] || { echo "ERROR: $1 is not clean"; exit 1; }
}

# test_weights <prefix> <weights.pth> <tester_repo> <logs_dir>
# Six test jobs on the FF++ c23 test crops, three per GPU:
#   all_asis    official layout (the four fake versions of a pair are pooled into one video, D020)
#   all_unique  every video scored separately
#   one job per fake type against the 140 real videos
test_weights() {
  local prefix="$1" weights="$2" trepo="$3" logs="$4"
  local uroot="$WORK/unique_root"
  mkdir -p "$logs"
  [ -d "$uroot/test/frames/original" ] || python "$TOOL" unique-root "$SAVE" "$uroot"
  one() {   # gpu tag root types
    local log="$logs/test_$prefix.$2.log"
    if grep -q "Current ACC" "$log" 2>/dev/null; then echo "$prefix.$2: already done"; return 0; fi
    python "$TOOL" config-test "$trepo" "$prefix.$2" --root "$3" --types "$4" --weights "$weights" > /dev/null
    local t0=$SECONDS
    ( cd "$trepo" && CUDA_VISIBLE_DEVICES=$1 python scripts/test.py --cfg "configs/test_$prefix.$2.yaml" 2>&1 ) | grep -v "Inferencing time" > "$log" || true
    grep -q "Current ACC" "$log" || { echo "ERROR: $prefix.$2 gave no result, see $log"; tail -n 5 "$log"; return 1; }
    echo "$prefix.$2: done in $((SECONDS - t0)) s on GPU $1"
  }
  local g1=$([ "$(nvidia-smi -L | wc -l)" -ge 2 ] && echo 1 || echo 0)
  ( one 0 all_asis "$SAVE" "$ALL_TYPES" && one 0 Deepfakes "$SAVE" original,Deepfakes && one 0 FaceSwap "$SAVE" original,FaceSwap ) &
  local p0=$!
  ( one "$g1" all_unique "$uroot" "$ALL_TYPES" && one "$g1" Face2Face "$SAVE" original,Face2Face && one "$g1" NeuralTextures "$SAVE" original,NeuralTextures ) &
  local p1=$!
  local fail=0
  wait $p0 || fail=1
  wait $p1 || fail=1
  return $fail
}
