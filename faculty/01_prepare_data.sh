#!/usr/bin/env bash
# Faculty machine: face crops and landmarks for R1 (train, val) and for the test check.
# Needs: 00_setup_env.sh finished with "ENVIRONMENT OK", and the FF++ c23 videos on this machine.
# Run:  bash 01_prepare_data.sh /path/to/FaceForensics++_C23 2>&1 | tee prepare_data.log
# Safe to run again: finished steps are skipped.
#
# Downloads it makes (each approved by the user first, see DECISIONS.md):
#   - dlib 81-landmark file, 19.7 MB, github.com/codeniko/shape_predictor_81_face_landmarks
#     (not downloaded if shape_predictor_81_face_landmarks.dat lies next to this script, as in the USB bundle)
#   - RetinaFace detector weights, 96.9 MB, github.com/ternaus/retinaface (fetched by the crop script)
#   - official FF++ split lists train/val/test.json (small text files), github.com/ondyari/FaceForensics
set -euo pipefail

FF="${1:?give the FF++ c23 folder: either the official layout (original_sequences/, manipulated_sequences/) or the Kaggle layout (original/, Deepfakes/, ...)}"
FF="$(readlink -f "$FF")"
if [ -d "$FF/original_sequences/youtube/c23/videos" ]; then
  LAYOUT=official      # as written by the FF++ authors' download script (the user's copy, D030)
  src() { if [ "$1" = original ]; then echo "$FF/original_sequences/youtube/c23/videos"; else echo "$FF/manipulated_sequences/$1/c23/videos"; fi; }
else
  LAYOUT=kaggle        # xdxd003/ff-c23
  src() { echo "$FF/$1"; }
fi
echo "FF++ folder: $FF ($LAYOUT layout)"
WORK="${WORK:-$HOME/laa_visual}"
ENV_NAME="${ENV_NAME:-laa310}"
REPO="$WORK/LAA-Net"
ROOT="$WORK/datasets_df"
OD="$ROOT/FaceForensics++/original_download"
SAVE="$ROOT/FaceForensics++/c23"
LOGS="$WORK/prep_logs"
DLIB_SHA=8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869
TYPES="Deepfakes Face2Face FaceSwap NeuralTextures"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
mkdir -p "$LOGS" "$OD"
[ -d "$REPO/.git" ] || { echo "ERROR: $REPO missing, run 00_setup_env.sh first"; exit 1; }

echo "== 1. videos =="
for d in original $TYPES; do
  n=$(find -L "$(src $d)" -maxdepth 1 -name '*.mp4' | wc -l)
  echo "$d: $n mp4"
  [ "$n" -eq 1000 ] || { echo "ERROR: expected 1000 videos in $(src $d)"; exit 1; }
done

echo "== 2. landmark model =="
LM="$REPO/pretrained/shape_predictor_81_face_landmarks.dat"
mkdir -p "$(dirname "$LM")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$LM" ] && [ -f "$HERE/shape_predictor_81_face_landmarks.dat" ]; then
  cp "$HERE/shape_predictor_81_face_landmarks.dat" "$LM"      # copy brought on the USB drive, no download
fi
if [ ! -f "$LM" ]; then
  wget -q -O "$LM" https://github.com/codeniko/shape_predictor_81_face_landmarks/raw/master/shape_predictor_81_face_landmarks.dat
fi
echo "$DLIB_SHA  $LM" | sha256sum -c -

echo "== 3. folder layout and official splits =="
link() { mkdir -p "$(dirname "$2")"; [ -L "$2" ] || ln -s "$1" "$2"; }
link "$(src original)" "$OD/original_sequences/youtube/c23/videos"
for t in $TYPES; do link "$(src $t)" "$OD/manipulated_sequences/$t/c23/videos"; done
for s in train val test; do
  [ -f "$OD/$s.json" ] || wget -q -O "$OD/$s.json" "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/$s.json"
  cp "$OD/$s.json" "$WORK/split_$s.json"          # keep the split with the results
done
python - "$OD" <<'PY'
import json, sys
want = {'train': 360, 'val': 70, 'test': 70}
seen = set()
for s, n in want.items():
    p = json.load(open(f'{sys.argv[1]}/{s}.json'))
    ids = {v for pair in p for v in pair}
    assert len(p) == n and len(ids) == 2 * n and not (ids & seen), (s, len(p))
    seen |= ids
    print(s, 'pairs', len(p), '| real videos', len(ids))
PY

echo "== 4. crop script: two path constants (same change as in R0) =="
C="$REPO/package_utils/images_crop.py"
sed -i "s#^ROOT = '/data/deepfake_cluster/datasets_df'#ROOT = '$ROOT'#; s#FaceForensics++/c0'#FaceForensics++/c23'#" "$C"
grep -n -E "^ROOT = |^SAVE_DIR = " "$C"
grep -q "^ROOT = '$ROOT'" "$C" && grep -q "FaceForensics++/c23'" "$C" || { echo "ERROR: path edit failed"; exit 1; }

echo "== 5. face crops (GPU 0: train; GPU 1: val, then the five test folders) =="
crop() {   # gpu dataset frames split
  local tag="crop_$4_$2"
  if [ -f "$LOGS/$tag.done" ]; then echo "$tag: already done"; return 0; fi
  local t0=$SECONDS
  ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$1 python package_utils/images_crop.py -d "$2" -c c23 -n "$3" -t "$4" ) > "$LOGS/$tag.log" 2>&1
  touch "$LOGS/$tag.done"
  echo "$tag: $((SECONDS - t0)) s" | tee -a "$LOGS/times.txt"
}
NGPU=$(nvidia-smi -L | wc -l)
G1=$([ "$NGPU" -ge 2 ] && echo 1 || echo 0)
crop 0 Original 128 train &
P0=$!
( crop "$G1" Original 32 val; crop "$G1" Original 32 test; for t in $TYPES; do crop "$G1" "$t" 32 test; done ) &
P1=$!
FAIL=0
wait $P0 || FAIL=1
wait $P1 || FAIL=1
[ "$FAIL" -eq 0 ] || { echo "ERROR: a crop job failed, see $LOGS/crop_*.log"; tail -n 5 "$LOGS"/crop_*.log; exit 1; }

echo "== 6. folder name and counts =="
for s in train val test; do
  if [ -d "$SAVE/$s/frames/Original" ]; then mv "$SAVE/$s/frames/Original" "$SAVE/$s/frames/original"; fi   # loaders expect 'original'
done
python - "$SAVE" "$WORK" <<'PY'
import glob, json, os, sys
save, work = sys.argv[1:3]
want = {'train': (['original'], 720, 128), 'val': (['original'], 140, 32),
        'test': (['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'], 140, 32)}
out, bad = [], False
for split, (folders, nvid, nfr) in want.items():
    for f in folders:
        vids = sorted(os.listdir(f'{save}/{split}/frames/{f}'))
        c = [len(glob.glob(f'{save}/{split}/frames/{f}/{v}/*.png')) for v in vids]
        short = [v for v, k in zip(vids, c) if k < nfr]
        out.append({'split': split, 'folder': f, 'videos': len(vids), 'frames': sum(c), 'min_frames': min(c) if c else 0,
                    'videos_with_fewer_frames': short})
        print(split, f, 'videos', len(vids), '| frames', sum(c), '| videos short of', nfr, 'frames:', len(short))
        bad |= len(vids) != nvid
json.dump(out, open(f'{work}/crop_counts.json', 'w'), indent=1)
if bad:
    sys.exit('ERROR: a folder does not have the expected number of videos')
PY

echo "== 7. landmarks for train and val (dlib, 81 points; CPU) =="
FREE_RAM=$(free -g | awk '/^Mem:/{print $7}')
echo "available RAM: ${FREE_RAM} GB (the official script keeps all train images in memory, about 20 GB)"
[ "$FREE_RAM" -ge 28 ] || echo "WARNING: little RAM; if the train landmark job is killed, tell Claude."
mkdir -p "$REPO/processed_data"
lm() {   # split
  local out="$REPO/processed_data/c23/$1_FaceForensics_original_81.json"
  if [ -f "$out" ]; then echo "landmarks $1: already done"; return 0; fi
  cat > "$REPO/configs/r1_lm_$1.yaml" <<YAML
PREPROCESSING:
  DATASET: FaceForensics
  COMPRESSION: c23
  SPLIT: $1
  ROOT: $SAVE/
  FAKETYPE: [original]
  IMAGE_SUFFIX: png
  DATA_TYPE: frames
  LABEL: [real]
  facial_lm_pretrained: pretrained/shape_predictor_81_face_landmarks.dat
  N_LANDMARKS: 81
  SAMPLING:
    ACTIVE: false
    NUMBERS: 8
  DEBUG: false
YAML
  local t0=$SECONDS
  ( cd "$REPO" && python package_utils/geo_landmarks_extraction.py --config "configs/r1_lm_$1.yaml" --extract_landmark --save_aligned ) > "$LOGS/landmarks_$1.log" 2>&1
  [ -f "$out" ] || { echo "ERROR: landmark file for $1 was not written, see $LOGS/landmarks_$1.log"; return 1; }
  echo "landmarks_$1: $((SECONDS - t0)) s" | tee -a "$LOGS/times.txt"
}
lm train &
L0=$!
lm val &
L1=$!
FAIL=0
wait $L0 || FAIL=1
wait $L1 || FAIL=1
[ "$FAIL" -eq 0 ] || { tail -n 5 "$LOGS"/landmarks_*.log; exit 1; }
python - "$REPO" "$WORK" <<'PY'
import json, sys
repo, work = sys.argv[1:3]
counts = {c['split']: c['frames'] for c in json.load(open(f'{work}/crop_counts.json')) if c['folder'] == 'original'}
res = {}
for split in ['train', 'val']:
    d = json.load(open(f'{repo}/processed_data/c23/{split}_FaceForensics_original_81.json'))['data']
    empty = sum(1 for it in d if not len(it['aligned_lms']))
    res[split] = {'items': len(d), 'no_landmarks': empty}
    print(split, 'items', len(d), '| without landmarks', empty, f'({100 * empty / len(d):.1f}%)', '| example', d[0]['image_path'])
    assert len(d) == counts[split], (split, len(d), counts[split])
json.dump(res, open(f'{work}/landmark_counts.json', 'w'), indent=1)
PY

echo "== times =="
cat "$LOGS/times.txt"
du -sh "$SAVE" 2>/dev/null || true
echo "PREPARATION OK. Send prepare_data.log (or everything printed above) back."
