"""One-off edit (2026-11-25, D033): let the faculty scripts use a chosen GPU list. Run from F:/Visual/faculty."""


def edit(p, pairs):
    s = open(p, 'rb').read().decode()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:60], s.count(a))
        s = s.replace(a, b)
    open(p, 'wb').write(s.encode())


edit('common.sh', [
    ('''source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
''',
     '''# GPUs these scripts may use, e.g. GPUS=1 when GPU 0 is busy with other work. Default: both.
GPUS="${GPUS:-0 1}"
read -r -a GPU_LIST <<< "$GPUS"
[ "${#GPU_LIST[@]}" -ge 1 ] || { echo "ERROR: GPUS is empty"; exit 1; }
for g in "${GPU_LIST[@]}"; do
  nvidia-smi -i "$g" > /dev/null 2>&1 || { echo "ERROR: GPU $g (from GPUS=$GPUS) does not exist"; exit 1; }
done
G_A="${GPU_LIST[0]}"
G_B="${GPU_LIST[1]:-${GPU_LIST[0]}}"
echo "using GPU(s): $GPUS"

set +u                                                 # conda's own scripts trip over 'set -u'
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
'''),
    ('''# Six test jobs on the FF++ c23 test crops, three per GPU:''',
     '''# Six test jobs on the FF++ c23 test crops (three per GPU with two GPUs, else one after the other):'''),
    ('''  local g1=$([ "$(nvidia-smi -L | wc -l)" -ge 2 ] && echo 1 || echo 0)
  ( one 0 all_asis "$SAVE" "$ALL_TYPES" && one 0 Deepfakes "$SAVE" original,Deepfakes && one 0 FaceSwap "$SAVE" original,FaceSwap ) &
  local p0=$!
  ( one "$g1" all_unique "$uroot" "$ALL_TYPES" && one "$g1" Face2Face "$SAVE" original,Face2Face && one "$g1" NeuralTextures "$SAVE" original,NeuralTextures ) &
  local p1=$!
  local fail=0
  wait $p0 || fail=1
  wait $p1 || fail=1
  return $fail''',
     '''  local fail=0
  if [ "${#GPU_LIST[@]}" -ge 2 ]; then
    ( one "$G_A" all_asis "$SAVE" "$ALL_TYPES" && one "$G_A" Deepfakes "$SAVE" original,Deepfakes && one "$G_A" FaceSwap "$SAVE" original,FaceSwap ) &
    local p0=$!
    ( one "$G_B" all_unique "$uroot" "$ALL_TYPES" && one "$G_B" Face2Face "$SAVE" original,Face2Face && one "$G_B" NeuralTextures "$SAVE" original,NeuralTextures ) &
    local p1=$!
    wait $p0 || fail=1
    wait $p1 || fail=1
  else
    one "$G_A" all_asis "$SAVE" "$ALL_TYPES" || fail=1
    one "$G_A" all_unique "$uroot" "$ALL_TYPES" || fail=1
    for t in $TYPES; do one "$G_A" "$t" "$SAVE" "original,$t" || fail=1; done
  fi
  return $fail'''),
])

edit('01_prepare_data.sh', [
    ('''# Run:  bash 01_prepare_data.sh /path/to/FaceForensics++_C23 2>&1 | tee prepare_data.log''',
     '''# Run:  bash 01_prepare_data.sh /path/to/FaceForensics 2>&1 | tee prepare_data.log
# One GPU only (the other is busy):  GPUS=1 bash 01_prepare_data.sh /path/to/FaceForensics ...'''),
    ('''source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
''',
     '''GPUS="${GPUS:-0 1}"
read -r -a GPU_LIST <<< "$GPUS"
[ "${#GPU_LIST[@]}" -ge 1 ] || { echo "ERROR: GPUS is empty"; exit 1; }
for g in "${GPU_LIST[@]}"; do
  nvidia-smi -i "$g" > /dev/null 2>&1 || { echo "ERROR: GPU $g (from GPUS=$GPUS) does not exist"; exit 1; }
done
echo "using GPU(s): $GPUS"
set +u                                                 # conda's own scripts trip over 'set -u'
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
'''),
    ('''echo "== 5. face crops (GPU 0: train; GPU 1: val, then the five test folders) =="''',
     '''echo "== 5. face crops (two GPUs: train on one, val and test on the other; one GPU: one after the other) =="'''),
    ('''NGPU=$(nvidia-smi -L | wc -l)
G1=$([ "$NGPU" -ge 2 ] && echo 1 || echo 0)
crop 0 Original 128 train &
P0=$!
( crop "$G1" Original 32 val; crop "$G1" Original 32 test; for t in $TYPES; do crop "$G1" "$t" 32 test; done ) &
P1=$!
FAIL=0
wait $P0 || FAIL=1
wait $P1 || FAIL=1
''',
     '''G0="${GPU_LIST[0]}"
FAIL=0
if [ "${#GPU_LIST[@]}" -ge 2 ]; then
  G1="${GPU_LIST[1]}"
  crop "$G0" Original 128 train &
  P0=$!
  ( crop "$G1" Original 32 val; crop "$G1" Original 32 test; for t in $TYPES; do crop "$G1" "$t" 32 test; done ) &
  P1=$!
  wait $P0 || FAIL=1
  wait $P1 || FAIL=1
else
  # short jobs first, so a problem shows up early
  ( crop "$G0" Original 32 val; crop "$G0" Original 32 test; for t in $TYPES; do crop "$G0" "$t" 32 test; done; crop "$G0" Original 128 train ) || FAIL=1
fi
'''),
])

edit('00_setup_env.sh', [
    ('''# Run:  bash 00_setup_env.sh 2>&1 | tee setup_env.log''',
     '''# Run:  bash 00_setup_env.sh 2>&1 | tee setup_env.log
# While the GPUs are busy with other work:  GPUS=none bash 00_setup_env.sh ...   (skips the speed check)'''),
    ('''source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list''',
     '''set +u                                                 # conda's own scripts trip over 'set -u'
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list'''),
    ('''conda activate "$ENV_NAME"
python --version''',
     '''conda activate "$ENV_NAME"
set -u
python --version'''),
    ('''python - <<'PY'
import time, numpy, cv2, dlib, albumentations, imgaug, torch, torchvision''',
     '''export GPUS="${GPUS:-all}"
python - <<'PY'
import os, time, numpy, cv2, dlib, albumentations, imgaug, torch, torchvision'''),
    ('''print("GPUs:", torch.cuda.device_count(), "| built for:", torch.cuda.get_arch_list())
torch.from_numpy(numpy.zeros((2, 2), dtype=numpy.float32))
for i in range(torch.cuda.device_count()):
    dev = f"cuda:{i}"
''',
     '''archs = torch.cuda.get_arch_list()
print("GPUs:", torch.cuda.device_count(), "| built for:", archs)
assert "sm_120" in archs, "this PyTorch build does not support RTX 50-series cards"
torch.from_numpy(numpy.zeros((2, 2), dtype=numpy.float32))
want = os.environ.get("GPUS", "all")
ids = list(range(torch.cuda.device_count())) if want == "all" else [int(x) for x in want.split() if x.isdigit()]
if not ids:
    print("GPU speed check skipped (GPUS=%s)" % want)
for i in ids:
    dev = f"cuda:{i}"
'''),
    ('''    x = torch.randn(16, 3, 384, 384, device=dev)
    for dt in (torch.float32, torch.float64):
        net.to(dt); xx = x.to(dt)
        net(xx).mean().backward(); torch.cuda.synchronize(dev)
        t = time.time()
        for _ in range(10):
            net(xx).mean().backward()
        torch.cuda.synchronize(dev)
        print(dev, torch.cuda.get_device_name(i), str(dt), f"{(time.time() - t) / 10 * 1000:.1f} ms per pass")
''',
     '''    try:
        x = torch.randn(8, 3, 384, 384, device=dev)
        for dt in (torch.float32, torch.float64):
            net.to(dt); xx = x.to(dt)
            net(xx).mean().backward(); torch.cuda.synchronize(dev)
            t = time.time()
            for _ in range(10):
                net(xx).mean().backward()
            torch.cuda.synchronize(dev)
            print(dev, torch.cuda.get_device_name(i), str(dt), f"{(time.time() - t) / 10 * 1000:.1f} ms per pass")
    except RuntimeError as e:      # for example the card is full with someone else's job
        print(dev, "speed check not possible now:", str(e).splitlines()[0][:120])
    del net
    torch.cuda.empty_cache()
'''),
])
print('edits ok')
