import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# LAA-Net R0: authors' released SBI weights, official test code, FF++ c23 test split

Visual branch, Stage 1. Nothing is trained here.

**Before running**
1. Add input: dataset `xdxd003/ff-c23`.
2. Weights: nothing to do. The notebook downloads the authors' Dropbox folder (831 MB zip) and
   keeps the SBI weights file. If you prefer, attach a dataset holding
   `PoseEfficientNet_EFN_hm10_EFPN_..._8SBI_..._model_best.pth` and it is used instead.
3. Settings: Accelerator = **GPU T4 x2** (jobs are split over both GPUs), Internet = On.
4. Smoke mode (2 video pairs) worked on 2026-11-23. For the real test set `MODE = 'full'` and use
   Save Version -> Save & Run All (Commit), so it runs in the background (several hours).

Paper numbers to compare with: FF++ AUC 99.96 (Table 1, compression not stated) and
**89.30 on FF++ c23** (supplement Sec. 8.3).
""")

code("""
MODE = 'full'             # 'smoke' = 2 pairs of the test split, 'full' = whole test split (70 pairs)
PER_TYPE = True           # also report each fake type against the real videos
COMMIT = 'a33446ab94b3bc0d176a879d43e0700adc52c817'
FAKES = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

import os, glob, json, subprocess, shutil, sys
W = '/kaggle/working'

def sh(cmd, **kw):
    print('$', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash', **kw)

# locate the FF++ c23 videos and the weights among the attached inputs
cands = [d for d in glob.glob('/kaggle/input/**/original', recursive=True)
         if os.path.isdir(d) and len(glob.glob(d + '/*.mp4')) > 0]
assert len(cands) == 1, f'expected one folder named original with mp4 files, found {cands}'
FF = os.path.dirname(cands[0])
for name in ['original'] + FAKES:
    n = len(glob.glob(f'{FF}/{name}/*.mp4'))
    print(name, n)
    assert n == 1000, f'{name}: expected 1000 videos, found {n}'

wts = glob.glob('/kaggle/input/**/*8SBI*model_best.pth', recursive=True)
if not wts:
    # not attached as an input: fetch the authors' Dropbox folder (one zip, 870,989,302 bytes,
    # holds the SBI and the BI weights), keep only the SBI file
    os.makedirs(f'{W}/weights', exist_ok=True)
    wts = glob.glob(f'{W}/weights/*8SBI*model_best.pth')
    if not wts:
        url = ('https://www.dropbox.com/scl/fo/dzmldaytujdeuky69d5x1/AIJrH2mit1hxnl1qzavM3vk'
               '?rlkey=nzzliincrfwejw2yr0ovldru1&st=z8ds7il7&dl=1')
        sh(f'wget -q -O {W}/weights/laa.zip "{url}"')
        print('zip bytes:', os.path.getsize(f'{W}/weights/laa.zip'))
        sh(f'cd {W}/weights && unzip -o -j laa.zip "*8SBI*" && rm laa.zip')
        wts = glob.glob(f'{W}/weights/*8SBI*model_best.pth')
assert len(wts) == 1, f'expected one SBI weights file, found {wts}'
WEIGHTS = wts[0]
import hashlib
h = hashlib.sha256()
with open(WEIGHTS, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 24), b''):
        h.update(chunk)
print('weights sha256:', h.hexdigest())
print('FF root:', FF)
print('weights:', WEIGHTS, round(os.path.getsize(WEIGHTS) / 2**20, 2), 'MB')
""")

md("""
## Environment
The repo is pinned to PyTorch 1.8 / torchvision 0.9 / albumentations 1.1.0 and does not import on
current versions, so a separate Python 3.8 environment is built with the versions from the repo's
README and Dockerfile. Packages the repo imports but does not list (scikit-learn, matplotlib,
retinaface-pytorch) are installed unpinned. `retinaface-pytorch` downloads its ResNet-50 face
detector weights on first use.
""")

code("""
sh('pip install -q uv')
sh(f'uv venv --python 3.8 {W}/env')
PY = f'{W}/env/bin/python'
sh(f'uv pip install -q --python {PY} torch==1.8.0+cu111 torchvision==0.9.0+cu111 '
   '--extra-index-url https://download.pytorch.org/whl/cu111')
sh(f'uv pip install -q --python {PY} albumentations==1.1.0 python-box==7.1.1 imgaug==0.4.0 '
   'imutils==0.5.4 numpy==1.23.3 opencv-python-headless==4.5.1.48 pandas==1.3.5 Pillow==9.3.0 '
   'scikit-image==0.19.3 scipy==1.9.3 simplejson tensorboardX==2.5.1 tqdm PyYAML '
   'scikit-learn matplotlib retinaface-pytorch')
sh(f'{PY} -c "import torch, torchvision, albumentations, numpy; '
   'print(torch.__version__, torchvision.__version__, albumentations.__version__, numpy.__version__, '
   'torch.cuda.is_available(), torch.cuda.get_device_name(0))"')
sh(f'uv pip freeze --python {PY} > {W}/env_freeze.txt')
""")

md("""
## Official code at the commit we read
""")

code("""
REPO = f'{W}/LAA-Net'
if not os.path.isdir(REPO):
    sh(f'git clone -q https://github.com/10Ring/LAA-Net {REPO}')
sh(f'git -C {REPO} checkout -q {COMMIT} && git -C {REPO} log -1 --format="%H %ad"')
""")

md("""
## Data layout expected by `package_utils/images_crop.py`
The script has fixed paths. The videos are linked into the folder tree it expects, and only two
path constants in the script are changed (`ROOT`, `SAVE_DIR`). The official FF++ test split file is
fetched from the FaceForensics repository. In smoke mode the first 2 pairs are written as `val.json`
and cropped with `-t val`, so the script itself needs no other change.
""")

code("""
ROOT = f'{W}/datasets_df'
OD = f'{ROOT}/FaceForensics++/original_download'
SAVE = f'{ROOT}/FaceForensics++/c23'
os.makedirs(OD, exist_ok=True)

def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.islink(dst):
        os.symlink(src, dst)

link(f'{FF}/original', f'{OD}/original_sequences/youtube/c23/videos')
for ft in FAKES:
    link(f'{FF}/{ft}', f'{OD}/manipulated_sequences/{ft}/c23/videos')

if not os.path.isfile(f'{OD}/test.json'):
    sh(f'wget -q -O {OD}/test.json '
       'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json')
pairs = json.load(open(f'{OD}/test.json'))
assert len(pairs) == 70 and all(len(p) == 2 for p in pairs), len(pairs)
json.dump(pairs[:2], open(f'{OD}/val.json', 'w'))
shutil.copy(f'{OD}/test.json', f'{W}/split_test.json')     # keep the split with the results

p = f'{REPO}/package_utils/images_crop.py'
s = open(p).read()
a = "ROOT = '/data/deepfake_cluster/datasets_df'"
b = "SAVE_DIR = f'{ROOT}/FaceForensics++/c0'"
if a in s:
    assert b in s
    s = s.replace(a, f"ROOT = '{ROOT}'").replace(b, "SAVE_DIR = f'{ROOT}/FaceForensics++/c23'")
    open(p, 'w').write(s)
sh(f'git -C {REPO} diff --stat')
""")

md("""
## Face crops (official script: RetinaFace, 32 evenly spaced frames per video, 256x256 PNG)
""")

code("""
import time, queue, concurrent.futures as cf
NGPU = len(subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True).stdout.strip().splitlines())
print('GPUs:', NGPU)
gpuq = queue.Queue()
for g in range(NGPU):
    gpuq.put(g)
TIMES = {}

def on_gpu(job):
    # job = (name, function gpu -> shell command); one job per GPU at a time
    name, make = job
    g = gpuq.get()
    t0 = time.time()
    try:
        sh(make(g))
    finally:
        gpuq.put(g)
    TIMES[name] = round((time.time() - t0) / 60, 1)
    print(f'[{name}] done in {TIMES[name]} min on GPU {g}', flush=True)

def parallel(jobs):
    with cf.ThreadPoolExecutor(NGPU) as ex:
        list(ex.map(on_gpu, jobs))
    json.dump(TIMES, open(f'{W}/times_{MODE}.json', 'w'))

TASK = 'val' if MODE == 'smoke' else 'test'
jobs = []
for ds in ['Original'] + FAKES:
    done = f'{W}/crop_{TASK}_{ds}.done'
    if os.path.isfile(done):
        continue
    jobs.append((f'crop {ds}', lambda g, ds=ds, done=done:
                 f'cd {REPO} && CUDA_VISIBLE_DEVICES={g} {PY} package_utils/images_crop.py '
                 f'-d {ds} -c c23 -n 32 -t {TASK} > {W}/crop_{TASK}_{ds}.log 2>&1 && touch {done}'))
parallel(jobs)

# The crop script writes real videos to frames/Original, but the test loader labels a folder as
# real only if it is called 'original'. Rename so the labels are right.
fr = f'{SAVE}/{TASK}/frames'
if os.path.isdir(f'{fr}/Original'):
    os.rename(f'{fr}/Original', f'{fr}/original')

import collections
rows = []
for ft in ['original'] + FAKES:
    vids = sorted(os.listdir(f'{fr}/{ft}'))
    counts = [len(glob.glob(f'{fr}/{ft}/{v}/*.png')) for v in vids]
    rows.append((ft, len(vids), min(counts), sum(counts)))
    print(ft, 'videos', len(vids), 'min frames', min(counts), 'total frames', sum(counts),
          'videos with fewer than 32:', sum(c < 32 for c in counts))
json.dump(rows, open(f'{W}/crop_counts_{TASK}.json', 'w'))
""")

md("""
## Test with the official script and the released SBI config
Changes to `configs/efn4_fpn_sbi_adv.yaml`, all of them paths or switches the README asks for:
test dataset name and root, the list of folders to test, the weights path, `DATASET.TRAIN: False`.
Everything else is as released (flip test on, 32 frames per video, video-level scores, seed 317).
`COMPRESSION: c0` is left as is: in this code it only selects the folder layout.
""")

code("""
import yaml
# the test loader always reads <root>/test/frames/<type>; in smoke mode point 'test' at the val crops
if MODE == 'smoke':
    TROOT = f'{W}/smoke_root'
    os.makedirs(TROOT, exist_ok=True)
    if not os.path.islink(f'{TROOT}/test'):
        os.symlink(f'{SAVE}/val', f'{TROOT}/test')
else:
    TROOT = SAVE

# Second layout with unique video names. In the official layout Deepfakes/000_003,
# Face2Face/000_003, ... share the folder name 000_003, and the test code identifies a video by
# that name only, so the four fake versions of a pair are pooled into ONE video (seen in the smoke
# run: 640 images loaded, 256 kept = 8 "videos" instead of 20). Here each fake video folder is
# linked under <type>__<name>, so every video is scored separately. No code is changed.
UROOT = f'{W}/unique_root'
for ft in ['original'] + FAKES:
    src = f'{TROOT}/test/frames/{ft}'
    for v in os.listdir(src):
        link(os.path.realpath(f'{src}/{v}'), f'{UROOT}/test/frames/{ft}/{v if ft == "original" else ft + "__" + v}')

def job(tag, types, root):
    cfg = yaml.safe_load(open(f'{REPO}/configs/efn4_fpn_sbi_adv.yaml'))
    cfg['DATASET']['TRAIN'] = False
    t = cfg['DATASET']['DATA']['TEST']
    t['NAME'] = 'FF++'; t['ROOT'] = root + '/'; t['FROM_FILE'] = False; t['FAKETYPE'] = types
    cfg['TEST']['pretrained'] = WEIGHTS
    yaml.safe_dump(cfg, open(f'{REPO}/configs/r0_{tag}.yaml', 'w'), sort_keys=False)
    log = f'{W}/r0_{MODE}_{tag}.log'
    return (f'test {tag}', lambda g:
            f'set -o pipefail; cd {REPO} && CUDA_VISIBLE_DEVICES={g} {PY} scripts/test.py '
            f'--cfg configs/r0_{tag}.yaml 2>&1 | grep -v "Inferencing time" > {log}; '
            f'rc=$?; tail -n 4 {log}; exit $rc')

def finished(tag):
    f = f'{W}/r0_{MODE}_{tag}.log'
    return os.path.isfile(f) and 'Current ACC' in open(f).read()

jobs = [('all_asis', ['original'] + FAKES, TROOT),          # official layout: fake types pooled per pair
        ('all_unique', ['original'] + FAKES, UROOT)]        # every video scored separately
if PER_TYPE:
    jobs += [(ft, ['original', ft], TROOT) for ft in FAKES]
parallel([job(*j) for j in jobs if not finished(j[0])])
print(TIMES)
""")

md("""
## Results
The line `Current ACC, AUC, AP, AR, mF1 ...` in each log is the result. Download
`r0_*.log`, `crop_*.log`, `crop_counts_*.json`, `times_*.json`, `env_freeze.txt`, `split_test.json`
from the Output tab. `all_asis` = official layout (the four fake versions of a pair pooled into one
video); `all_unique` = every video scored separately; the others are one fake type against the reals.
""")

code("""
for f in sorted(glob.glob(f'{W}/r0_{MODE}_*.log')):
    for line in open(f):
        if 'Current ACC' in line:
            print(os.path.basename(f), '|', ' '.join(line.split()))
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("ok", len(cells))
