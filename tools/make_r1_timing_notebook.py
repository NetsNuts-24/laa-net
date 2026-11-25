import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# LAA-Net R1 timing test: how long would training take on a Kaggle T4?

Visual branch, Stage 1. **This is not a training run and gives no result numbers.** It runs the
official training script for 2 short epochs on a handful of videos and measures seconds per step
and GPU memory, so we can decide the R1 settings (open question O9).

**Before running**
1. Add input: dataset `xdxd003/ff-c23`.
2. Settings: Accelerator = **GPU T4 x2**, Internet = On.
3. Run all cells (interactive run is fine; expected about 1 hour, hard stop per training job 75 min).

**What it downloads** (approved by the user before the first run): dlib 81-landmark file
(19,743,860 bytes, github.com/codeniko), EfficientNet-B4 AdvProp weights (78,000,637 bytes,
github.com/lukemelas, fetched by the official code), RetinaFace detector weights (as in R0), the
official FF++ `train.json` / `val.json` split lists.

**Two jobs are timed**
- `f64`: the official code as it is (64-bit precision, SAM, 8 real + 8 pseudo-fake images per step).
- `f32`: a copy of the repo where `torch.float64` is replaced by `torch.float32` in three files.
  Timing only; using it for a real run would be a logged deviation.
""")

code("""
N_TRAIN_PAIRS = 8         # first 8 pairs of the official train split = 16 real videos
N_VAL_PAIRS = 4           # first 4 pairs of the official val split   =  8 real videos
TRAIN_SAMPLES = 24        # frames drawn per train video (official: 8); 16 x 24 / 8 = 48 steps per epoch
VARIANTS = ['f64', 'f32']
JOB_TIMEOUT_MIN = 75
COMMIT = 'a33446ab94b3bc0d176a879d43e0700adc52c817'
DLIB_SHA = '8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869'   # user's local copy (D021)

import os, glob, json, subprocess, shutil, sys, time, re, hashlib, threading, statistics
W = '/kaggle/working'

def sh(cmd, check=True, **kw):
    print('$', cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, executable='/bin/bash', **kw).returncode

cands = [d for d in glob.glob('/kaggle/input/**/original', recursive=True)
         if os.path.isdir(d) and len(glob.glob(d + '/*.mp4')) > 0]
assert len(cands) == 1, f'expected one folder named original with mp4 files, found {cands}'
FF = os.path.dirname(cands[0])
n = len(glob.glob(f'{FF}/original/*.mp4'))
print('FF root:', FF, '| original videos:', n)
assert n == 1000
NGPU = len(subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True).stdout.strip().splitlines())
print('GPUs:', NGPU, '| CPU cores:', os.cpu_count())
assert NGPU >= 1
TIMES = {}
""")

md("""
## Environment (same pinned versions as R0, plus dlib for landmarks)
The repo does not list dlib. A prebuilt wheel (`dlib-bin`) is used; if that fails it is compiled.
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
if sh(f'uv pip install -q --python {PY} dlib-bin==19.24.6', check=False) != 0:
    sh(f'uv pip install -q --python {PY} cmake')
    sh(f'PATH={W}/env/bin:$PATH uv pip install -q --python {PY} dlib')
sh(f'{PY} -c "import torch, dlib, albumentations, numpy; '
   'print(torch.__version__, dlib.__version__, albumentations.__version__, numpy.__version__, '
   'torch.cuda.is_available(), torch.cuda.get_device_name(0))"')
sh(f'uv pip freeze --python {PY} > {W}/env_freeze_timing.txt')

REPO = f'{W}/LAA-Net'
if not os.path.isdir(REPO):
    sh(f'git clone -q https://github.com/10Ring/LAA-Net {REPO}')
sh(f'git -C {REPO} checkout -q {COMMIT} && git -C {REPO} log -1 --format="%H %ad"')

LM = f'{REPO}/pretrained/shape_predictor_81_face_landmarks.dat'
os.makedirs(os.path.dirname(LM), exist_ok=True)
if not os.path.isfile(LM):
    sh(f'wget -q -O {LM} https://github.com/codeniko/shape_predictor_81_face_landmarks/raw/master/'
       'shape_predictor_81_face_landmarks.dat')
got = hashlib.sha256(open(LM, 'rb').read()).hexdigest()
print('dlib 81 file:', os.path.getsize(LM), 'bytes, sha256', got)
assert got == DLIB_SHA, 'landmark file differs from the copy checked in D021'
""")

md("""
## Data: a small part of the official train and val splits
The crop script is the official one with the same two path constants changed as in R0. Train videos
get 128 frames and val videos 32, as the README says.
""")

code("""
ROOT = f'{W}/datasets_df'
OD = f'{ROOT}/FaceForensics++/original_download'
SAVE = f'{ROOT}/FaceForensics++/c23'
os.makedirs(OD, exist_ok=True)
dst = f'{OD}/original_sequences/youtube/c23/videos'
os.makedirs(os.path.dirname(dst), exist_ok=True)
if not os.path.islink(dst):
    os.symlink(f'{FF}/original', dst)

sizes = {'train': (360, N_TRAIN_PAIRS), 'val': (70, N_VAL_PAIRS)}
for split, (full, keep) in sizes.items():
    f = f'{W}/official_{split}.json'
    if not os.path.isfile(f):
        sh(f'wget -q -O {f} https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{split}.json')
    pairs = json.load(open(f))
    assert len(pairs) == full, (split, len(pairs))
    json.dump(pairs[:keep], open(f'{OD}/{split}.json', 'w'))          # what the crop script reads
    json.dump(pairs[:keep], open(f'{W}/timing_subset_{split}.json', 'w'))
    print(split, 'pairs used:', pairs[:keep])

p = f'{REPO}/package_utils/images_crop.py'
s = open(p).read()
a = "ROOT = '/data/deepfake_cluster/datasets_df'"
b = "SAVE_DIR = f'{ROOT}/FaceForensics++/c0'"
if a in s:
    assert b in s
    s = s.replace(a, f"ROOT = '{ROOT}'").replace(b, "SAVE_DIR = f'{ROOT}/FaceForensics++/c23'")
    open(p, 'w').write(s)
sh(f'git -C {REPO} diff --stat')

def timed(name, cmd):
    t0 = time.time()
    sh(cmd)
    TIMES[name] = round(time.time() - t0, 1)
    print(f'[{name}] {TIMES[name]} s', flush=True)

def crop(split, nframes, gpu):
    if not os.path.isfile(f'{W}/crop_{split}.done'):
        timed(f'crop {split} (s)', f'cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} {PY} package_utils/images_crop.py '
              f'-d Original -c c23 -n {nframes} -t {split} > {W}/timing_crop_{split}.log 2>&1 && touch {W}/crop_{split}.done')

# one after the other on one GPU, so the crop times are not disturbed by each other
crop('train', 128, 0)
crop('val', 32, 0)

COUNTS = {}
for split in ['train', 'val']:
    fr = f'{SAVE}/{split}/frames'
    if os.path.isdir(f'{fr}/Original'):
        os.rename(f'{fr}/Original', f'{fr}/original')      # README tree and the loaders use 'original'
    vids = sorted(os.listdir(f'{fr}/original'))
    c = [len(glob.glob(f'{fr}/original/{v}/*.png')) for v in vids]
    COUNTS[split] = {'videos': len(vids), 'frames': sum(c), 'min_frames': min(c)}
    print(split, COUNTS[split])
""")

md("""
## Landmarks (official script, dlib 81 points)
Config files are written here because the released `data_preprocessing_c23.yaml` does not match the
crop script's output (it says `images` / `jpg` and has no `COMPRESSION` key). No code is changed.
The script swallows a failed save, so the output file is checked afterwards.
""")

code("""
import yaml
os.makedirs(f'{REPO}/processed_data', exist_ok=True)
ANNO = {}
for split in ['train', 'val']:
    cfg = {'PREPROCESSING': {
        'DATASET': 'FaceForensics', 'COMPRESSION': 'c23', 'SPLIT': split, 'ROOT': SAVE + '/',
        'FAKETYPE': ['original'], 'IMAGE_SUFFIX': 'png', 'DATA_TYPE': 'frames', 'LABEL': ['real'],
        'facial_lm_pretrained': 'pretrained/shape_predictor_81_face_landmarks.dat', 'N_LANDMARKS': 81,
        'SAMPLING': {'ACTIVE': False, 'NUMBERS': 8}, 'DEBUG': False}}
    yaml.safe_dump(cfg, open(f'{REPO}/configs/r1t_lm_{split}.yaml', 'w'), sort_keys=False)
    ANNO[split] = f'processed_data/c23/{split}_FaceForensics_original_81.json'
    if not os.path.isfile(f'{REPO}/{ANNO[split]}'):
        timed(f'landmarks {split} (s)', f'cd {REPO} && {PY} package_utils/geo_landmarks_extraction.py '
              f'--config configs/r1t_lm_{split}.yaml --extract_landmark --save_aligned '
              f'> {W}/timing_landmarks_{split}.log 2>&1')
    assert os.path.isfile(f'{REPO}/{ANNO[split]}'), f'landmark file missing, see timing_landmarks_{split}.log'
    d = json.load(open(f'{REPO}/{ANNO[split]}'))['data']
    empty = sum(1 for it in d if not len(it['aligned_lms']))
    COUNTS[split]['landmark_items'] = len(d)
    COUNTS[split]['no_landmarks'] = empty
    print(split, 'items', len(d), '| without landmarks', empty, '| example path', d[0]['image_path'])
    assert len(d) == COUNTS[split]['frames']
""")

md("""
## Training configs
Starting from the released `configs/efn4_fpn_sbi_adv.yaml`. Changed: data paths and annotation
files; `begin_epoch: -1`, `resume: False`, no checkpoint (train from scratch; the released file is a
mid-training snapshot); and three timing-only values: `epochs: 2`, `warm_up: 1` (epoch 0 has the
backbone frozen, epoch 1 does not, so both speeds are measured), `SAMPLES_PER_VIDEO.TRAIN: 24`.
Everything else is as released (batch 8 + 8, SAM, 7 loader workers, loss weights).
""")

code("""
# train.py also builds a test loader; point it at the val crops (it is never evaluated)
TROOT = f'{W}/timing_test_root'
os.makedirs(TROOT, exist_ok=True)
if not os.path.islink(f'{TROOT}/test'):
    os.symlink(f'{SAVE}/val', f'{TROOT}/test')

def make_cfg(repo, tag):
    cfg = yaml.safe_load(open(f'{repo}/configs/efn4_fpn_sbi_adv.yaml'))
    cfg['TASK'] = f'r1timing_{tag}'
    D = cfg['DATASET']['DATA']
    D['SAMPLES_PER_VIDEO']['TRAIN'] = TRAIN_SAMPLES
    for split in ['train', 'val']:
        D[split.upper()]['ROOT'] = SAVE + '/'
        D[split.upper()]['ANNO_FILE'] = ANNO[split]
    D['TEST'].update({'NAME': 'FF++', 'ROOT': TROOT + '/', 'FROM_FILE': False, 'FAKETYPE': ['original']})
    T = cfg['TRAIN']
    T.update({'epochs': 2, 'begin_epoch': -1, 'warm_up': 1, 'resume': False, 'pretrained': ''})
    yaml.safe_dump(cfg, open(f'{repo}/configs/r1t_{tag}.yaml', 'w'), sort_keys=False)

REPOS = {'f64': REPO, 'f32': f'{W}/LAA-Net-f32'}
if 'f32' in VARIANTS and not os.path.isdir(REPOS['f32']):
    sh(f'cp -r {REPO} {REPOS["f32"]}')
    sh(f'cd {REPOS["f32"]} && sed -i "s/torch.float64/torch.float32/g" scripts/train.py lib/core_function.py losses/losses.py '
       '&& git diff --stat')
for tag in VARIANTS:
    make_cfg(REPOS[tag], tag)

# Check that pseudo-fake generation works before training: the loader hides every error and
# retries with another image for ever, so a broken setup would hang instead of failing.
probe = '''
import sys, os, json, signal, time
sys.path.insert(0, os.getcwd())
import numpy as np
from configs.get_config import load_config
from datasets import *
from datasets.sbi.utils import gen_target, reorder_landmark
from package_utils.image_utils import load_image
cfg = load_config(sys.argv[1])
d = json.load(open(cfg.DATASET.DATA.TRAIN.ANNO_FILE))['data']
it = next(x for x in d if len(x['aligned_lms']))
img = load_image(it['image_path'])
out = gen_target(img, reorder_landmark(np.array(it['aligned_lms'])), margin=[10, 10])
print('pseudo-fake ok', [o.shape for o in out])
signal.alarm(300)
ds = build_dataset(cfg.DATASET, DATASETS, default_args=dict(split='train', config=cfg.DATASET))
t = time.time()
x = ds[0]
print('one training sample ok', tuple(x[0].shape), x[0].dtype, round(time.time() - t, 2), 's', '| dataset length', len(ds))
'''
open(f'{W}/probe_sbi.py', 'w').write(probe)
sh(f'cd {REPO} && {PY} {W}/probe_sbi.py configs/r1t_f64.yaml')
""")

md("""
## Timed training jobs (one at a time, on GPU 0)
""")

code("""
PEAK = {}
RC = {}

def watch_mem(tag, stop):
    peak = 0
    while not stop.is_set():
        out = subprocess.run('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0',
                             shell=True, capture_output=True, text=True).stdout.strip()
        if out.isdigit():
            peak = max(peak, int(out))
        stop.wait(2)
    PEAK[tag] = peak

for tag in VARIANTS:
    log = f'{W}/timing_train_{tag}.log'
    stop = threading.Event()
    th = threading.Thread(target=watch_mem, args=(tag, stop))
    th.start()
    t0 = time.time()
    RC[tag] = sh(f'cd {REPOS[tag]} && CUDA_VISIBLE_DEVICES=0 timeout {JOB_TIMEOUT_MIN}m {PY} scripts/train.py '
                 f'--cfg configs/r1t_{tag}.yaml > {log} 2>&1', check=False)
    stop.set(); th.join()
    TIMES[f'train job {tag} (s)'] = round(time.time() - t0, 1)
    print(tag, 'exit code', RC[tag], '(124 = stopped by the time limit)', '| peak GPU memory MiB', PEAK[tag],
          '| wall', TIMES[f'train job {tag} (s)'], 's', flush=True)
    sh(f"tr '\\\\r' '\\\\n' < {log} | grep -v 'it/s\\\\|s/it' | tail -n 6", check=False)
    sh(f'rm -f {REPOS[tag]}/logs/*/*.pth', check=False)       # the saved checkpoint is not needed
""")

md("""
## Summary and projection to the full run
Full official run: 720 train videos x 8 frames / 8 = 720 steps per epoch, 140 val videos x 8 / 8 =
140 val steps per epoch, 100 epochs, backbone frozen for the first 6.
""")

code("""
PAT = re.compile(r'Epoch: \\[(\\d+)\\]\\[(\\d+)/(\\d+)\\]\\s+Time ([\\d.]+)s \\(([\\d.]+)s\\)\\s+Speed [\\d.]+ samples/s\\s+Data ([\\d.]+)s')

def parse(log):
    rows = []
    for line in re.split(r'[\\r\\n]+', open(log, errors='replace').read()):
        m = PAT.search(line)
        if m:
            e, i, n = int(m[1]), int(m[2]), int(m[3])
            rows.append((e, i, n, float(m[4]), float(m[6])))
    return rows

def med(v):
    return round(statistics.median(v), 3) if v else None

SUMMARY = {'counts': COUNTS, 'times': TIMES, 'peak_gpu_mem_mib': PEAK, 'exit_codes': RC, 'variants': {}}
for tag in VARIANTS:
    log = f'{W}/timing_train_{tag}.log'
    text = open(log, errors='replace').read()
    rows = parse(log)
    ntrain = max([r[2] for r in rows], default=0)
    out = {'out_of_memory': 'out of memory' in text, 'train_steps_per_epoch_here': ntrain}
    for e, name in [(0, 'frozen'), (1, 'unfrozen')]:
        tr = [r for r in rows if r[0] == e and r[2] == ntrain and r[1] > 0]
        va = [r for r in rows if r[0] == e and r[2] != ntrain and r[1] > 0]
        out[name] = {'train_step_s': med([r[3] for r in tr]), 'of_which_data_wait_s': med([r[4] for r in tr]),
                     'val_step_s': med([r[3] for r in va]), 'train_lines': len(tr), 'val_lines': len(va)}
    f, u = out['frozen'], out['unfrozen']
    if f['train_step_s'] and u['train_step_s'] and f['val_step_s']:
        vs = u['val_step_s'] or f['val_step_s']
        ep_f = (720 * f['train_step_s'] + 140 * f['val_step_s']) / 3600
        ep_u = (720 * u['train_step_s'] + 140 * vs) / 3600
        out['projection'] = {'hours_per_frozen_epoch': round(ep_f, 2), 'hours_per_unfrozen_epoch': round(ep_u, 2),
                             'hours_100_epochs': round(6 * ep_f + 94 * ep_u, 1)}
    SUMMARY['variants'][tag] = out

tv, vv = COUNTS['train']['videos'], COUNTS['val']['videos']
prep = {}
if 'crop train (s)' in TIMES:
    prep['crop_s_per_train_video_128f'] = round(TIMES['crop train (s)'] / tv, 1)
    prep['crop_hours_720_train_videos_one_gpu'] = round(TIMES['crop train (s)'] / tv * 720 / 3600, 1)
if 'crop val (s)' in TIMES:
    prep['crop_hours_140_val_videos_one_gpu'] = round(TIMES['crop val (s)'] / vv * 140 / 3600, 2)
if 'landmarks train (s)' in TIMES:
    per = TIMES['landmarks train (s)'] / COUNTS['train']['frames']
    prep['landmark_s_per_image'] = round(per, 3)
    prep['landmark_hours_92160_train_images_one_core'] = round(per * 92160 / 3600, 1)
SUMMARY['preparation'] = prep
json.dump(SUMMARY, open(f'{W}/timing_summary.json', 'w'), indent=1)
print(json.dumps(SUMMARY, indent=1))
""")

md("""
Send back the printed summary (or `timing_summary.json`). If a job failed, also send
`timing_train_f64.log` / `timing_train_f32.log` from the Output tab.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("ok", len(cells))
