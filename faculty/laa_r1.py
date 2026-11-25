#!/usr/bin/env python
"""Helper for the R1 runs of the visual branch (LAA-Net reproduction). Decision log: DECISIONS.md D031.

Sub-commands (all work on a clean copy of the official repo at commit a33446a):
  patch <repo> <control|fixed|test>   apply our code changes; every edit is checked, nothing is guessed
  config-train <repo> <task> ...      write configs/<task>.yaml for training from scratch
  config-test <repo> <tag> ...        write configs/test_<tag>.yaml for scripts/test.py
  probe <repo> <cfg> <control|fixed>  build the training set, draw one sample, check frame resampling
  unique-root <src_root> <dst_root>   test layout in which every fake video keeps its own name
  summarize <logs_dir> <out.json>     collect the result lines of test logs into one table

Changes made by `patch` (all are deviations from the official code and are logged):
  both runs  float64 -> float32 (4 files); a "last" checkpoint and one summary line per epoch
             (bookkeeping only: no random numbers are drawn, training is not altered)
  fixed run  SBIFaceForensic._reload_data() writes the attributes that __getitem__ reads, so a new
             set of frames per video is drawn every epoch, as config DATA_RELOAD: True intends
"""
import argparse
import glob
import json
import os
import re
import sys

FLOAT_FILES = {            # file -> number of 'torch.float64' occurrences at commit a33446a
    'scripts/train.py': 4,
    'lib/core_function.py': 16,
    'losses/losses.py': 2,
    'scripts/test.py': 8,
}

RELOAD_OLD = "        self.image_paths, self.labels, self.mask_paths, self.ot_props = self._load_data(self.split)\n"
RELOAD_NEW = "        self.image_paths_r, self.labels_r, self.mask_paths_r, self.ot_props_r = self._load_data(self.split)  # R1-fixed (D031)\n"

SUMMARY_ANCHOR = "            if ((acc_val.avg > max_val_acc) and (loss_val.avg < min_val_loss)):\n"
SUMMARY_NEW = ("            logger.info('R1_EPOCH {} train_loss {:.6f} train_acc {:.5f} val_loss {:.6f} val_acc {:.5f}'.format(\n"
               "                epoch, loss_avg.avg, float(acc_avg.avg), loss_val.avg, float(acc_val.avg)))  # R1 bookkeeping (D031)\n")
LAST_ANCHOR = "        lr_scheduler.step()\n"
LAST_NEW = ("        save_model(path=os.path.join(LOG_DIR, '{}_{}_model_last.pth'.format(cfg.MODEL.type, cfg.TASK)),\n"
            "                   epoch=epoch, model=model, optimizer=optimizer)  # R1 bookkeeping (D031)\n")


def _read(path):
    with open(path, encoding='utf-8', newline='') as f:
        return f.read()


def _write(path, text):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)


def _replace_once(text, old, new, where):
    crlf = chr(13) + chr(10)
    if crlf in text:                  # a Windows checkout of the repo; Linux has plain line feeds
        old, new = old.replace(chr(10), crlf), new.replace(chr(10), crlf)
    n = text.count(old)
    if n != 1:
        sys.exit(f'PATCH ERROR in {where}: expected exactly 1 match, found {n}. The file is not the expected version.')
    return text.replace(old, new)


def cmd_patch(a):
    repo, variant = a.repo, a.variant
    marker = os.path.join(repo, '.r1_patched')
    if os.path.exists(marker):
        sys.exit(f'{repo} is already patched ({_read(marker).strip()}); start from a clean copy.')
    for rel, want in FLOAT_FILES.items():
        p = os.path.join(repo, rel)
        s = _read(p)
        n = s.count('torch.float64')
        if n != want:
            sys.exit(f'PATCH ERROR in {rel}: expected {want} torch.float64, found {n}.')
        _write(p, s.replace('torch.float64', 'torch.float32'))
        print(f'{rel}: {n} x float64 -> float32')
    if variant in ('control', 'fixed'):
        p = os.path.join(repo, 'scripts/train.py')
        s = _read(p)
        s = _replace_once(s, SUMMARY_ANCHOR, SUMMARY_NEW + SUMMARY_ANCHOR, 'scripts/train.py (epoch summary)')
        s = _replace_once(s, LAST_ANCHOR, LAST_NEW + LAST_ANCHOR, 'scripts/train.py (last checkpoint)')
        _write(p, s)
        print('scripts/train.py: epoch summary line and last checkpoint added')
    if variant == 'fixed':
        p = os.path.join(repo, 'datasets/face_forensic_sbi.py')
        s = _replace_once(_read(p), RELOAD_OLD, RELOAD_NEW, 'datasets/face_forensic_sbi.py (_reload_data)')
        _write(p, s)
        print('datasets/face_forensic_sbi.py: _reload_data now updates the frames that are used')
    _write(marker, variant + '\n')


def _yaml():
    import yaml
    return yaml


def cmd_config_train(a):
    yaml = _yaml()
    cfg = yaml.safe_load(_read(os.path.join(a.repo, 'configs/efn4_fpn_sbi_adv.yaml')))
    cfg['TASK'] = a.task
    data = cfg['DATASET']['DATA']
    root = a.data_root.rstrip('/') + '/'
    for split, anno in (('TRAIN', a.train_anno), ('VAL', a.val_anno)):
        assert os.path.isfile(anno), anno
        data[split]['ROOT'] = root
        data[split]['ANNO_FILE'] = os.path.abspath(anno)
    # train.py also builds a test loader (never evaluated there); point it at the real test crops
    data['TEST'].update({'NAME': 'FF++', 'ROOT': root, 'FROM_FILE': False, 'FAKETYPE': ['original']})
    # DATASET.COMPRESSION stays 'c0': in this code it only selects the folder layout
    # (<split>/frames/<type>/<video>/<frame>.png), which is what images_crop.py writes for c23 too.
    assert cfg['DATASET']['COMPRESSION'] == 'c0'
    tr = cfg['TRAIN']
    tr.update({'epochs': a.epochs, 'begin_epoch': -1, 'warm_up': a.warm_up, 'resume': False, 'pretrained': ''})
    tr['loss']['hm_lmda'] = a.hm_lmda
    tr['loss']['cstency_lmda'] = a.cstency_lmda
    if a.num_workers is not None:
        cfg['DATASET']['NUM_WORKERS'] = a.num_workers
    if a.no_pretrained:                      # local smoke test only: no download of ImageNet weights
        cfg['MODEL']['INIT_WEIGHTS']['pretrained'] = False
    if a.batch_size is not None:             # local smoke test only
        tr['batch_size'] = a.batch_size
    if a.samples_train is not None:          # local smoke test only
        data['SAMPLES_PER_VIDEO']['TRAIN'] = a.samples_train
    out = os.path.join(a.repo, 'configs', a.task + '.yaml')
    _write(out, yaml.safe_dump(cfg, sort_keys=False))
    keep = {'TASK': cfg['TASK'], 'SEED': cfg['SEED'], 'DATA_RELOAD': cfg['DATA_RELOAD'],
            'epochs': tr['epochs'], 'warm_up': tr['warm_up'], 'batch_size': tr['batch_size'], 'lr': tr['lr'],
            'optimizer': tr['optimizer'], 'hm_lmda': tr['loss']['hm_lmda'], 'cstency_lmda': tr['loss']['cstency_lmda'],
            'workers': cfg['DATASET']['NUM_WORKERS'], 'frames_per_video': data['SAMPLES_PER_VIDEO']['TRAIN']}
    print('wrote', out, json.dumps(keep))


def cmd_config_test(a):
    yaml = _yaml()
    cfg = yaml.safe_load(_read(os.path.join(a.repo, 'configs/efn4_fpn_sbi_adv.yaml')))
    assert os.path.isfile(a.weights), a.weights
    cfg['DATASET']['TRAIN'] = False
    t = cfg['DATASET']['DATA']['TEST']
    t.update({'NAME': 'FF++', 'ROOT': a.root.rstrip('/') + '/', 'FROM_FILE': False, 'FAKETYPE': a.types.split(',')})
    cfg['TEST']['pretrained'] = os.path.abspath(a.weights)
    if a.num_workers is not None:
        cfg['DATASET']['NUM_WORKERS'] = a.num_workers
    out = os.path.join(a.repo, 'configs', 'test_' + a.tag + '.yaml')
    _write(out, yaml.safe_dump(cfg, sort_keys=False))
    print('wrote', out, '| types', t['FAKETYPE'], '| root', t['ROOT'])


def cmd_unique_root(a):
    """Official layout: Deepfakes/000_003, Face2Face/000_003 ... share a folder name and the test code
    pools them into one video (D020). Here each fake video is linked as <type>__<name>."""
    n = 0
    base = os.path.join(a.src, 'test', 'frames')
    for ft in sorted(os.listdir(base)):
        for v in sorted(os.listdir(os.path.join(base, ft))):
            dst_dir = os.path.join(a.dst, 'test', 'frames', ft)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, v if ft == 'original' else f'{ft}__{v}')
            if not os.path.islink(dst):
                os.symlink(os.path.realpath(os.path.join(base, ft, v)), dst)
            n += 1
    print('unique layout:', n, 'video folders under', a.dst)


def cmd_probe(a):
    """Runs inside the patched repo (cwd = repo). Fails loudly instead of hanging: the official loader
    hides every error and retries with another image for ever."""
    import signal
    import time
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import random
    import torch
    from configs.get_config import load_config
    from datasets import DATASETS, build_dataset
    from datasets.sbi.utils import gen_target, reorder_landmark
    from package_utils.image_utils import load_image
    cfg = load_config(a.cfg)
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    ds = build_dataset(cfg.DATASET, DATASETS, default_args=dict(split='train', config=cfg.DATASET))
    n_vid = len({p.replace('\\', '/').split('/')[-2] for p in ds.image_paths_r})
    empty = sum(1 for o in ds.ot_props_r if not len(o.get('aligned_lms', [])))
    print(f'training set: {len(ds)} frames from {n_vid} videos | drawn frames without landmarks: {empty}')
    assert len(ds) == n_vid * cfg.DATASET.DATA.SAMPLES_PER_VIDEO.TRAIN, 'sampler did not group frames by video'
    i = next(k for k, o in enumerate(ds.ot_props_r) if len(o.get('aligned_lms', [])))
    out = gen_target(load_image(ds.image_paths_r[i]), reorder_landmark(np.array(ds.ot_props_r[i]['aligned_lms'])), margin=[10, 10])
    print('pseudo-fake ok', [o.shape for o in out])
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(300)
    t = time.time()
    x = ds[i]
    print('one training sample ok', tuple(x[0].shape), x[0].dtype, f'{time.time() - t:.2f} s')
    assert len(x) == 8 and tuple(x[0].shape) == (3, 384, 384)
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(0)
    before = list(ds.image_paths_r)
    ds._reload_data()
    after = list(ds.image_paths_r)
    changed = sum(b != c for b, c in zip(before, after))
    print(f'after _reload_data: {changed} of {len(before)} frame slots changed')
    if a.variant == 'fixed':
        assert len(after) == len(before) and changed > len(before) // 2, 'R1-fixed: reload did not change the frames'
        assert len({p.replace('\\', '/').split('/')[-2] for p in after}) == n_vid
    else:
        assert changed == 0, 'R1-control: frames changed, but the official code keeps them'
    print('PROBE OK', a.variant)


RESULT = re.compile(r"Current ACC, AUC, AP, AR, mF1 for (\[.*?\]) --- .*? --\s+([\d.]+) -- ([\d.]+) -- ([\d.]+) -- ([\d.]+) -- ([\d.]+)")
# D022: released SBI weights, official environment, float64, Kaggle copy of FF++ c23 (AUC, AP)
R0 = {'all_asis': (94.93, 94.34), 'all_unique': (89.07, 97.07), 'Deepfakes': (95.77, 96.74),
      'Face2Face': (87.43, 88.27), 'FaceSwap': (92.44, 93.18), 'NeuralTextures': (80.59, 81.57)}


def cmd_summarize(a):
    rows = {}
    worst = 0.0
    for f in sorted(glob.glob(os.path.join(a.logs, 'test_*.log'))):
        tag = os.path.basename(f)[len('test_'):-len('.log')]
        m = None
        for m in RESULT.finditer(open(f, errors='replace').read()):
            pass
        if m is None:
            rows[tag] = None
            print(f'{tag:28s} NO RESULT LINE (see {f})')
            continue
        acc, auc, ap, ar, mf1 = (round(float(m.group(k)), 2) for k in range(2, 7))
        rows[tag] = {'types': m.group(1), 'acc': acc, 'auc': auc, 'ap': ap, 'ar': ar, 'mf1': mf1}
        line = f'{tag:28s} AUC {auc:6.2f}  AP {ap:6.2f}  ACC@0.5 {acc:6.2f}'
        key = tag.split('.')[-1]
        if a.compare_r0 and key in R0:
            line += f'   | R0 (D022) AUC {R0[key][0]:.2f}  AP {R0[key][1]:.2f}   diff AUC {auc - R0[key][0]:+.2f}'
            worst = max(worst, abs(auc - R0[key][0]))
        print(line)
    json.dump(rows, open(a.out, 'w'), indent=1)
    print('saved', a.out)
    if any(v is None for v in rows.values()) or not rows:
        sys.exit('some test jobs have no result')
    if a.compare_r0:
        if len(rows) != len(R0):
            sys.exit(f'expected {len(R0)} test jobs, found {len(rows)}')
        print(f'largest AUC difference from R0: {worst:.2f}')
        if a.max_diff is not None and worst > a.max_diff:
            sys.exit(f'PORT CHECK NOT PASSED: an AUC differs from R0 by more than {a.max_diff}')
        if a.max_diff is not None:
            print(f'PORT CHECK PASSED (limit {a.max_diff})')


STEP = re.compile(r'Epoch: \[(\d+)\]\[(\d+)/(\d+)\]\s+Time ([\d.]+)s \(([\d.]+)s\)\s+Speed [\d.]+ samples/s\s+Data ([\d.]+)s')


def cmd_timing(a):
    """Median seconds per step from a training log, and a projection to the full 100-epoch run
    (720 training steps and 140 validation steps per epoch, backbone frozen for the first 6)."""
    import statistics
    text = open(a.log, errors='replace').read()
    rows = [(int(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[6])) for m in STEP.finditer(text)]
    if not rows:
        print('no step lines found in', a.log)
        sys.exit(1)
    ntrain = max(r[2] for r in rows)
    med = lambda v: round(statistics.median(v), 3) if v else None
    out = {}
    for e in sorted({r[0] for r in rows}):
        tr = [r for r in rows if r[0] == e and r[2] == ntrain and r[1] > 0]
        va = [r for r in rows if r[0] == e and r[2] != ntrain and r[1] > 0]
        out[e] = (med([r[3] for r in tr]), med([r[4] for r in tr]), med([r[3] for r in va]))
        print(f'epoch {e}: train step {out[e][0]} s (waiting for data {out[e][1]} s) | val step {out[e][2]} s | {ntrain} steps per epoch')
    if 0 in out and 1 in out and all(out[0]) and out[1][0]:
        val = out[1][2] or out[0][2]
        frozen = (720 * out[0][0] + 140 * out[0][2]) / 3600
        unfrozen = (720 * out[1][0] + 140 * val) / 3600
        print(f'projection (if epoch 0 was frozen and epoch 1 unfrozen): {6 * frozen + 94 * unfrozen:.1f} h for 100 epochs')
    if 'out of memory' in text or 'Traceback' in text:
        print('the log contains an error')
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('patch'); p.add_argument('repo'); p.add_argument('variant', choices=['control', 'fixed', 'test'])
    p.set_defaults(fn=cmd_patch)
    p = sub.add_parser('config-train'); p.add_argument('repo'); p.add_argument('task')
    p.add_argument('--data-root', required=True); p.add_argument('--train-anno', required=True); p.add_argument('--val-anno', required=True)
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--warm-up', type=int, default=6)
    p.add_argument('--hm-lmda', type=float, default=10); p.add_argument('--cstency-lmda', type=float, default=100)
    p.add_argument('--num-workers', type=int); p.add_argument('--batch-size', type=int); p.add_argument('--samples-train', type=int)
    p.add_argument('--no-pretrained', action='store_true')
    p.set_defaults(fn=cmd_config_train)
    p = sub.add_parser('config-test'); p.add_argument('repo'); p.add_argument('tag')
    p.add_argument('--root', required=True); p.add_argument('--types', required=True); p.add_argument('--weights', required=True)
    p.add_argument('--num-workers', type=int)
    p.set_defaults(fn=cmd_config_test)
    p = sub.add_parser('unique-root'); p.add_argument('src'); p.add_argument('dst'); p.set_defaults(fn=cmd_unique_root)
    p = sub.add_parser('probe'); p.add_argument('cfg'); p.add_argument('variant', choices=['control', 'fixed'])
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser('summarize'); p.add_argument('logs'); p.add_argument('out'); p.add_argument('--compare-r0', action='store_true')
    p.add_argument('--max-diff', type=float)
    p.set_defaults(fn=cmd_summarize)
    p = sub.add_parser('timing'); p.add_argument('log'); p.set_defaults(fn=cmd_timing)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
