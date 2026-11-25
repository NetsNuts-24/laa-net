# Visual branch of a multimodal deepfake detector

## What this project is

We are building a multimodal deepfake detection model, one branch at a time. This folder covers
**only the visual (spatial / frame-level) part**. The project map shared by all branches is
`F:\Multimodal\PROJECT.md`.

- Reference paper: **LAA-Net** (Nguyen et al., CVPR 2024), "Localized Artifact Attention Network for
  Quality-Agnostic and Generalizable Deepfake Detection". PDF in this folder:
  `Nguyen_LAA-Net_..._CVPR_2024_paper.pdf` (main paper only, 11 pages; the supplementary is not here).
- Official code: https://github.com/10Ring/LAA-Net (commit `a33446a`, dated 2025-11-26; cloned to
  `LAA-Net/`). Licence: "snt academic license" (not read in detail yet).
- Authors released trained weights for both variants (Dropbox link in the repo README). Not downloaded.

Other branches: temporal is in `F:\Ti2net` (TI2Net, WACV 2023) and is handled by another session.
**Never edit anything in `F:\Ti2net`**; read it only for formats and shared findings.

## Working order (agreed, do not skip ahead)

1. **Stage 1 — Reproduce the reference paper.** ← current stage (paper and code read; no run yet)
2. Stage 2 — Diagnose where it fails (by manipulation type, compression, dataset shift).
3. Stage 3 — Beat it with simple upgrades.
4. Stage 4 — Only then design the new paper.

Do not propose or build new-paper ideas while Stage 1 is open. Do not pick the reference paper or
run a literature search for one; the user names it.

## Rules for working here

- **Every decision goes in `DECISIONS.md`** at the moment it is made: what, why, alternatives rejected.
  Never change a decided item silently; add a new entry that supersedes the old one.
- Every deviation from the paper or from the official code is a decision and must be logged.
- Report numbers as "paper", "official code as-is" or "our run"; never mix them.
- Fix random seeds and save the train/test split to disk for every run.
- Ask before downloading datasets or model weights (state source and size).
- Ask the user before making choices for them; explain in plain words.

On 2026-11-23 the user chose to brainstorm upgrade ideas before the first run (D008). Ideas are
discussed and logged now (`DECISIONS.md` D010: A, B, C, D, G) but none is built or committed to
until we have our own LAA-Net numbers. A short novelty check (D012) found A and C already published
(Hopf–Timofte 2025; FakeSTormer, ICCV 2025, by the LAA-Net authors; LAA-X 2026), B crowded, G mostly
covered by AVPF and SAVe (2026), and no direct match for D (heatmap as the bridge to the
temporal/identity branch). D is the lead candidate, not yet confirmed novel.

## Paper targets (Table 1; trained on FF++ real videos only; video-level AUC % / AP %)

| Variant | FF++ (AUC) | CDF2 | DFW | DFD | DFDC |
|---|---|---|---|---|---|
| LAA-Net w/ SBI (main) | 99.96 | 95.40 / 97.64 | 80.03 / 81.08 | 98.43 / 99.40 | 86.94 / 97.70 |
| LAA-Net w/ BI | 99.95 | 86.28 / 91.93 | 57.13 / 56.89 | 99.51 / 99.80 | 69.69 / 93.67 |

Other paper numbers worth keeping:
- **FF++ c23 (supplement Sec. 8.3): AUC 89.30.** This is the number to compare with when we test on
  the Kaggle c23 copy. The supplement is not in this folder; it was read from arXiv 2401.13856v2.
- Robustness on FF++ (Table 2, AUC): saturation 99.96, contrast 99.96, block 99.96, **Gaussian noise
  53.9**, blur 98.22, pixelation 99.80. The paper itself calls the method "extremely sensitive" to noise.
- Ablation (Table 3, CDF2 / DFD / DFDC / DFW AUC): nothing 74.54 / 92.24 / 70.81 / 59.81;
  heatmap + E-FPN without consistency 95.56 / 98.54 / 82.21 / 74.98; full 95.40 / 98.43 / 86.94 / 80.03.
  So the consistency branch does not help on CDF2 or DFD; it helps on DFDC and DFW.
- λ sensitivity (Table 5): λ1=10, λ2=100 is the paper's best; λ1=λ2=100 gives CDF2 93.72.

## What the method does

- Trains on **real faces only**. Fakes are made on the fly by blending (SBI = a face blended with an
  augmented copy of itself; BI = blended with another person's face with similar landmarks).
- Because we made the fake, we know the blending mask M. Blending boundary B = 4·M·(1−M); the
  "vulnerable points" are the pixels where B is largest (both images contribute equally).
- Network: EfficientNet-B4 encoder + **E-FPN** decoder. E-FPN passes low-level features down to the
  output but multiplies them by (1 − sigmoid(upsampled deeper feature)) before concatenating, to
  avoid repeating what the deeper feature already says.
- Three heads on the 96×96 decoder output (input 384×384): (1) real/fake classification,
  (2) **heatmap**: Gaussians centred on vulnerable points, focal loss, (3) **self-consistency**:
  similarity of every pixel to one vulnerable point, C = 1 − |b_uv − B|, BCE loss.
- Loss = BCE + λ1·heatmap + λ2·consistency, λ1 = 10, λ2 = 100.
- At test time only the classification output is used (code: `METRICS_BASE: binary`); frame scores
  are averaged per video.

## What the official code actually does (read 2026-11-23, SBI config `configs/efn4_fpn_sbi_adv.yaml`)

Agrees with the paper: 384×384 input, EfficientNet-B4, SAM optimizer, weight decay 1e-4, lr 5e-5
rising to 2e-4 over the first quarter then linear decay to 0, 100 epochs, backbone frozen for 6
epochs, 8 frames per video per epoch, 16 images per batch (config `batch_size: 8` reals, each with
its pseudo-fake), official FF++ train/val/test split, seed fixed (317), checkpoint chosen on the
FF++ validation split (not on test).

### Mismatches and things the paper does not say (each needs a decision before we train)

1. **FF++ compression.** Paper does not state it for training or the main test. Code and README use
   `c0` (raw). The supplement gives one c23 test result (89.30). We only have c23 at hand.
2. **Loss weights.** Paper: λ1 = 10, λ2 = 100. Released SBI config: `hm_lmda: 100`,
   `cstency_lmda: 100` (the BI config has 10 / 100). By the paper's own Table 5, 100/100 is worse.
3. **SBI config is a mid-training snapshot.** `begin_epoch: 48`, `resume: True`, and a private
   checkpoint path. It must be edited to train from scratch.
4. **Vulnerable points.** Paper: argmax of B over the whole mask. Code (`_new_encode_target`): the
   mask is cut into 4 quadrants and the maximum of each quadrant gives points, so up to 4 groups.
5. **Label smoothing.** Paper says it is used. In code there is no smoothing of the class label; the
   only trace is `epsilon=0.35` inside the heatmap loss, which just scales its positive term by 0.72.
6. **Heatmap loss.** Paper Eq. 5 is a plain focal loss. Code uses the CornerNet form: γ = 2 and
   negatives weighted by (1 − target)^4.
7. **Consistency head.** Paper describes a predicted map Ĉ. Code predicts a 256-channel feature map
   and builds Ĉ as sigmoid(dot product with the feature at the ground-truth vulnerable point / 16),
   so it needs the ground truth and is training-only.
8. **Pretraining.** Paper: "pretrained on ImageNet". Code: EfficientNet-B4 **AdvProp** weights.
9. **Optimizer base.** Paper: SAM. Code: SAM wrapped around Adam.
10. **Face crop.** Paper: "RetinaNet", enlargement 1.25, resized to 384. Code: RetinaFace
    (ResNet-50), 25% of box width/height added on **each** side, saved at 256×256, then warped to 384.
    At test a further 5-pixel margin is cut off.
11. **Landmarks.** Paper: 68 Dlib landmarks. Code/README: 81-landmark predictor for SBI, 68 for BI.
12. **Test-time flip.** Not in the paper. Code default `flip_test: True` (averages image and mirror).
13. **Test frames.** Paper gives 32 frames only for validation. Code tests on 32 evenly spaced frames
    per video and averages the raw (pre-sigmoid) outputs per video.
14. **Which DFDC.** Table 1 says DFDC but cites the DFDC *preview* paper; the code has loaders for
    both. Test subsets for DFD and DFW are also not described.
15. **Precision.** Whole model runs in float64 (`PRECISION: float64`), which costs speed and memory.
16. **Environment.** Pinned to PyTorch 1.8 / CUDA 11.4, Linux paths; video id is taken with
    `path.split('/')`, which breaks on Windows paths.
17. **Ablation baseline.** The "no component" row gets 74.54 on CDF2, far below what SBI alone
    reports (93.18) with the same backbone and the same pseudo-fakes. The paper does not explain this.

18. **Augmentation (found later, D009).** Paper lists Gaussian noise, blurring and colour jitter.
    In the SBI data path the colour pipeline is commented out; only RGB shift, hue/saturation,
    brightness/contrast and JPEG compression are applied, plus flip/crop/scale/erasing. Possible plain
    cause of the 53.9 AUC under noise. BI path not checked.

19. **Released weights vs released config (D015).** The SBI weights file name says hm10,
    "1e7_boost500", "UnFZ"; the SBI config says hm 100, lr 5e-5 ×4, freeze 6 epochs. The config does
    not describe how the released weights were trained.
20. **Real-folder name.** `images_crop.py` writes `frames/Original`; the test loader treats only
    `original` as real. As written, real test videos get the fake label.
21. **Test frame sampling.** 32 draws with replacement from the 32 saved frames (seeded).
22. **Silent errors.** A failed image load at test time is replaced by a random other image.

23. **FF++ test pools fake types (D020).** A video is identified by folder name only, so with all
    four fake types in one run the four fake versions of a pair (`000_003`) become one video. The
    official all-types number is over 140 real and 140 mixed fake "videos", not 560 fakes.

24. **Frame resampling is a no-op in the SBI loader (D023).** `_reload_data()` writes attributes
    that `__getitem__` never reads, so the 8 frames per video drawn at start-up are used for all
    epochs (at most 5,760 real frames, not 92,160).
25. **Landmark step (D023).** README flag name is wrong, the released c23 preprocessing config
    does not match the crop output and lacks `COMPRESSION`, the output file name differs from the
    one in the training config, a failed save is only printed, dlib is not in the requirements.
26. **From scratch (D023).** Needs `begin_epoch: -1`, `resume: False`, no checkpoint path.
    `train.py` also needs test crops to exist. The training loader retries failed samples for ever.

### Current state of Stage 1

- Official repo cloned unmodified in `LAA-Net/` (commit `a33446a`).
- **R0 is done (D022, 2026-11-24)** = released SBI weights + official test code on the FF++ c23
  test split, on Kaggle (`kaggle/r0_released_weights.ipynb`), 5.3 h on T4 x2, all 700 videos
  cropped with 32 frames. Video-level AUC / AP %:

  | Run | Label | AUC | AP | ACC@0.5 |
  |---|---|---|---|---|
  | all four types, official layout (140 real / 140 pooled fake) | official code as-is | 94.93 | 94.34 | 83.57 |
  | all four types, every video separate (140 / 560) | our run | 89.07 | 97.07 | 65.00 |
  | Deepfakes | official code as-is | 95.77 | 96.74 | 89.64 |
  | Face2Face | official code as-is | 87.43 | 88.27 | 73.21 |
  | FaceSwap | official code as-is | 92.44 | 93.18 | 78.57 |
  | NeuralTextures | official code as-is | 80.59 | 81.57 | 65.36 |

  Paper c23: 89.30. The "every video separate" run (89.07) and the per-type mean (89.06) match it
  within 0.2; the official pooled layout is 5.6 higher. At threshold 0.5 about 97% of reals but
  only about 57% of fakes are called correctly (inferred), so use AUC/AP, not accuracy.
  Hardest types: NeuralTextures, then Face2Face. Raw-video 99.96 is not tested (no raw data).
- R0 run record is in `runs/r0` (D025): six test logs, test split, times, crop counts, package
  list; copied as text through the browser. The environment had two OpenCV packages installed
  (`opencv-python` 5.1 and `opencv-python-headless` 4.5.1); pin one before R1.
- Measured on a T4: cropping about 15 s per video (32 frames); testing about 2.9 s per batch of 8
  images with flip in float64.
- **R1 plan (D029, user's decision 2026-11-25):** two runs side by side on the faculty machine,
  R1-control (official method) on GPU 0 and R1-fixed (with flaw fixes) on GPU 1, same data, seed
  and epochs, both float32. Both are "our run". Upgrade ideas stay out of this round.
- **R1 settings (D031):** FF++ c23, SBI variant, loss weights 10 / 100, 100 epochs (6 frozen),
  seed 317. R1-fixed differs from R1-control by one line: `_reload_data()` writes the `_r`
  attributes, so new frames are drawn each epoch. Both runs also get float32 and two bookkeeping
  additions (an `R1_EPOCH` log line and a `model_last.pth` checkpoint each epoch).
- **Faculty scripts (`faculty/`, D031):** `00_setup_env.sh`, `01_prepare_data.sh <FF++ folder>`,
  `02_check_port.sh` (released weights vs R0), `03_train.sh smoke|full`, `04_test_runs.sh`,
  `status.sh`, `common.sh`, and the helper `laa_r1.py` (patch, configs, probe, summaries).
  Tested on this PC on CPU with the faculty package versions and synthetic data: patches apply,
  both variants train 2 epochs (epoch 0 identical, epoch 1 differs), checkpoints load, test script
  runs. **Nothing has run on the faculty machine yet**; GPU, 7-worker loading, RetinaFace and dlib
  there are untested.
- **GPU sharing (D034):** the faculty GPUs are shared with the temporal branch's training. Every
  script takes `GPUS` (default `0 1`), e.g. `GPUS=1 bash 03_train.sh full control`. The two R1 runs
  may start at different times on different GPUs. `run_night.sh` chains prepare → port check
  (stops if an AUC is more than 1.0 from R0) → trial → full R1-control on one GPU, unattended.
- **USB bundle:** `to_faculty/` (scripts, dlib file, released SBI weights, README, manifest; 455 MB),
  rebuilt with `python tools/make_faculty_bundle.py` after any script change.
- Downloads on the faculty machine (RetinaFace 96.9 MB, AdvProp 78 MB) approved 2026-11-25 (D032).
  Waiting for the user to run the steps there and send each log back.
- When editing files from Bash heredocs on this PC, backslashes are collapsed; use the Write/Edit
  tools for text that contains backslashes.
- **Timing test done (D027, 2026-11-25, Kaggle, one T4, timings only):**
  `kaggle/r1_timing_test.ipynb` (generator `tools/make_r1_timing_notebook.py`).
  - As released (float64): 7.9 s per training step with the backbone frozen, then **out of
    memory** when it unfreezes. The official code as released cannot train on a Kaggle T4.
  - float32 copy: 0.84 s per step frozen, 2.2 s unfrozen, 0.31 s per validation step, peak
    13.7 GB; ran end to end from scratch. Projection: **about 44 h for 100 epochs** on one T4.
  - Preparation: crop 48.9 s per train video (128 frames), 9.8 h for 720 videos on one GPU;
    landmarks about 1.1 h. About 3% of frames get no landmarks; the loader silently replaces them.
  - Second GPU: batch is `batch_size × number of GPUs`, so using it changes the batch (deviation).

### What a reproduction needs

- **Training data:** FF++ original (real) videos only: 720 train + 140 val by the official split,
  128 frames per train video and 32 per val video, face-cropped, with landmarks.
- **Test data:** FF++ test split (140 real + 140 for each of 4 fake types), Celeb-DF v2, DFD, DFDC,
  WildDeepfake. None is on this PC. The Kaggle FF++ c23 copy covers the FF++ part at c23 only.
- **Model files (all downloads, ask first):** RetinaFace ResNet-50 weights, dlib 81-landmark
  predictor, EfficientNet-B4 AdvProp weights, and optionally the authors' trained LAA-Net weights.
- **Compute (paper):** one Tesla V100, 100 epochs. Per epoch 720 steps of 16 images at 384×384, and
  SAM does two forward/backward passes per step. Time on a Kaggle T4 is not measured yet.
- **Cheapest first check:** run the authors' released weights on FF++ c23 test in a Kaggle notebook
  ("official code as-is"); no training needed. Not decided (O3).

## Resources known so far

- **This PC:** Windows, no GPU. Code is prepared here.
- **Datasets:** none on this PC's disks. The user has FF++ c23 on a **USB drive** (seen as
  `G:\FaceForensics` while plugged in; official layout, D030, corrected in D031): YouTube originals
  1,000, five fake types 1,000 each, DFD 363 real + 3,068 fake; 32.7 GB. Counts and sizes checked,
  videos not decoded. The user is carrying it to the faculty machine.
- **FF++ c23 on Kaggle:** `xdxd003/ff-c23` (17.92 GB; 1,000 mp4 each in `original`, `Deepfakes`,
  `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `DeepFakeDetection`). Unofficial copy; the
  official FF++ terms still apply. `DeepFakeDetection` there is fakes only, not the full DFD test set.
  The temporal branch found its frame counts match the authors' for the videos it checked
  (`F:\Ti2net\DECISIONS.md`, D018 and D020).
- **Compute:** Kaggle notebooks (GPU T4, 4 CPU cores, 12 h sessions) for heavy work.
- **Faculty machine (seen 2026-11-25, D026):** Linux, 2 × RTX 5090 (32 GB each), driver 595.99,
  CUDA 13.2, conda base Python 3.8.5. PyTorch 1.8 cannot use these cards; it needs a recent PyTorch
  (CUDA 12.8+ build, Python 3.9+), so the official code would not be "as-is" there. Disk, internet
  and availability not known.
- **Training runs on the faculty machine (D028, user's decision 2026-11-25).** Environment chosen:
  conda `laa310`, Python 3.10, torch 2.7.1+cu128, torchvision 0.22.1, opencv-python-headless
  4.5.5.64, the rest as in R0. Script `faculty/00_setup_env.sh` (packages and code only; not run
  yet). The user runs things there over remote desktop; I cannot reach that machine. Estimate for
  100 epochs on one 5090 in float32: 4–8 h (not measured). Port check planned: repeat R0 there.
- Nothing above is a decision for this branch yet; what we use is decided after the paper is read.

## Layout (planned; folders are created when first needed)

- `LAA-Net/` — clone of the official repo (commit `a33446a`), kept unmodified as reference.
- `kaggle/` — notebooks that run on Kaggle (generated by `tools/`).
- `faculty/` — scripts for the faculty machine; `to_faculty/` — the bundle copied to it (generated).
- `runs/` — records of finished runs (`r0`, `r1_timing`).
- `src/` — our reproduction code.
- `docs/` — notes.
- `Pretrained/` — authors' weights zip (SBI + BI) and the dlib 81-landmark file, placed by the user,
  checked in D021. Not committed.
- `data/`, `weights/` — not committed.
- `DECISIONS.md` — decision log.
