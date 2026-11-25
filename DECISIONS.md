# Decision log — visual branch

Newest entries at the bottom. Never edit a past decision; add a new entry that supersedes it.

Format: **ID — date — title** / Decision / Why / Alternatives rejected / Status.

---

## Decided

### D001 — 2026-11-23 — Scope is the visual part only
- **Decision:** work here covers only the visual (spatial / frame-level) branch of the multimodal
  detector. Temporal (`F:\Ti2net`, another session), audio and fusion are out of scope. Nothing in
  `F:\Ti2net` is edited from here.
- **Why:** one branch per folder (`F:\Multimodal\PROJECT.md`).
- **Status:** active.

### D002 — 2026-11-23 — Order of work: reproduce → diagnose → beat → new paper
- **Decision:** first reproduce the reference paper, then analyse its failures, then beat it, and
  only then plan a new paper. Shared rule 1 of the project map.
- **Why:** we need our own baseline numbers, and the failures we measure should choose the upgrade,
  not ideas from reading alone.
- **Alternatives rejected:** picking a new-paper idea first.
- **Status:** active. Stage 1 open, waiting for the reference paper (O1).

### D003 — 2026-11-23 — Keep CLAUDE.md and this decision log
- **Decision:** `CLAUDE.md` holds project context and rules; `DECISIONS.md` records every decision
  and every deviation from paper or code. Same format as the temporal branch.
- **Why:** shared rule 2 of the project map; the fusion stage reads every branch's log.
- **Status:** active.

### D004 — 2026-11-23 — The user chooses the reference paper
- **Decision:** the visual reference paper is supplied by the user. No paper is picked and no
  literature search is run from this side.
- **Why:** instruction for this branch.
- **Status:** active; see O1.

### D005 — 2026-11-23 — Reference paper is LAA-Net (closes O1)
- **Decision:** LAA-Net (Nguyen et al., CVPR 2024, "Localized Artifact Attention Network for
  Quality-Agnostic and Generalizable Deepfake Detection") is the reference for the visual part.
  PDF placed in `F:\Visual` by the user.
- **Why:** chosen by the user.
- **Status:** active.

### D006 — 2026-11-23 — Finding: paper and official code read
- **Read:** the 11-page main paper in full (supplementary not available here) and the official repo
  https://github.com/10Ring/LAA-Net at commit `a33446a` (shallow copy in a temporary folder, not in
  this project yet). Files read: README, `configs/efn4_fpn_sbi_adv.yaml`, parts of
  `configs/efn4_fpn_hm_adv.yaml`, `scripts/train.py`, `scripts/test.py`, `lib/core_function.py`,
  `lib/metrics.py`, `lib/scheduler/linear_decay.py`, `losses/losses.py`,
  `models/networks/pose_efficientNet.py`, `datasets/common.py`, `datasets/face_forensic_sbi.py`,
  `datasets/sbi/utils.py`, `package_utils/images_crop.py`. Not read: the BI data path, the
  cross-dataset loaders, the landmark extraction script, the licence text.
- **Targets (paper, Table 1, w/ SBI, video-level AUC):** FF++ 99.96, CDF2 95.40, DFW 80.03,
  DFD 98.43, DFDC 86.94. Full table in `CLAUDE.md`.
- **Official code exists and trained weights are released** (BI and SBI, Dropbox).
- **Seventeen gaps or mismatches between paper and code** are listed in `CLAUDE.md`. The ones that
  change results most plausibly: FF++ compression (code uses raw, paper silent), loss weights in the
  released SBI config (100/100, paper says 10/100), vulnerable points taken per quadrant instead of
  one global argmax, label smoothing not present on the class label, test-time flip on by default.
- **Good news compared with the temporal branch:** the code fixes the seed, uses the official FF++
  split, and picks the checkpoint on the validation split, not on test.
- **Weak points stated or shown in the paper itself:** AUC 53.9 on FF++ under Gaussian noise
  (Table 2); the consistency branch does not help on CDF2 or DFD (Table 3); the "no component"
  ablation row (74.54 on CDF2) is far below SBI's published 93.18 and this is not explained; no
  results at c23 or c40; no per-manipulation FF++ results.
- **Status:** finding.

### D007 — 2026-11-23 — Upgrade directions parked as unverified notes
- **Context:** the user asked to brainstorm what could be upgraded into a new paper. Under D002 the
  new-paper stage comes last, so these are recorded only, not acted on, until O2 is answered.
- **Candidates (from reading only; no novelty check, no experiment):**
  - A. Robustness to noise and compression. The paper's own stated future work; 53.9 AUC under noise.
  - B. Fakes with no blending boundary (fully generated faces, diffusion, lip-sync). The method
    assumes every fake has a blend; the paper never tests this.
  - C. Using time. Also the paper's stated future work, and in our project it is where the visual and
    temporal branches meet. To check first: I recall a follow-up from the same group on
    vulnerability-driven video detection (2025); not verified.
  - D. Using the heatmap at test time. Today it only shapes training; the decision uses the class
    output alone. It could also serve as the localisation signal handed to fusion.
  - E. Testing the "quality-agnostic" claim properly (c23 / c40, not only Mask-SSIM bins on CDF2).
- **Why parked:** which of these matters should come from our own failure measurements (Stage 2).
- **Status:** parked.

### D008 — 2026-11-23 — Brainstorm before the first run (answers O2 in part; narrows D002)
- **Decision:** the user chose to brainstorm upgrade ideas before the first reproduction run.
  Brainstorming and planning happen now; no idea is built or committed to until we have our own
  LAA-Net numbers. The order reproduce → diagnose → beat → new paper still holds for experiments.
- **Why:** the user's choice on 2026-11-23 ("Brainstorm ideas first").
- **Alternatives rejected:** running the released weights first (my recommendation, O3).
- **Note:** this differs from shared rule 1 of `F:\Multimodal\PROJECT.md` only in when ideas are
  discussed, not in when they are built.
- **Status:** active.

### D009 — 2026-11-23 — Finding: the SBI training path applies no noise or blur augmentation
- **Finding:** the paper lists Gaussian noise, blurring and colour jitter among the augmentations,
  and the config has `gaussnoise: 0.3`, `gaussianblur: 0.3`. But in `datasets/face_forensic_sbi.py`
  the line that builds the colour pipeline is commented out; the SBI dataset uses
  `get_transforms()` from `datasets/sbi/utils.py`, which has only RGB shift, hue/saturation,
  brightness/contrast and JPEG compression (quality 40–100, p = 0.5). The geometry pipeline
  (flip, crop, scale, random erasing) is still built.
- **Effect:** the SBI model may never see noise or blur in training. This is a plausible plain cause
  of the 53.9 AUC under Gaussian noise (paper Table 2). Not verified by a run, and I have not checked
  whether the BI path differs.
- **Consequence for ideas:** direction A (noise robustness) may be largely fixed by switching the
  augmentation on. That must be measured before A is treated as research.
- **Status:** finding; mismatch 18 in `CLAUDE.md`.

### D010 — 2026-11-23 — Brainstorm round 1 (supersedes the list in D007)
From reading only. "Related work" lines are from memory and **not verified**; no search was run.
- **A. Robust vulnerable-point learning.** Make heatmap and decision stable under noise, blur and
  compression (for example: same image clean and degraded must give the same heatmap).
  Evidence: Table 2 noise 53.9. Risk: D009 says plain augmentation may fix most of it; robustness
  to compression is a crowded topic.
- **B. Pseudo-fakes without a blend boundary.** LAA-Net assumes every fake is a blend. Add
  training fakes that mimic generator artefacts (re-encoding a face through an autoencoder or
  diffusion model, frequency-level changes) and define "vulnerable points" for them.
  Evidence: none in the paper; testable on FaceShifter and newer sets. Risk: crowded; several
  2024 papers extend SBI this way.
- **C. Vulnerable points over time.** Predict heatmaps per frame and learn how they should move.
  Risk: I recall the LAA-Net authors published this themselves in 2025; likely taken.
- **D. Heatmap as the bridge to the temporal branch.** TI2Net looks at identity change of the whole
  face between frames, and the temporal branch found that signal is dominated by face-box jitter
  (`F:\Ti2net\DECISIONS.md`, D021). Use LAA-Net's heatmap to say *where* to measure change over
  time: identity or texture consistency around vulnerable regions, or the stability of the heatmap
  itself across frames, as the feature handed to fusion.
  Evidence needed: do heatmaps on real deepfakes land on the true manipulation boundary? FF++ ships
  manipulation masks, so this can be measured. Risk: overlaps with C if done as visual-only.
- **G. Mouth-region self-blend across time, for the audio branch.** Make pseudo-fakes by blending
  the mouth area from another moment of the same real video. This imitates lip-sync fakes with real
  data only, keeps known vulnerable points, and creates audio–lip mismatch for the future audio
  branch to train on. Risk: related pseudo-fake ideas for audio-visual detection may exist.
- **Dropped as a paper idea, kept as an evaluation:** E (c23/c40 testing) is part of Stage 1/2.
- **My ranking for this project:** D and G first (they use the fact that we are multimodal), then B;
  A only if augmentation does not fix it; C last.
- **Status:** parked ideas; next step is O8.

### D011 — 2026-11-23 — Finding: supplementary material read; corrects part of D006
Source: arXiv 2401.13856v2 (its Sections 6–8 are the supplement), read through a fetch helper, not
by me line by line. The CVPR supplemental PDF itself could not be fetched (HTTP 403).
- **Correction to D006:** D006 said the paper has no results at c23. The supplement (Sec. 8.3) reports
  one: LAA-Net tested on FF++ **c23 gives AUC 89.30** (against 99.96 in Table 1). No c40 result, no
  other method compared, and it does not say which compression the model was trained on.
- **Still not stated anywhere in paper or supplement:** training compression, compression of the main
  FF++ test, DFDC version, DFD subset, test frames per video, test-time flip, label-smoothing value,
  per-manipulation FF++ results.
- **Extra targets (paper, supplement Tables 6–7, w/ SBI):** FF++ ACC 99.04, AP 99.99, AR 99.29,
  mF1 99.64; cross-dataset AR / mF1: CDF2 87.71 / 92.41, DFW 65.66 / 72.56, DFD 88.55 / 93.64,
  DFDC 73.37 / 83.81.
- **Use:** 89.30 is the paper's own number to compare with when we run the released weights on the
  Kaggle FF++ c23 copy.
- **Status:** finding.

### D012 — 2026-11-23 — Novelty check of the five ideas in D010 (closes O8)
Method: seven web searches and four page fetches through helpers. Short check, not a full survey.
Where a point rests on a search snippet or on a helper's summary, not on text read verbatim, it is
marked (unverified).
- **A. Noise/compression robustness — taken.**
  - Hopf and Timofte, "Practical Manipulation Model for Robust Deepfake Detection" (arXiv 2506.05119,
    Jun 2025; an ICCV 2025 workshop version appears in search results): adds strong degradations and
    generator artefacts to SBI-style pseudo-fakes, on top of LAA-Net. Their Table 1 (their own run
    of the official LAA-Net code where starred): LAA CDF2 0.9540, DFDC 0.7170*, DFDCP 0.8694;
    with their method 0.9153 / 0.7521 / 0.9315. They report LAA-Net at chance under Gaussian noise
    (unverified wording).
  - LAA-X (arXiv 2604.04086, 2026), the LAA-Net authors' journal extension with transformer
    backbones, claims better noise robustness (unverified; snippet only).
  - FakeSTormer also reports better robustness than LAA-Net.
- **B. Pseudo-fakes without a blend boundary — crowded.** FreqBlender (arXiv 2404.13872, frequency
  blending), ProDet (NeurIPS 2024, arXiv 2408.17052), the Hopf–Timofte paper above (Stable Diffusion
  and GAN artefacts in pseudo-fakes), and a 2026 critique, "The Alpha Blending Hypothesis"
  (arXiv 2605.10334; not read, fetch failed).
- **C. Vulnerable points over time — taken by the same authors.** FakeSTormer, "Vulnerability-Aware
  Spatio-Temporal Learning for Generalizable Deepfake Video Detection" (Nguyen et al., ICCV 2025,
  arXiv 2501.01184): TimeSformer backbone, self-blended videos, a temporal head that regresses how
  the blending boundary changes between frames, a spatial head, real data only. Reported
  (FF++ c23 training): CDF 92.4, DFD 98.5, DFDCP 90.0, DFDC 74.6, DFW 74.2.
- **G. Mouth self-blend across time for audio-visual — mostly taken (2026).**
  - AVPF (Wei and Li, arXiv 2604.09110, Apr 2026): shifts the video in time and blends it back
    inside a face mask, shifts and blends the audio, splices snippets; real data only.
  - SAVe (Shahzad et al., arXiv 2603.25140, Mar 2026): lip and lower-face self-blends plus an
    audio–lip sync branch; real data only.
  - Gap that seems left: neither predicts where the forgery is; AVPF lists localisation as future
    work (unverified wording). Vulnerable-point supervision on such pseudo-fakes was not found.
- **D. Heatmap as the bridge to the temporal/identity branch — no direct match found.** Closest is
  FakeSTormer, which uses no identity embedding, no audio, and uses its vulnerability outputs only
  as training signals, not in the decision. Two targeted searches found no work that measures
  identity or texture change over time at the predicted vulnerable regions, or that passes the
  heatmap to a fusion stage. Not finding it in a short search is weak evidence; a proper search is
  needed before this is called novel.
- **Side finding for Stage 1:** two independent sources put LAA-Net's DFDC number well below the
  paper's 86.94 (Hopf–Timofte's own run 71.70; the DFDC subset may differ). Our reproduction should
  expect this and state exactly which DFDC subset is used.
- **Outcome:** A and C dropped as paper ideas. B only as a component. D is the lead candidate; the
  localisation gap in G can be attached to it later when the audio branch exists.
- **Status:** finding; ideas remain parked until we have our own numbers (D008).

### D013 — 2026-11-23 — Back to Stage 1 only (closes O2)
- **Decision:** the user confirmed the order: implement the existing paper, then beat or upgrade it,
  and only after that think about a new paper. Work is now Stage 1 (reproduce LAA-Net) only.
  The ideas in D010/D012 stay parked.
- **Note:** the user's message said "temporal part"; this folder and the repo they linked
  (https://github.com/10Ring/LAA-Net) are the visual branch, so it is read as the visual part.
- **Status:** active.

### D014 — 2026-11-23 — Official repo cloned, kept unmodified (closes O7)
- **Decision:** cloned to `F:\Visual\LAA-Net` at commit `a33446a`. This copy stays as the reference;
  changes needed to run are made in the run environment (Kaggle) and logged.
- **Status:** active.

### D015 — 2026-11-23 — Finding: released weights and the official test path
- **Released weights** (authors' Dropbox folder "LAA-Net pretrained", seen in the browser), 415.32 MB each:
  - `PoseEfficientNet_EFN_hm10_EFPN_NoBasedCLS_Focal_C3_256Cst100_8SBI_SAM(Adam)_ADV_Era1_OutSigmoid_1e7_boost500_UnFZ_model_best.pth`
  - `PoseEfficientNet_EFN_hm100_EFPN_NoBasedCLS_Focal_C3_256Cstency100_32BI_SAM(Adam)_ADV_Erasing1_OutSigmoid_model_best.pth`
- **Read from the file names only (not confirmed):** the SBI weights were trained with heatmap
  weight 10 (the paper's value, not the 100 in the released SBI config), with "1e7_boost500", which
  would be a learning rate from 1e-7 up to 5e-5, and "UnFZ", which suggests the backbone was not
  frozen. None of these three matches the released SBI config (hm 100, lr 5e-5 with booster 4,
  freeze 6 epochs), and the last two do not match the paper. So the released config does not
  describe how the released weights were trained.
- **Official test path (code read):**
  - `images_crop.py` saves real videos under `frames/Original`, but the test loader labels a folder
    real only if its name is `original`. Used as written, real test videos would be labelled fake.
  - Test frames are drawn with `random.choices` (with replacement): 32 draws from the 32 saved
    frames, so some frames repeat and some are never used. Seeded (317).
  - Any exception while loading a test image is swallowed and a random other image is returned.
  - `COMPRESSION: c0` only selects the folder layout; it does not check what the data is.
  - `images_crop.py` has fixed paths and, when resuming a partly cropped video, skips reading frames
    that already exist, which shifts later frames. Do not resume inside a video.
- **Environment:** the code does not import on current library versions (`DualTransform` import
  path in albumentations, imgaug with numpy 2). It needs the pinned versions (PyTorch 1.8,
  albumentations 1.1.0).
- **Status:** finding; mismatches 19–22 in `CLAUDE.md`.

### D016 — 2026-11-23 — R0 prepared: released SBI weights + official test code on FF++ c23 test
- **Proposal (O3 option a):** first run "R0" = the authors' released SBI weights evaluated with the
  official `scripts/test.py` on the official FF++ test split (70 pairs: 140 real + 140 per fake
  type), from the Kaggle c23 copy. Compared with the paper's 89.30 (c23, supplement) and 99.96.
- **Prepared:** `kaggle/r0_released_weights.ipynb`. Smoke mode (2 pairs) first, then full.
- **Changes from the official code, all needed to run:** Python 3.8 environment with the repo's
  pinned versions plus unpinned scikit-learn, matplotlib, retinaface-pytorch; two path constants in
  `images_crop.py`; videos linked into the folder tree the script expects; `frames/Original` renamed
  to `original`; in the config: test dataset name, root, folder list, weights path, `DATASET.TRAIN: False`.
  Flip test, 32 frames, seed and everything else as released.
- **Deviation to report with R0:** c23 data (the code's default is raw); the weights' training
  compression is unknown.
- **Not verified:** the notebook has not been run anywhere. Only its Python syntax was checked. The
  old PyTorch build on Kaggle's GPU, the retinaface package version and loading the checkpoint are
  the likely trouble spots; the smoke run is there to find them.
- **Needs the user's approval (downloads):** the SBI weights file (415.32 MB, authors' Dropbox),
  uploaded by the user as a private Kaggle dataset; on Kaggle, the RetinaFace ResNet-50 detector
  weights fetched by the `retinaface-pytorch` package (size not checked) and the FF++ `test.json`
  split file from the official FaceForensics GitHub repository.
- **Status:** waiting for approval.

### D017 — 2026-11-23 — R0 approved; weights fetched by the notebook (closes O3)
- **Decision:** the user approved the R0 downloads ("Yes, approved"): the released SBI weights, the
  RetinaFace detector weights fetched by `retinaface-pytorch`, and the FF++ `test.json` split file.
  R0 (D016) is the first run.
- **Change to D016:** the user does not need to upload the weights. Checked from this PC that the
  authors' Dropbox folder link with `dl=1` returns one zip (`LAA-Net pretrained.zip`,
  870,989,302 bytes). The notebook downloads it on Kaggle, extracts only the SBI file, deletes the
  zip, and prints the file's SHA-256. The zip also contains the BI weights (not kept). An attached
  Kaggle dataset with the SBI file is used instead if present.
- **Nothing was kept on this PC:** the check pulled the first 48 MB of the zip into a temporary
  folder, which was deleted.
- **Also kept:** `kaggle/make_r0_notebook.py`, the script that generates the notebook.
- **Status:** active; waiting for the smoke run on Kaggle.

### D018 — 2026-11-23 — Generator script moved out of `kaggle/`
- **What happened:** the user ran `make_r0_notebook.py` (it prints "ok 15") instead of the notebook;
  in that file the notebook's code sits inside text strings, so it looked like comments.
- **Decision:** the script now lives at `tools/make_r0_notebook.py` (supersedes the path in D017).
  `kaggle/` holds only files meant to be uploaded to Kaggle. The notebook itself was re-checked:
  15 cells, 8 code and 7 text, all code cells parse.
- **Status:** done.

### D019 — 2026-11-23 — Finding: R0 smoke run works on Kaggle
Run by the user on Kaggle (GPU T4), smoke mode: first 2 pairs of the FF++ test split, c23.
- **Works without any further change:** Python 3.8.20 + torch 1.8.0+cu111 + torchvision 0.9.0+cu111
  + albumentations 1.1.0 + numpy 1.23.3 on a Tesla T4; official repo at `a33446a`; RetinaFace
  cropping (4 videos per folder, 32 frames each, none short); the released SBI weights load with
  `strict=True`; the official test script runs to the end.
- **Weights file:** 415.32 MB, SHA-256
  `fde726610e778045f65c3fbe839bc463c4d3a9970f72ca109d5911917305c991`.
- **Kaggle path of the videos:** `/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23`.
- **Numbers (official code as-is, 4 real + 4 fake videos, so not a result):** video AUC 100 for
  Deepfakes, Face2Face, FaceSwap and all pooled; 93.75 for NeuralTextures. Accuracy at threshold
  0.5 is only 50–87.5, i.e. the ranking is right but the 0.5 threshold is off on these c23 videos.
  To be checked on the full split.
- **Speed:** 2.88 s per batch of 8 images (with flip test, float64), about 0.36 s per image.
- **Status:** finding.

### D020 — 2026-11-23 — Finding: the official FF++ test pools the four fake types into one video
- **Evidence:** in the all-types smoke run the loader read 640 images (20 videos × 32) but kept 256
  after sampling, which is 8 videos × 32, not 20.
- **Cause (code):** a video is identified by its folder name only (`path.split('/')[-2]`), both in
  the frame sampler and in the per-video averaging in `scripts/test.py`. `Deepfakes/000_003`,
  `Face2Face/000_003`, `FaceSwap/000_003` and `NeuralTextures/000_003` all have the name `000_003`.
  So with all four types in one run, the four fake versions of a pair are treated as one video:
  32 frames are drawn from their 128 and one averaged score is produced.
- **Effect:** the official "FF++" number with all types is over 140 real videos and 140 mixed fake
  "videos", not 560 fake videos. Averaging across four manipulations can hide a type the model is
  weak on. Whether the paper's 99.96 and 89.30 were computed this way is not known.
  Runs with a single fake type are not affected.
- **Decision for R0:** report three things, never mixed: (a) `all_asis`, the official layout,
  labelled "official code as-is"; (b) `all_unique`, the same code and weights with each fake video
  linked under a unique folder name so all 700 videos are scored separately (a data-layout change
  only, no code change), labelled "our run"; (c) each fake type against the 140 reals.
- **Notebook changes for the full run:** jobs are spread over two GPUs (T4 x2), each job is timed,
  finished jobs are skipped on a re-run, `MODE` defaults to `full`.
- **Expected time (from the smoke speed, cropping not measured):** tests about 6.7 GPU-hours in
  total, so about 3.5 hours on two GPUs, plus cropping 700 videos.
- **Status:** active; waiting for the full run. Mismatch 23 in `CLAUDE.md`.

### D021 — 2026-11-23 — Finding: files the user placed in `Pretrained/` checked
The user downloaded these into `F:\Visual\Pretrained`. Checked on this PC (CPU), nothing modified.
- **`LAA-Net pretrained.zip`**, 870,989,302 bytes: same size as the Dropbox download seen in D017;
  zip CRC test passes. Holds the two checkpoints, each 435,494,2xx bytes (415.32 MB).
- **SBI checkpoint** (`..._hm10_..._8SBI_..._UnFZ_model_best.pth`): SHA-256
  `fde726610e778045f65c3fbe839bc463c4d3a9970f72ca109d5911917305c991`, **identical to the file the
  Kaggle smoke run used** (D019). Saved at epoch index 46.
- **BI checkpoint** (`..._hm100_..._32BI_..._model_best.pth`): SHA-256
  `b1279335e0c5ad87e1bbf87de20deb9310621ae99bb88a9e974527b0d786d26b`. Saved at epoch index 48.
- **Both checkpoints** (opened with PyTorch's safe loader, `weights_only=True`): keys `epoch`,
  `state_dict`, `optimizer`; 803 tensors, 27,286,761 values, float64; EfficientNet-B4 stem
  (48 channels), decoder `deconv_1`–`deconv_4` + `inception_block`, three heads `hm` (1 channel),
  `cls` (1 output), `cstency` (256 channels); no NaN or Inf. This matches the model the SBI config
  builds (E-FPN with c3/c4/c51, no c2).
- **Reading:** the "best" checkpoints come from about half-way through the 100 epochs (46 and 48),
  i.e. the validation criterion in `train.py` (higher accuracy and lower loss than before) last
  fired there. The optimizer state is included, so resuming is possible.
- **`shape_predictor_81_face_landmarks.dat`**, 19,743,860 bytes, SHA-256
  `8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869`. Size is as expected for this
  file; I have no reference hash to compare with, and it was not loaded (dlib is not installed
  here). Needed only for training (SBI pseudo-fakes), not for R0.
- **Status:** finding. `Pretrained/` is a not-committed folder.

### D022 — 2026-11-24 — Result: R0 full run (released SBI weights, official test code, FF++ c23 test)
Run by the user on Kaggle (T4 x2, `kaggle/r0_released_weights.ipynb`, `MODE = 'full'`), finished
in 5.3 hours with no failed cell. Numbers below are copied from the run log the user pasted.
- **Inputs confirmed by the log:** 1000 mp4 in each of the five folders; weights SHA-256
  `fde72661…5c991` (same file as D019 and D021); repo at `a33446a`; only `images_crop.py` changed
  (2 lines, paths); torch 1.8.0+cu111 on Tesla T4; official `test.json` (70 pairs).
- **Crops:** 140 videos in each of the five folders, 32 frames each (4,480 per folder), no video
  with fewer than 32 frames.
- **Results (video-level, %, 32 frames per video, flip test on):**

  | Run | Label | Videos (real / fake) | ACC@0.5 | AUC | AP |
  |---|---|---|---|---|---|
  | `all_asis` | official code as-is | 140 / 140 pooled (D020) | 83.57 | 94.93 | 94.34 |
  | `all_unique` | our run (layout change only) | 140 / 560 | 65.00 | **89.07** | 97.07 |
  | Deepfakes | official code as-is | 140 / 140 | 89.64 | 95.77 | 96.74 |
  | Face2Face | official code as-is | 140 / 140 | 73.21 | 87.43 | 88.27 |
  | FaceSwap | official code as-is | 140 / 140 | 78.57 | 92.44 | 93.18 |
  | NeuralTextures | official code as-is | 140 / 140 | 65.36 | 80.59 | 81.57 |

  Mean of the four per-type AUCs: 89.06.
- **Against the paper:** paper FF++ c23 AUC is 89.30 (supplement Sec. 8.3). `all_unique` gives
  89.07 and the per-type mean 89.06, both 0.2 below it. The official pooled layout gives 94.93,
  5.6 above it. Paper Table 1's 99.96 has no stated compression; the code uses raw video, which we
  do not have, so it is not tested here.
- **Reading (mine, not proven):** the paper's c23 number matches a test where every fake video is
  scored separately (or a per-type average), not the pooled layout that the released loader
  produces when all four types are given. The paper does not say how it was computed. Pooling
  raises AUC because each pooled "fake video" averages frames from four fake types.
- **Threshold 0.5 (inferred from ACC and AR, not read from per-video scores):** in `all_unique`,
  455 of 700 correct with mean per-class recall 77.05 gives 136/140 reals correct and 319/560 fakes
  detected (57%). Per type, assuming the same 136 reals: Deepfakes 115/140, FaceSwap 84/140,
  Face2Face 69/140, NeuralTextures 47/140. So on c23 the scores still rank fakes above reals fairly
  well (AUC) but most fake scores sit below 0.5. Accuracy at 0.5 is not a useful number here; the
  paper reports AUC and AP only.
- **Hardest types on c23:** NeuralTextures (80.59), then Face2Face (87.43). Both are
  reenactment methods without a face-swap blending boundary. Input for Stage 2, no action now.
- **Measured times:** cropping 32–36 min per 140 videos (about 15 s per video, 32 frames) on one
  T4; testing 53–54 min per 280 videos (2.83–2.89 s per batch of 8 images with flip, float64);
  `all_unique` 132.8 min.
- **Not kept locally yet:** the per-run logs, `env_freeze.txt`, `split_test.json` and
  `times_full.json` are in the Kaggle notebook's Output tab only. No per-video scores are saved by
  the official test script.
- **Correction to the Open table:** the O6 note said the released weights' name says hm100. The
  SBI weights file name says `hm10` (D015); `hm100` is the BI file. Table fixed below.
- **Status:** R0 done. Stage 1 continues with R1 (train from scratch); O4–O6 and new O9 open.

### D023 — 2026-11-24 — Finding: the official training path, read before the timing test
Read: `scripts/train.py`, `lib/core_function.py`, `datasets/face_forensic_sbi.py`,
`datasets/common.py`, `datasets/ff.py`, `package_utils/geo_landmarks_extraction.py`,
`package_utils/images_crop.py`, `models/utils.py`, `lib/scheduler/linear_decay.py`, both
`data_preprocessing_*.yaml`. Nothing run yet; these are from reading the code.
- **Per-epoch frame resampling does nothing in the SBI loader (mismatch 24).** The config says
  `DATA_RELOAD: True` and "dynamically random number of frames in each epoch". `train.py` calls
  `train_dataset._reload_data()` each epoch, but in `SBIFaceForensic` that method writes
  `self.image_paths / labels / ot_props`, while `__getitem__` and `__len__` read
  `self.image_paths_r / labels_r / ot_props_r`, set once in `__init__`. So the 8 frames per video
  drawn at start-up (with replacement, seed 317) are the only real frames ever used: at most
  5,760 of the 92,160 cropped train frames. The BI loader (`face_forensic_hm.py`) does not have
  this slip. Not changed; whether to keep it for R1 is a decision to take with the user.
- **Landmark step does not run as documented (mismatch 25).** README flag `--extract_landmarks`
  is `--extract_landmark` in the script. `data_preprocessing_c23.yaml` says `images` / `jpg` /
  four fake types and has no `COMPRESSION` key, while the crop script writes `frames/Original/*.png`
  and the script reads `cfg.COMPRESSION`. The output is written as
  `processed_data/<comp>/<split>_<DATASET>_<type>_<N>.json`, not the
  `<split>_81_FF++_processed.json` named in the training config. A failed save is caught and only
  printed. All rotated images are held in memory until the end (about 18 GB for 92,160 train
  frames at 256×256). dlib is not in the repo's requirements.
- **Starting from scratch (mismatch 26).** `begin_epoch` is passed to the scheduler as
  `last_epoch`; any value other than −1 without a resumed optimizer raises in PyTorch 1.8
  (`initial_lr` missing), so a fresh run needs `begin_epoch: -1`, `resume: False` and no
  checkpoint path. `train.py` also builds a test loader, so test crops must exist even though the
  training loop never evaluates on them.
- **Loader hides errors.** As at test time (mismatch 22), a training sample that raises is
  replaced by a random other one in an endless loop, so a broken setup hangs instead of failing.
- **Validation** uses pseudo-fakes too (8 frames per val video), and the "best" checkpoint is the
  last epoch where val accuracy rose and val loss fell together.
- **Status:** finding. Mismatches 24–26 added to `CLAUDE.md`.

### D024 — 2026-11-24 — R1 timing test prepared (works on O9)
The user chose to time training before fixing the R1 settings.
- **What:** `kaggle/r1_timing_test.ipynb` (generator `tools/make_r1_timing_notebook.py`). It crops
  16 train videos (128 frames) and 8 val videos (32 frames) from the first pairs of the official
  splits, extracts 81-point landmarks with the official script, then runs the official `train.py`
  for 2 short epochs and reads seconds per step from its log, plus peak GPU memory from
  `nvidia-smi`. Output: `timing_summary.json` with a projection to 720 train + 140 val steps per
  epoch and 100 epochs. **No result numbers come from this run.**
- **Two jobs:** `f64` = official code as it is; `f32` = a copy of the repo with `torch.float64`
  replaced by `torch.float32` in `scripts/train.py`, `lib/core_function.py`, `losses/losses.py`.
  The `f32` job is for timing only. Using float32 in R1 would be a deviation and needs its own
  entry and the user's choice.
- **Config changes from `efn4_fpn_sbi_adv.yaml`:** data roots and annotation files;
  `begin_epoch: -1`, `resume: False`, no checkpoint (D023); timing-only: `epochs: 2`,
  `warm_up: 1` (epoch 0 frozen backbone, epoch 1 unfrozen), `SAMPLES_PER_VIDEO.TRAIN: 24` (48
  steps per epoch from 16 videos). Batch 8 + 8, SAM, 7 loader workers and loss weights untouched.
- **Code changes in the `f64` repo:** the same two path constants in `images_crop.py` as in R0.
  Landmark configs are written by the notebook (D023 explains why the released one cannot be used).
- **Guards:** a probe builds one pseudo-fake and one training sample with errors visible before
  training; each training job has a 75-minute limit; the landmark file is checked after the run.
- **Downloads the notebook makes on Kaggle (need the user's approval before the run):**
  dlib 81-landmark file, 19,743,860 bytes, github.com/codeniko/shape_predictor_81_face_landmarks
  (checked against the SHA-256 of the user's local copy, D021); EfficientNet-B4 AdvProp weights,
  78,000,637 bytes, github.com/lukemelas/EfficientNet-PyTorch release 1.0 (fetched by the official
  code); RetinaFace weights as in R0; the official FF++ `train.json` and `val.json`; the Python
  package `dlib-bin` 19.24.6. Sizes read from the servers' headers; nothing downloaded to this PC.
- **Checked locally:** notebook cells parse; the log pattern matches the repo's logger format;
  config edits apply to the released config. The notebook itself has not run anywhere yet.
- **Status:** waiting for approval of the downloads and the Kaggle run.

### D025 — 2026-11-25 — R0 run record saved to `runs/r0`; checked against D022
- **What:** the R0 result files were copied from the Kaggle notebook output (`yashwanthvemuri/laa-net1`,
  script version 373185070) into `runs/r0`: six `r0_full_*.log`, `split_test.json`,
  `times_full.json`, `crop_counts_test.json`, `env_freeze.txt`, and a `README.md` saying how.
- **How:** the user's browser downloads kept failing (single files open as text; the whole-output
  download breaks at 500–600 MB because it includes the environment, crops and weights folders).
  With the user's permission I read each file as page text through Chrome and wrote it locally.
  So the logs are text copies, not byte-identical files: runs of spaces are collapsed. All digits
  are kept. `split_test.json` was rebuilt and its SHA-256 equals the Kaggle file's text.
- **Check:** every number in the six logs equals D022. `split_test.json` holds 70 pairs (the
  official test list). `crop_counts_test.json`: 140 videos × 32 frames = 4,480 for each of the five
  folders. The pooled run's log shows the pooling directly: 22,400 image paths loaded, 8,960 kept.
- **New observation:** `env_freeze.txt` lists both `opencv-python 5.1.0.95` and
  `opencv-python-headless 4.5.1.48` in the same environment. Which one `import cv2` picked is not
  recorded. Not checked whether it matters; to pin to one package before R1.
- **Not copied:** the five `crop_test_*.log` (about 29 KB each); they stay on Kaggle.
- **Also in `F:\Visual` (placed by the user):** a 435 MB file named `PoseEfficientNet_…_model_best.zip`.
  It is the released SBI weights file itself (SHA-256 `fde72661…`, same as D021/D022), a duplicate of
  the one inside `Pretrained/LAA-Net pretrained.zip`. Left untouched.

### D026 — 2026-11-25 — Timing-test downloads approved; faculty machine configuration seen
- **Approved by the user:** the two weight downloads the timing notebook makes on Kaggle (dlib
  81-landmark file 19.7 MB; EfficientNet-B4 AdvProp 78 MB), as listed in D024. The timing test
  (`kaggle/r1_timing_test.ipynb`) can now be run by the user. Approval covers this notebook only.
- **Faculty machine (from the user's screenshot of `nvidia-smi; python --version`, 2026-11-25):**
  Linux, 2 × NVIDIA GeForce RTX 5090 with 32,607 MiB each, both idle; driver 595.99.02, CUDA 13.2;
  conda `base` with Python 3.8.5. CPU, RAM, disk, internet access and how long we may use it: not known.
- **Consequence (my knowledge, not tested on the machine):** the official pinned environment
  (PyTorch 1.8, CUDA 11.1 build) does not support RTX 50-series cards; they need a recent PyTorch
  built for CUDA 12.8 or later, which also needs Python 3.9 or newer. So the official code cannot
  run there "as-is": it needs a new environment and probably small code changes for the newer
  libraries. That is a deviation and must be logged when chosen.
- **Not decided:** where R1 runs (Kaggle T4 with the official environment, or the faculty machine
  with a new environment). Added as O11.

### D027 — 2026-11-25 — Result: R1 timing test on Kaggle (closes O9)
Run by the user on Kaggle (T4, one GPU used, official pinned environment, commit `a33446a`),
`kaggle/r1_timing_test.ipynb`; output pasted in chat. Timings only, no result numbers.
16 train videos (2,048 frames) and 8 val videos (256 frames); 48 training steps per epoch, 2 epochs.

| Measured | as released (float64) | float32 copy (timing only) |
|---|---|---|
| Training step, backbone frozen (16 images) | 7.895 s | 0.844 s |
| Training step, backbone unfrozen | **out of memory** | 2.216 s |
| Validation step | 2.472 s | 0.310 s |
| Peak GPU memory (`nvidia-smi`, includes PyTorch's cache) | 14,887 MiB, then crash | 13,655 MiB |
| Job exit | error (CUDA out of memory, card has 14.56 GiB) | finished, checkpoint saved |

Step times are medians of 9 logged lines per epoch, so they are rough. Data loading wait 0.005 s
per step, so the 4 CPU cores are not the limit.

- **Finding 1: the official code as released cannot train on a Kaggle T4.** It runs out of memory
  as soon as the backbone is unfrozen (epoch 7 of 100 in a real run). Time would also rule it out:
  a frozen epoch alone is 720 × 7.9 s = 1.6 h. Extrapolating the unfrozen step with the float32
  ratio (2.6×) gives about 21 s per step, about 4 h per epoch, about 400 h for 100 epochs
  (extrapolation, not measured).
- **Finding 2: with float32 it fits and runs end to end** (crop → landmarks → pseudo-fakes →
  train → validate → checkpoint, from scratch with `begin_epoch: -1`). Projection for 100 epochs
  on one T4: 6 × 0.18 h + 94 × 0.46 h = **about 44 h**. Memory is close to the card's limit.
  Kaggle allows 12 h per session and (as far as I know) about 30 GPU hours per week, so one full run
  is 4 sessions and about 1.5 weeks of quota; a control plus one fixed version is about 3 weeks.
- **Finding 3: preparation is not free.** Cropping 128 frames takes 48.9 s per train video:
  9.8 h for 720 videos on one GPU (about 5 h using both T4s side by side), plus 0.6 h for val.
  Landmarks 0.041 s per image: about 1.1 h for 92,160 images on one core.
- **Finding 4: some frames have no landmarks.** 60 of 2,048 train frames (2.9%) and 1 of 256 val
  frames. In the SBI loader such a frame raises an error and is replaced by a random other sample
  (`face_forensic_sbi.py` lines 126 and 261–264), silently. Same behaviour as mismatch 22/26.
- **Finding 5: the second T4 is not a free speed-up.** The config has `gpus: [0]` and the batch is
  `batch_size × number of GPUs`, so adding GPU 1 doubles the batch to 32 images. Keeping 16 images
  over two GPUs means 8 per GPU, which changes the BatchNorm statistics. Either is a deviation.
  Not timed.
- **Pipeline points confirmed by the run:** dlib-bin 19.24.6 works in the Python 3.8 environment;
  dlib file SHA matches; the landmark step works with our own config and `--extract_landmark`;
  the float32 edit is 22 changed lines in `scripts/train.py`, `lib/core_function.py`,
  `losses/losses.py`.
- **Consequence:** float64 training is not possible on Kaggle. Any R1 there is float32, which is
  a deviation the user must choose (O12). Whether float32 changes results is not measured; a cheap
  partial check exists: re-run the R0 test of the released weights in float32 and compare with D022.
- **Not measured:** anything on the faculty machine (O11).

### D028 — 2026-11-25 — Training moves to the faculty machine; environment port chosen (closes O11)
- **Decided by the user:** training runs on the faculty machine (2 × RTX 5090), and whatever
  environment or version changes that needs are made there. The user asked me to pick the versions.
- **Environment (my choice, delegated; a deviation from the official pinned environment):**
  conda env `laa310`, Python 3.10; `torch 2.7.1+cu128`, `torchvision 0.22.1+cu128` (official: 1.8.0 /
  0.9.0); `opencv-python-headless 4.5.5.64` (official 4.5.1.48 has no Python 3.10 build); one
  OpenCV package only (fixes the D025 observation); every other package at the R0 versions
  (numpy 1.23.3, albumentations 1.1.0, imgaug 0.4.0, scikit-image 0.19.3, scipy 1.9.3, pandas 1.3.5,
  Pillow 9.3.0, dlib-bin 19.24.6, retinaface-pytorch 0.0.7).
- **Why these:** PyTorch 2.7 is the oldest release line with a CUDA 12.8 build, which RTX 50-series
  cards need; checked on download.pytorch.org that 2.7.0–2.11.0 cu128 wheels exist for Python 3.10,
  and on PyPI that every pinned package has a Python 3.10 Linux wheel. Keeping numpy 1.23.3 avoids
  the removed `np.float`-style names and imgaug breakage. Alternatives rejected: newest PyTorch
  (bigger jump for no need); Python 3.8 (no PyTorch build for these cards).
- **Known risk points in the official code under PyTorch ≥ 2.6:** `torch.load` now defaults to
  `weights_only=True` (`models/utils.py` lines 48 and 79, `efficientNet.py` line 470); the released
  weights load that way (D021), our own checkpoints are untested. Nothing has run on the machine yet.
- **Script:** `faculty/00_setup_env.sh` builds the environment, clones the repo at `a33446a`,
  prints machine facts (cores, RAM, disk, internet) and a small float32/float64 speed check on both
  GPUs. It downloads Python packages and code only; no dataset, no weights. Syntax-checked here only.
- **Check planned for the port:** repeat the R0 test of the released weights in the new
  environment and compare with D022 before any training result is trusted.
- **Run time estimate for 100 epochs on one RTX 5090 (my estimate, not measured):** float32 about
  4–8 h; float64 probably days (consumer cards are slow in 64-bit). Basis: D027's T4 float32 steps
  (0.84 s / 2.2 s) and an assumed 6–10× speed-up; CPU pseudo-fake generation may cap it.
- **Still open:** O10 (the user wants flaw fixes and upgrade ideas in the first run; my
  recommendation is control on GPU 0 and fixed version on GPU 1 at the same time, ideas after),
  O12 (float32), O13 (epochs: 100 looks affordable there). Data and weights must be copied to the
  machine; each download needs the user's approval.

### D029 — 2026-11-25 — R1 = control and fixed version side by side, in float32 (closes O10, O12)
- **Decided by the user** (answer "Yes, side by side" to: control and fixed version side by side,
  one per GPU, in 32-bit?):
  - R1 is two training runs started together on the faculty machine: **R1-control** (the official
    method) on GPU 0 and **R1-fixed** (the official method with flaw fixes) on GPU 1, with the same
    data, seed, number of epochs and environment.
  - Both run in **float32**. Deviation from the official code (float64). Reason: float64 does not
    fit a Kaggle T4 and is expected to take days on a consumer card (D027, D028).
- **Alternatives rejected:** fixed version only (a result could not be attributed to the fixes,
  since we also change compression, PyTorch and precision); official code only with fixes later
  (slower for no gain when two GPUs are free).
- **Upgrade ideas (D010) stay out of this round** (my recommendation, D013 still stands); the user
  asked for them in the first run and has not objected to this since. To be confirmed if raised again.
- **Label for results:** both are "our run". Neither is "official code as-is" (new PyTorch, float32,
  c23, plumbing fixes).
- **Check on the deviations:** the R0 test of the released weights is repeated in the new
  environment in float32 and compared with D022 before training results are read.
- **Still to decide before training starts:** exactly which fixes go into R1-fixed (O14), loss
  weights (O6), compression (O4), variant (O5), epochs (O13).

### D030 — 2026-11-25 — Finding: the user downloaded FF++ c23 to this PC (`G:\FaceForensics`)
- **What:** the user downloaded FaceForensics++ c23 themselves (files dated 2026-11-24 15:35 to
  2026-11-25 10:33) and asked whether it is correct. Source not stated; the layout and the two
  leftover `tmp…` files look like the FF++ authors' own download script. To confirm with the user.
- **Checked here (file names, counts and sizes only; no video was decoded):**

  | Folder under `G:\FaceForensics` | mp4 files | Size |
  |---|---|---|
  | `original_sequences\youtube\c23\videos` | 1,000 (`000`–`999`) | 1.81 GB |
  | `manipulated_sequences\Deepfakes\c23\videos` | 1,000 | 1.86 GB |
  | `manipulated_sequences\Face2Face\c23\videos` | 1,000 | 1.81 GB |
  | `manipulated_sequences\FaceSwap\c23\videos` | 1,000 | 1.53 GB |
  | `manipulated_sequences\NeuralTextures\c23\videos` | 1,000 | 1.42 GB |
  | `manipulated_sequences\FaceShifter\c23\videos` | 1,000 | 1.79 GB |
  | `original_sequences\actors\c23\videos` (DFD real) | 363 | 2.70 GB |
  | `manipulated_sequences\DeepFakeDetection\c23\videos` (DFD fake) | 3,068 | 19.81 GB |

  These are the published FF++ counts (1,000 per YouTube folder; DFD 363 real and 3,068 fake).
  Smallest video 338 KB, none empty. Total about 32.7 GB.
- **Two stray files** in the original YouTube folder: `tmpnopmwzas` (2.8 MB) and `tmpvz3l_e6e`
  (0 bytes), left by interrupted downloads. Harmless (our scripts read `*.mp4` only); not deleted.
- **What it adds over the Kaggle copy:** the real DFD videos (`actors`), so the DFD cross-dataset
  test at c23 becomes possible (paper: 98.43, compression not stated).
- **Not checked:** that every video decodes, and that it is identical to the Kaggle copy used in
  R0. The crop step counts frames per video and will show a broken file.
- **Still needed:** the data is on this Windows PC; training is on the faculty machine. For R1 only
  the YouTube originals and the four fake types are needed, about 8.4 GB.
- **Script:** `faculty/01_prepare_data.sh` now accepts this layout as well as the Kaggle one.
- This compression choice is still formally open (O4); nothing else exists locally.

### D031 — 2026-11-25 — R1 settings fixed; R1 code written and tested locally (closes O4, O5, O6, O13, O14)
- **Settings (user, 2026-11-25: "check the code and everything with fixes you proposed"). I read
  this as accepting my recommendations; any of them can still be changed before `03_train.sh full`:**
  - O4: FF++ **c23** for training and test (deviation: the official code and README use raw).
  - O5: **SBI** variant only.
  - O6: loss weights **hm 10, consistency 100** in both runs (paper's values; released SBI config
    says 100 / 100; the released weights' file name says hm10).
  - O13: **100 epochs**, backbone frozen for the first 6, as in the paper and the released config.
  - O14: **R1-fixed = one result-changing fix: frame resampling** (mismatch 24). Noise and blur
    augmentation (mismatch 18) is left for the next round so each effect can be read separately.
- **Code (`faculty/`), all changes applied by `laa_r1.py patch` to a clean copy of commit `a33446a`,
  each by exact-match replacement that stops if the file is not as expected:**
  - both runs: `torch.float64` → `torch.float32`, 30 places in `scripts/train.py` (4),
    `lib/core_function.py` (16), `losses/losses.py` (2), `scripts/test.py` (8) (D029);
  - both runs, bookkeeping only: one `R1_EPOCH` summary line per epoch and a `model_last.pth`
    checkpoint every epoch. Reason: the official rule saves a checkpoint only when validation
    accuracy rises **and** validation loss falls, so "best" can be an early epoch;
  - R1-fixed only, one line in `datasets/face_forensic_sbi.py`: `_reload_data()` assigns to
    `image_paths_r, labels_r, mask_paths_r, ot_props_r` (the attributes `__getitem__` reads).
  - Config, both runs (from `configs/efn4_fpn_sbi_adv.yaml`): task name, data root and landmark
    files, test loader pointed at the FF++ test crops, `begin_epoch: -1`, `resume: False`, no
    checkpoint path, `hm_lmda: 10`. `03_train.sh` stops if the two configs differ in anything but
    the task name. The exact code difference is saved as `control.patch` / `fixed.patch`.
- **Deliberately not changed:** 8 real + 8 pseudo-fake images per step, SAM(Adam), lr 5e-5 ×4 schedule,
  seed 317, 7 loader workers, all augmentation, the best-checkpoint rule, validation on pseudo-fakes,
  frames without landmarks replaced by a random other sample (about 3%, D027), and
  `DATASET.COMPRESSION: c0` (in this code the key only selects the folder layout; with `c23` the
  sampler would treat every frame file as its own video).
- **Local test (this PC, CPU, Windows; Python 3.10.22, torch 2.7.1+cpu, every other package at the
  faculty versions; synthetic images with fabricated 81-point landmarks, tiny settings; `.cuda()`
  made a no-op by a local wrapper). Results are code checks, not model results:**
  - patches apply to checkouts with Unix and with Windows line endings; a second patch is refused;
  - data probe: after `_reload_data()` the control keeps all 24 frame slots, the fixed copy changes
    21 of 24, videos unchanged;
  - both variants trained 2 epochs and exited normally. Epoch 0 is identical to six decimals in
    both (train loss 146.832094, validation loss 93.074021), so the patch and bookkeeping do not
    disturb the shared random stream. Epoch 1 differs (119.33 vs 159.82), so the fix is active;
  - `model_best.pth` and `model_last.pth` load with PyTorch 2.7's default safe loader;
  - `scripts/test.py` (float32) runs with our checkpoints and with the released float64 weights,
    and the summary table is produced. (AUC values on synthetic data are meaningless.)
- **Not testable here:** GPU execution, loading with 7 worker processes, RetinaFace cropping and
  dlib under the new environment, real data. The scripts stage these on the faculty machine:
  `00_setup_env.sh` → `01_prepare_data.sh` → `02_check_port.sh` (released weights vs R0) →
  `03_train.sh smoke` (2 short epochs, timings) → `03_train.sh full` → `04_test_runs.sh`.
- **Bundle for the USB drive:** `to_faculty/` (11 files, 455.3 MB, `MANIFEST.sha256`), built by
  `tools/make_faculty_bundle.py`: the scripts, the dlib file, and the released SBI weights extracted
  from `Pretrained/LAA-Net pretrained.zip` (SHA-256 `fde72661…`, same as R0).
- **Corrections to D030:** `G:` was the user's USB drive, not a disk of this PC; it is unplugged
  now, so the FF++ videos are not on this PC. The 435 MB weights file noted in D025 is no longer
  in `F:\Visual` (removed by the user; the copy inside `Pretrained/` remains).
- **Folder audit (2026-11-25):** official clone clean at `a33446a` (77 tracked files); both Kaggle
  notebooks regenerate byte-identically from `tools/`; `Pretrained/` hashes match D021; `runs/r0`
  complete (D025); `runs/r1_timing` added (summary typed from the user's paste). Not in the folder:
  FF++ videos, the paper's supplement PDF, the R0 crop logs and the timing-test logs (on Kaggle).
  `F:\Visual` is not under version control.
- **Downloads still to be approved for the faculty machine:** RetinaFace weights 96.9 MB (step 01),
  EfficientNet-B4 AdvProp 78 MB (step 03). Earlier approvals covered Kaggle only (D017, D026).

### D032 — 2026-11-25 — Faculty-machine downloads approved (closes O15)
- **Approved by the user ("Yes, approved"):** on the faculty machine, RetinaFace detector weights,
  96.9 MB, github.com/ternaus/retinaface (fetched by the crop script in `01_prepare_data.sh`), and
  EfficientNet-B4 AdvProp weights, 78 MB, github.com/lukemelas/EfficientNet-PyTorch (fetched by the
  official code when `03_train.sh` first builds the model).
- The FF++ videos, the dlib landmark file and the released SBI weights go by USB drive (no download).
- Step 00 also installs Python packages and clones the official code; step 01 fetches the official
  FF++ split lists (text).
- **Status:** all steps are cleared to run. Nothing has run on the faculty machine yet.

### D033 — 2026-11-25 — `F:\Visual` goes under git, pushed to github.com/NetsNuts-24/laa-net
- **User's request:** put the folder under version control and push it to the new repo
  `https://github.com/NetsNuts-24/laa-net.git` (created by the user).
- **Committed:** `CLAUDE.md`, `DECISIONS.md`, `faculty/`, `kaggle/`, `tools/`, `runs/`, `.gitignore`.
- **Not committed (`.gitignore`):** `Pretrained/` (850 MB weights), `to_faculty/` (435 MB, generated),
  `data/`, `weights/`, any `*.pth/*.zip/*.dat/*.mp4`, the paper PDF (not ours to redistribute),
  caches, secrets (`kaggle.json`, `.env`).
- **Official code `LAA-Net/`:** kept as a **git submodule** pinned to `a33446a`
  (https://github.com/10Ring/LAA-Net), not copied in. Why: it records the exact commit, keeps the
  clone unmodified, and does not re-publish code under the authors' academic licence.
  Rejected: copying the files in (licence, and edits would be mixed with ours); ignoring the folder
  (the commit would only be written in prose).

### D034 — 2026-11-25 — One GPU at a time: scripts take a GPU list (amends D029)
- **Why:** the user reports both faculty GPUs are busy with the temporal branch's training: GPU 1
  is free from about 22:00, GPU 0 from about 05:00–06:00 the next morning. The bundle was already
  copied to the machine; no step had been run.
- **Amends D029:** the two R1 runs no longer have to start together or sit on fixed GPUs. They are
  independent processes with the same code base, config, seed and data, so start time and GPU
  index do not affect the comparison. Everything else in D029 and D031 stands.
- **Script changes (`faculty/`):**
  - every script reads `GPUS` (default `0 1`), e.g. `GPUS=1 bash …`; a GPU that does not exist stops it;
  - `01_prepare_data.sh`: with one GPU the crop jobs run one after the other, short ones first;
  - `common.sh` test queue and `03_train.sh smoke`: sequential on one GPU, side by side on two;
  - `03_train.sh <smoke|full> [control|fixed|both]`: one variant can be started alone;
    `full both` on a single GPU is refused; starting a run twice is refused;
  - `02_check_port.sh` now stops with an error if any AUC is more than 1.0 from R0 (D022). The
    limit 1.0 is my choice, not derived from anything;
  - new `run_night.sh`: prepare data → port check → 2-epoch trial of R1-control → full R1-control,
    stopping at the first failure. **Trade-off:** it replaces my reading of each log between steps
    with these automatic checks; the data-preparation log is not judged automatically. Worst case
    is one wasted control run;
  - `00_setup_env.sh`: `GPUS=none` skips the GPU speed check; the check no longer fails the setup
    when a card is full; it asserts the PyTorch build lists `sm_120` (RTX 50-series);
  - `set +u` around conda activation in all scripts (conda's own hooks can fail under `set -u`).
- **Tested here (simulation: stub `conda`, `nvidia-smi` and GPU jobs; real bash flow and real helper
  tool):** one GPU trial of both variants runs them one after the other on the given GPU; two GPUs
  side by side; `full both` on one GPU refused; control on GPU 1 then fixed on GPU 0 started
  separately, with the config-identity check; second start refused; unknown GPU refused; port check
  sequential and parallel; the 1.0 limit passes close numbers and stops on a far one.
  `run_night.sh` and `01_prepare_data.sh` themselves could not be simulated (they need the real
  crop tools); they are syntax-checked only.
- **Bundle rebuilt:** `to_faculty/` now has 12 files; the two large files are unchanged, so only the
  small ones (`*.sh`, `laa_r1.py`, `README.txt`, `MANIFEST.sha256`) need copying again.
- **Not committed yet:** these changes are in the working tree only (repo from D033). A leftover
  one-off helper `tools/_edit_gpus.py` should be deleted, not committed.

---

## Open

| ID | Question | Options | Recommendation |
|---|---|---|---|
| (none) | All questions for R1 are decided. | | |

Closed so far: O1 (→D005), O2 (→D008, D013), O7 (→D014), O8 (→D012), O3 (→D016, D017, result in D022), O9 (→D024, result in D027), O11 (→D028), O10 and O12 (→D029), O4, O5, O6, O13, O14 (→D031), O15 (→D032).
