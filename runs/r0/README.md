# R0 run record (D022)

Run: authors' released SBI weights + official test code on the FF++ c23 test split.
Kaggle notebook `yashwanthvemuri/laa-net1` (private), script version 373185070, run 2026-11-24.

The files here were copied on 2026-11-25 from the notebook's Output tab **as page text read
through the browser**, not as byte-for-byte downloads (the browser download kept failing):

- `split_test.json`: rebuilt from the 70 pairs; SHA-256 equals the Kaggle file's text
  (`886f5a0da623c25820692e0d8dc33d197ddb1db527a7f1cfcb9bcbca60fe4f40`).
- `times_full.json`, `crop_counts_test.json`, `env_freeze.txt`: same text as on Kaggle.
- `r0_full_*.log`: same lines and numbers; runs of spaces are collapsed and the progress-bar
  line is as the browser showed it. All digits of the metric line are kept.

Not copied: the five `crop_test_*.log` files (about 29 KB each, per-video frame counts) and the
folders `LAA-Net`, `datasets_df`, `env`, `weights`. They remain in the Kaggle notebook output.
