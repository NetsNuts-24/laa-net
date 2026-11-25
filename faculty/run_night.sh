#!/usr/bin/env bash
# Faculty machine, one free GPU, nobody watching: do the steps in order and stop at the first problem.
#   1. data preparation (crops + landmarks)            01_prepare_data.sh
#   2. port check; stops if an AUC is more than 1.0 away from our Kaggle result R0   02_check_port.sh
#   3. 2-epoch trial of R1-control                      03_train.sh smoke control
#   4. full 100-epoch run of R1-control, in background  03_train.sh full control
# Start it like this (GPU 1 here) and leave it:
#   GPUS=1 nohup bash run_night.sh ~/FaceForensics > night.log 2>&1 &
#   tail -f night.log           # Ctrl+C stops only the watching
# R1-fixed is started later on whichever GPU is free:
#   GPUS=0 bash 03_train.sh smoke fixed 2>&1 | tee train_smoke_fixed.log
#   GPUS=0 bash 03_train.sh full fixed  2>&1 | tee train_full_fixed.log
set -euo pipefail
: "${GPUS:?say which GPU to use, for example GPUS=1}"
export GPUS
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FF="${1:?give the FF++ folder (the one that contains original_sequences and manipulated_sequences)}"
step() { echo; echo "######## $(date '+%Y-%m-%d %H:%M:%S')  $*  ########"; }

step "1/4 data preparation"
bash "$HERE/01_prepare_data.sh" "$FF" 2>&1 | tee "$HERE/prepare_data.log"
step "2/4 port check against R0"
bash "$HERE/02_check_port.sh" 2>&1 | tee "$HERE/check_port.log"
step "3/4 trial run of R1-control (2 short epochs)"
bash "$HERE/03_train.sh" smoke control 2>&1 | tee "$HERE/train_smoke_control.log"
step "4/4 full run of R1-control"
bash "$HERE/03_train.sh" full control 2>&1 | tee "$HERE/train_full_control.log"
step "NIGHT CHAIN DONE: R1-control is training on GPU $GPUS. Check with: bash status.sh"
