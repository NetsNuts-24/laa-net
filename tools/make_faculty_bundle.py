"""Build F:\\Visual\\to_faculty: everything (except the FF++ videos) that goes to the faculty machine.

  python tools/make_faculty_bundle.py

Copies the scripts from faculty/, the dlib 81-landmark file, and extracts the authors' released SBI
weights from Pretrained/LAA-Net pretrained.zip. Writes MANIFEST.sha256 (checked on the other side with
`sha256sum -c MANIFEST.sha256`) and README.txt. Safe to run again after a script changes.
"""
import hashlib
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'to_faculty')
SBI_SHA = 'fde726610e778045f65c3fbe839bc463c4d3a9970f72ca109d5911917305c991'   # D021, D022
DLIB_SHA = '8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869'  # D021

README = """LAA-Net visual branch: files for the faculty machine (2 x RTX 5090)
====================================================================

Copy this whole folder to the faculty machine, for example to ~/laa_bundle.
Copy the FF++ c23 videos too (only original_sequences/youtube and the four folders Deepfakes,
Face2Face, FaceSwap, NeuralTextures under manipulated_sequences are needed for R1; about 8.4 GB).

On the faculty machine, in a terminal:

  cd ~/laa_bundle
  sha256sum -c MANIFEST.sha256                               # every line must say OK
  bash 00_setup_env.sh              2>&1 | tee setup_env.log      # environment; must end ENVIRONMENT OK
  bash 01_prepare_data.sh /path/to/FaceForensics 2>&1 | tee prepare_data.log   # crops + landmarks (hours)
  bash 02_check_port.sh             2>&1 | tee check_port.log     # released weights; compare with R0
  bash 03_train.sh smoke            2>&1 | tee train_smoke.log    # 2 short epochs of both runs
  bash 03_train.sh full             2>&1 | tee train_full.log     # 100 epochs of both runs, in background
  bash status.sh                                                  # progress, any time
  bash 04_test_runs.sh              2>&1 | tee test_runs.log      # after both runs have finished

Send each .log file back after its step and wait for the answer before the next step.

WHEN A GPU IS BUSY WITH OTHER WORK
Put GPUS=<free GPU number> in front of a command and it uses only that GPU, for example
  GPUS=1 bash 02_check_port.sh 2>&1 | tee check_port.log
Step 00 while both GPUs are busy:      GPUS=none bash 00_setup_env.sh 2>&1 | tee setup_env.log
One free GPU and nobody watching (does steps 01, 02, a trial and the full run of R1-control, and
stops at the first problem):
  GPUS=1 nohup bash run_night.sh /path/to/FaceForensics > night.log 2>&1 &
  tail -f night.log                    # Ctrl+C stops only the watching
The second run, later, on whichever GPU is free:
  GPUS=0 bash 03_train.sh smoke fixed 2>&1 | tee train_smoke_fixed.log
  GPUS=0 bash 03_train.sh full fixed  2>&1 | tee train_full_fixed.log
The two runs do not have to start together or use the same GPU; they are independent.
/path/to/FaceForensics is the folder that contains original_sequences/ and manipulated_sequences/.
Everything is written under ~/laa_visual. Nothing outside it is changed, except a conda
environment called laa310 and PyTorch's download cache in ~/.cache/torch.

Downloads the scripts make (each approved by the user before the step is run):
  00  Python packages (PyTorch 2.7.1 for CUDA 12.8 and others) and the official LAA-Net code
  01  RetinaFace detector weights, 96.9 MB; the official FF++ split lists (small text files)
  03  EfficientNet-B4 AdvProp weights, 78 MB
"""


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(os.path.join(OUT, 'weights'), exist_ok=True)
    fac = os.path.join(ROOT, 'faculty')
    for stale in os.listdir(OUT):                       # drop scripts that no longer exist in faculty/
        if stale.endswith(('.sh', '.py')) and not os.path.exists(os.path.join(fac, stale)):
            os.remove(os.path.join(OUT, stale))
    for name in sorted(os.listdir(fac)):
        data = open(os.path.join(fac, name), 'rb').read()
        assert b'\r' not in data, f'{name} has Windows line endings'
        with open(os.path.join(OUT, name), 'wb') as f:
            f.write(data)
    dlib_dst = os.path.join(OUT, 'shape_predictor_81_face_landmarks.dat')
    if not os.path.isfile(dlib_dst):
        shutil.copyfile(os.path.join(ROOT, 'Pretrained', 'shape_predictor_81_face_landmarks.dat'), dlib_dst)
    assert sha(dlib_dst) == DLIB_SHA
    w_dst = os.path.join(OUT, 'weights', 'released_sbi_model_best.pth')
    if not os.path.isfile(w_dst):
        z = zipfile.ZipFile(os.path.join(ROOT, 'Pretrained', 'LAA-Net pretrained.zip'))
        entry = [i for i in z.infolist() if i.filename.endswith('1e7_boost500_UnFZ_model_best.pth')]
        assert len(entry) == 1, [i.filename for i in z.infolist()]
        with z.open(entry[0]) as src, open(w_dst, 'wb') as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
    assert sha(w_dst) == SBI_SHA, 'released SBI weights differ from the file used in R0'
    with open(os.path.join(OUT, 'README.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(README)
    lines = []
    for base, _, files in os.walk(OUT):
        for name in sorted(files):
            if name == 'MANIFEST.sha256':
                continue
            p = os.path.join(base, name)
            rel = os.path.relpath(p, OUT).replace(os.sep, '/')
            lines.append((rel, sha(p), os.path.getsize(p)))
    lines.sort()
    with open(os.path.join(OUT, 'MANIFEST.sha256'), 'w', encoding='utf-8', newline='\n') as f:
        for rel, h, _ in lines:
            f.write(f'{h}  {rel}\n')
    for rel, h, size in lines:
        print(f'{size:>12,}  {h[:12]}  {rel}')
    print('bundle:', OUT, '|', len(lines), 'files |', f'{sum(s for _, _, s in lines) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
