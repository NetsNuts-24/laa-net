#!/usr/bin/env bash
# Faculty machine (2 x RTX 5090): build the environment for LAA-Net and check the GPUs.
# Downloads Python packages and the official code only. No dataset, no model weights.
# Run:  bash 00_setup_env.sh 2>&1 | tee setup_env.log
# Decision log: F:\Visual\DECISIONS.md D028.
set -euo pipefail

WORK="${WORK:-$HOME/laa_visual}"
ENV_NAME="${ENV_NAME:-laa310}"
COMMIT=a33446ab94b3bc0d176a879d43e0700adc52c817

echo "== 1. machine =="
echo "cores: $(nproc)"
free -g | sed -n '1,2p'
df -h "$HOME" | sed -n '1,2p'
FREE_GB=$(df -BG --output=avail "$HOME" | tail -n 1 | tr -dc '0-9')
echo "free disk in home: ${FREE_GB} GB (FF++ c23 videos need about 18 GB, crops about 30 GB more)"
if [ "$FREE_GB" -lt 80 ]; then echo "WARNING: less than 80 GB free; tell Claude before copying data."; fi
if curl -sI --max-time 15 https://github.com >/dev/null; then echo "internet: ok"; else echo "ERROR: no internet to github.com"; exit 1; fi
command -v conda >/dev/null || { echo "ERROR: conda not found"; exit 1; }
conda --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

echo "== 2. environment ${ENV_NAME} (Python 3.10) =="
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "environment exists, reusing it"
else
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.10 pip
fi
conda activate "$ENV_NAME"
python --version

echo "== 3. packages =="
# PyTorch: oldest release line with RTX 50-series support (CUDA 12.8 build). Official code pins 1.8.0.
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# Everything else at the versions that worked in R0, except OpenCV:
# 4.5.1.48 has no Python 3.10 build, 4.5.5.64 is the nearest that has.
pip install -q albumentations==1.1.0 python-box==7.1.1 imgaug==0.4.0 imutils==0.5.4 numpy==1.23.3 \
  opencv-python-headless==4.5.5.64 pandas==1.3.5 Pillow==9.3.0 scikit-image==0.19.3 scipy==1.9.3 \
  simplejson tensorboardX==2.5.1 tqdm PyYAML scikit-learn==1.3.2 matplotlib==3.7.5 \
  retinaface-pytorch==0.0.7 dlib-bin==19.24.6
# Some packages pull in the non-headless OpenCV as well (seen in R0). Keep one copy only.
pip uninstall -y -q opencv-python 2>/dev/null || true
pip install -q --force-reinstall --no-deps opencv-python-headless==4.5.5.64
pip install -q --force-reinstall --no-deps numpy==1.23.3

echo "== 4. official code =="
mkdir -p "$WORK"
if [ ! -d "$WORK/LAA-Net/.git" ]; then git clone -q https://github.com/10Ring/LAA-Net "$WORK/LAA-Net"; fi
git -C "$WORK/LAA-Net" checkout -q "$COMMIT"
git -C "$WORK/LAA-Net" log -1 --format="%H %ad"

echo "== 5. checks =="
pip freeze > "$WORK/env_freeze_faculty.txt"
grep -i -E "^(torch|torchvision|numpy|opencv|albumentations|dlib)" "$WORK/env_freeze_faculty.txt"
python - <<'PY'
import time, numpy, cv2, dlib, albumentations, imgaug, torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| numpy", numpy.__version__,
      "| cv2", cv2.__version__, "| dlib", dlib.__version__, "| albumentations", albumentations.__version__)
assert torch.cuda.is_available(), "PyTorch does not see a GPU"
print("GPUs:", torch.cuda.device_count(), "| built for:", torch.cuda.get_arch_list())
torch.from_numpy(numpy.zeros((2, 2), dtype=numpy.float32))
for i in range(torch.cuda.device_count()):
    dev = f"cuda:{i}"
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.SiLU(),
                              torch.nn.Conv2d(64, 64, 3, padding=1, groups=64), torch.nn.Conv2d(64, 1, 1)).to(dev)
    x = torch.randn(16, 3, 384, 384, device=dev)
    for dt in (torch.float32, torch.float64):
        net.to(dt); xx = x.to(dt)
        net(xx).mean().backward(); torch.cuda.synchronize(dev)
        t = time.time()
        for _ in range(10):
            net(xx).mean().backward()
        torch.cuda.synchronize(dev)
        print(dev, torch.cuda.get_device_name(i), str(dt), f"{(time.time() - t) / 10 * 1000:.1f} ms per pass")
print("ENVIRONMENT OK")
PY
echo "Done. Send setup_env.log (or everything printed above) back."
