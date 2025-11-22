#!/usr/bin/env bash

set -e

echo "---------------------------------------------"
echo " Nerfstudio + Viewer setup for Apple Silicon "
echo " (M1 / M2 / M3 / M4 Macs)                    "
echo "---------------------------------------------"

ENV_NAME="nsenv"

# 0. Check python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Install Python 3 or Xcode CLT first."
    exit 1
fi

PYVER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "[INFO] Detected python3 version: $PYVER"

# 1. Create virtualenv
if [ -d "$ENV_NAME" ]; then
    echo "[INFO] Virtualenv '$ENV_NAME' already exists, reusing it."
else
    echo "[INFO] Creating virtualenv '$ENV_NAME'..."
    python3 -m venv "$ENV_NAME"
fi

# 2. Activate env (for this script)
# shellcheck source=/dev/null
source "$ENV_NAME/bin/activate"

echo "[INFO] Using Python from: $(which python)"
echo "[INFO] Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch (Metal/MPS backend is auto-used on Apple Silicon)
echo "[INFO] Installing PyTorch (this may take a bit)..."
pip install torch torchvision torchaudio

# 4. Install Nerfstudio
echo "[INFO] Installing Nerfstudio..."
pip install nerfstudio

echo "[INFO] Running ns-install-cli (to setup CLI entrypoints)..."
ns-install-cli || echo "[WARN] ns-install-cli returned non-zero exit (may still be fine)."

# 5. Extra deps for viewers / splats
echo "[INFO] Installing extra viewer dependencies..."
pip install plyfile opencv-python-headless imageio tqdm rich

# 6. Optional: gsplat for fast Gaussian splat viewing
echo "[INFO] Installing gsplat (optional but recommended)..."
pip install gsplat || echo "[WARN] gsplat install failed, continuing anyway."

echo "---------------------------------------------"
echo " Installation complete!"
echo
echo "To use this environment in a new terminal:"
echo "  cd $(pwd)"
echo "  source $ENV_NAME/bin/activate"
echo
echo "To test Nerfstudio:"
echo "  ns-viewer --help"
echo
echo "To run Shrish's nerfstudio viewer script (example):"
echo "  python viewer_nerfstudio.py --ply assets/pointclouds/pilot_plant_devices.ply"
echo
echo "---------------------------------------------"
