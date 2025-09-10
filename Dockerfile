# CUDA 12.1 runtime (works well with prebuilt PyTorch wheels)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps commonly required by wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential pkg-config \
    git ffmpeg \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY . /workspace

# ---- Install CUDA-matched PyTorch first (prebuilt, no compiling) ----
# Remove torch/torchvision/torchaudio from requirements.txt (next step),
# or keep them UNPINNED if you really must leave them.
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# ---- Make requirements more CI-friendly ----
# Swap GUI OpenCV -> headless (prevents lib errors on servers)
RUN if grep -qi "^opencv-python" requirements.txt; then \
      sed -i 's/^opencv-python.*/opencv-python-headless/gI' requirements.txt ; \
    fi
# If you used any of these and don’t absolutely need them, comment them out in requirements.txt:
# xformers / flash-attn / bitsandbytes / triton

# ---- Install the rest of your deps ----
# First try fast (no deps), fallback with full resolver if needed.
RUN python3 -m pip install --no-deps -r requirements.txt || \
    (echo "Retrying with deps…" && python3 -m pip install -r requirements.txt)

# Worker runtime deps (safe)
RUN python3 -m pip install runpod boto3 Pillow "huggingface_hub[cli]"

# Start your worker
CMD ["python3", "/workspace/handler.py"]
