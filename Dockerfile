
# CUDA 12.1 runtime (works with prebuilt PyTorch wheels)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120

# System deps commonly needed by Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential pkg-config \
    git ffmpeg \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# --- Copy requirements first for better caching + clearer errors ---
COPY requirements.txt /workspace/requirements.txt

# (Optional) auto-swap GUI OpenCV -> headless for CI
RUN if grep -qi "^opencv-python" requirements.txt; then \
      sed -i 's/^opencv-python.*/opencv-python-headless/gI' requirements.txt ; \
    fi

# Install CUDA-matched PyTorch first (prebuilt, no compiling)
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Try to install your deps (verbose so you see the failing package)
RUN python3 -m pip install --prefer-binary -v -r requirements.txt

# Now bring in the rest of your code
COPY . /workspace

# Worker runtime deps that are safe
RUN python3 -m pip install runpod boto3 Pillow "huggingface_hub[cli]"

CMD ["python3", "/workspace/handler.py"]
