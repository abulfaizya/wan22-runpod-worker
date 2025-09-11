FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=180

# System deps often needed by Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential pkg-config \
    git ffmpeg \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Pip basics + packaging (your log complained it was missing)
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

WORKDIR /workspace

# Copy requirements first so failure is obvious in Actions logs
COPY requirements.txt /workspace/requirements.txt

# Swap GUI OpenCV → headless (prevents CI failures)
RUN if grep -qi "^opencv-python" requirements.txt; then \
      sed -i 's/^opencv-python.*/opencv-python-headless==4.9.0.80/gI' requirements.txt ; \
    fi

# Install CUDA 12.1-matched PyTorch stack (prebuilt wheels)
# IMPORTANT: Do NOT list torch/vision/audio in requirements.txt
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Install the rest (verbose so the failing pkg is obvious if any)
RUN python3 -m pip install --prefer-binary -v -r requirements.txt

# Bring in the rest of your code
COPY . /workspace

# Safe runtime helpers
RUN python3 -m pip install runpod boto3 Pillow "huggingface_hub[cli]"

CMD ["python3", "/workspace/handler.py"]

