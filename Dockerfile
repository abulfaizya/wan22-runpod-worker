
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=180

# System deps that many wheels need
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential pkg-config \
    git ffmpeg \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -V

WORKDIR /workspace

# Copy requirements first so CI shows the failing package clearly
COPY requirements.txt /workspace/requirements.txt

# Auto-swap GUI OpenCV → headless in CI containers
RUN if grep -qi "^opencv-python" requirements.txt; then \
      sed -i 's/^opencv-python.*/opencv-python-headless/gI' requirements.txt ; \
    fi

# Install CUDA-matched PyTorch from official wheels (don’t pin torch in requirements.txt)
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Install the rest (verbose so the failing package is obvious)
RUN python3 -m pip install --prefer-binary -v -r requirements.txt

# Bring in your code
COPY . /workspace

# Runtime helpers
RUN python3 -m pip install runpod boto3 Pillow "huggingface_hub[cli]"

CMD ["python3", "/workspace/handler.py"]
