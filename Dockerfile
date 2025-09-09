FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# OS deps
RUN apt-get update && apt-get install -y python3-pip git ffmpeg && rm -rf /var/lib/apt/lists/*

# Python
RUN python3 -m pip install --upgrade pip

# Clone WAN 2.2
WORKDIR /workspace
RUN git clone https://github.com/Wan-Video/Wan2.2.git
WORKDIR /workspace/Wan2.2

# Install WAN deps (WAN repo requires torch>=2.4; it will be pulled via requirements.txt)
RUN pip install -r requirements.txt

# Serverless + storage helpers
RUN pip install runpod boto3 Pillow "huggingface_hub[cli]"

# (Optional) Pre-download TI2V-5B weights. You can also mount a volume or download at runtime.
# Uncomment the next line if you want to bake weights into the image (large!).
# RUN huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

# Worker code
COPY handler.py /workspace/handler.py

CMD ["python3", "/workspace/handler.py"]
