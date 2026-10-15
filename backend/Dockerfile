# backend/Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

# Install system runtime & build dependencies
# NOTE: 'libgl1' replaces deprecated 'libgl1-mesa-glx'; 'git' is added for OpenAI CLIP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy backend application source code
COPY . /workspace/

# Pre-download YOLO26n's weights at build time rather than letting the first
# real request pay for it.
RUN python -c "import torch; _old_load = torch.load; torch.load = lambda *args, **kwargs: _old_load(*args, **{**kwargs, 'weights_only': False}); from ultralytics import YOLO; YOLO('yolo26n.pt')"

# Pre-download YOLOE's weights AND exercise set_classes() at build time, not
# just load the checkpoint. set_classes() is what pulls in the CLIP text
# encoder that requirements.txt already pins — doing this at build time means
# it resolves from already-installed packages instead of Ultralytics silently
# pip-installing CLIP into a live, already-running process on first YOLOE use
# (confirmed, in an earlier version of this app, to cause a real SIGSEGV crash
# — the live install doesn't take effect without restarting the interpreter).
RUN python -c "import torch; _old_load = torch.load; torch.load = lambda *args, **kwargs: _old_load(*args, **{**kwargs, 'weights_only': False}); from ultralytics import YOLOE; m = YOLOE('yoloe-26n-seg.pt'); m.set_classes(['person'])"

EXPOSE 8000

# Start FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]