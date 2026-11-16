# backend/Dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

# git is required for the OpenAI CLIP dependency (installed via git+https in requirements.txt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch/torchvision first — ultralytics and sentence-transformers
# both pull torch transitively, and without this, pip resolves the default
# PyPI wheel (bundles full CUDA/cuDNN, hundreds of MB to ~1GB larger) even
# though this container never has a GPU.
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download both model checkpoints at build time instead of paying for it
# (and, for YOLOE, triggering a live pip-install of its CLIP dependency) on
# the first real request. Saved outside /workspace — docker-compose.yml
# bind-mounts ./backend over /workspace for hot-reload, which would hide
# anything baked in there. Placed before COPY so this layer only reruns when
# requirements.txt changes, not on every source edit.
RUN mkdir -p /opt/weights
RUN python -c "from ultralytics import YOLO; YOLO('/opt/weights/yolo26n.pt')"
RUN python -c "from ultralytics import YOLOE; m = YOLOE('/opt/weights/yoloe-26n-seg.pt'); m.set_classes(['person'])"

# Fine-tuned checkpoint (project's own 48-class lostfound scheme) - not a
# public Ultralytics release, so it can't be fetched the way the two weights
# above are; copied from this repo's own build context instead. This is
# "run 2" from the fine-tuning sweep (finetuning/kaggle/RUN_LOG.md) -
# coco-sample-size=5000, chosen over the original baseline because it
# improved the catastrophic-forgetting check on every one of the 20 kept
# COCO classes for only a marginal novel-class cost.
COPY weights/lostfound_yolo26n_best.pt /opt/weights/lostfound_yolo26n_best.pt

# Copy backend application source code
COPY . /workspace/

EXPOSE 8000

# Start FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]