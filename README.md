# Vision Assist

An AI-powered voice+vision Lost & Found assistant — FastAPI backend, React/Vite
frontend, communicating over REST + WebSockets.

```
+-----------------------------------------------------------------------------------+
|                                  REACT FRONTEND                                   |
|   (Vite + Tailwind CSS / WebCam Canvas Overlay / WebSocket Audio & Vision)        |
+-----------------------------------------------------------------------------------+
                                   │            ▲
                      REST APIs    │            │  WebSockets
                        (Auth)     │            │  (Live Frames & Bounding Boxes)
                                   ▼            │
+-----------------------------------------------------------------------------------+
|                                 FASTAPI BACKEND                                   |
|  /health   │   /ws/vision   │   /ws/audio   │   /api/v1/auth/login               |
+-----------------------------------------------------------------------------------+
       │                   │                      │
       ▼                   ▼                      ▼
 +-----------+     +---------------+      +---------------------+
 | PostgreSQL|     |  YOLO / YOLOE |      | Whisper STT + Ollama |
 | (Users/Log)     | (Vision Engine)      | (Audio + Intent)     |
 +-----------+     +---------------+      +---------------------+
```

Qdrant is also provisioned (`docker-compose.yml`) as the vector store for
upcoming semantic item search.

## Getting Started

```bash
git clone https://github.com/admirableanujj/vision-assist-v1.git
cd vision-assist-v1
cp .env.example .env   # fill in OPENAI_API_KEY if using cloud STT


mkdir -p secrets
printf '%s' 'devpassword123' > secrets/postgres_password.txt
printf '%s' 'unused-for-now' > secrets/qdrant_api_key.txt

#for Windows System
# New-Item -ItemType Directory -Path "secrets" -Force
#Set-Content -Path "secrets/postgres_password.txt" -Value "devpassword123" -NoNewline
#Set-Content -Path "secrets/qdrant_api_key.txt" -Value "unused-for-now" -NoNewline


# --no-deps skips ollama/qdrant — add them back if you need the voice/LLM path
docker-compose up -d --build --no-deps backend postgres frontend
```

Then open `http://localhost:3000`. For a verified setup walkthrough — local
or GitHub Codespaces, including test credentials — see
**`docs/TESTING_LOCAL_AND_CODESPACES.md`**.

## Vision: YOLO / YOLOE

Object detection defaults to YOLO26n (COCO's 80 classes). YOLOE, an
open-vocabulary alternative, can detect additional classes (e.g. `keys`,
`wallet`, `sunglasses` — none of which are COCO classes) at a lower accuracy,
selected via `VISION_MODEL_TYPE=yoloe` in `.env` (container restart required,
no live UI toggle yet). See **`docs/YOLO_VS_YOLOE_GUIDE.md`** for the full
comparison and switching instructions, and
**`docs/FINE_TUNING_YOLO_GUIDE.md`** for the (planning-only) plan to close
that class gap with a real fine-tuned model instead of zero-shot detection.

## Architecture

See **`CLAUDE.md`** for the repository layout, request flow, and key design
patterns.
