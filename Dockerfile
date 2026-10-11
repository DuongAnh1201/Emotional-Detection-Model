FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV HF_HOME=/app/.cache/huggingface \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Bake the Hugging Face models (~350 MB) into the image so it runs offline.
# Model IDs must match FaceDetection.py. Kept before COPY so code edits don't re-download.
RUN python -c "\
from huggingface_hub import hf_hub_download; \
from transformers import ViTImageProcessor, ViTModel; \
hf_hub_download(repo_id='AdamCodd/YOLOv11n-face-detection', filename='model.pt'); \
ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'); \
ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')"
ENV HF_HUB_OFFLINE=1

COPY . .
RUN uv sync --frozen --no-dev
ENTRYPOINT ["python", "predict.py"]
CMD ["test_face.jpg"]