# LTX-2 API image. Built on the official PyTorch CUDA image so torch + CUDA
# are already wired up; we layer uv on top to install the workspace.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# uv for the workspace install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Workspace metadata first so the dependency layer caches.
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --frozen --no-dev

# Application code
COPY api ./api
RUN uv pip install --system -r api/requirements.txt

# Models and outputs are mounted at runtime (large; not baked into the image).
ENV LTX2_MODELS_ROOT=/models \
    LTX2_OUTPUT_DIR=/outputs \
    LTX2_OFFLOAD_MODE=NONE \
    LTX2_PIPELINE=TI2VidTwoStagesPipeline

EXPOSE 8000

# Use the workspace venv python created by `uv sync`.
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
