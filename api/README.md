# LTX-2 HTTP API

A thin FastAPI wrapper around the LTX-2 pipelines. Swagger UI is auto-generated
from the Pydantic schemas.

## Run locally

```bash
uv sync --frozen
uv pip install -r api/requirements.txt

export LTX2_MODELS_ROOT=/abs/path/to/models
export LTX2_OUTPUT_DIR=./outputs
export LTX2_PIPELINE=TI2VidTwoStagesPipeline   # or DistilledPipeline, etc.

uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

## Endpoints

| Method | Path                          | Purpose                         |
|--------|-------------------------------|---------------------------------|
| GET    | `/health`                     | Liveness probe                  |
| POST   | `/v1/generate`                | Submit a generation job         |
| GET    | `/v1/jobs/{id}`               | Poll job status                 |
| GET    | `/v1/jobs/{id}/video`         | Download the resulting MP4      |
| GET    | `/docs`                       | Swagger UI                      |
| GET    | `/openapi.json`               | OpenAPI 3 spec                  |

## Environment variables

| Var                    | Default                                                | Purpose                                        |
|------------------------|--------------------------------------------------------|------------------------------------------------|
| `LTX2_MODELS_ROOT`     | `./models`                                             | Base directory for checkpoints                 |
| `LTX2_CHECKPOINT`      | `${MODELS_ROOT}/ltx-2.3-22b-distilled-1.1.safetensors` | Main DiT checkpoint                            |
| `LTX2_SPATIAL_UPSAMPLER` | `${MODELS_ROOT}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | Spatial upscaler                          |
| `LTX2_DISTILLED_LORA`  | `${MODELS_ROOT}/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | Distilled LoRA                          |
| `LTX2_GEMMA_ROOT`      | `${MODELS_ROOT}/gemma-3-12b-it-qat-q4_0-unquantized`   | Gemma text encoder                             |
| `LTX2_OUTPUT_DIR`      | `./outputs`                                            | Where MP4s are written                         |
| `LTX2_OFFLOAD_MODE`    | `NONE`                                                 | `NONE`, `BLOCK`, `LAYER` (lower VRAM = slower) |
| `LTX2_QUANTIZATION`    | unset                                                  | `fp8-cast` or `fp8-scaled-mm`                  |
| `LTX2_PIPELINE`        | `TI2VidTwoStagesPipeline`                              | Pipeline class name                            |
