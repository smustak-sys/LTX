"""Environment-based configuration for the LTX-2 API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    checkpoint_path: str
    spatial_upsampler_path: str
    distilled_lora_path: str
    gemma_root: str
    output_dir: Path
    offload_mode: str
    quantization: str | None
    pipeline_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        models_root = os.environ.get("LTX2_MODELS_ROOT", "./models")
        return cls(
            checkpoint_path=os.environ.get(
                "LTX2_CHECKPOINT", f"{models_root}/ltx-2.3-22b-distilled-1.1.safetensors"
            ),
            spatial_upsampler_path=os.environ.get(
                "LTX2_SPATIAL_UPSAMPLER", f"{models_root}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
            ),
            distilled_lora_path=os.environ.get(
                "LTX2_DISTILLED_LORA", f"{models_root}/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
            ),
            gemma_root=os.environ.get("LTX2_GEMMA_ROOT", f"{models_root}/gemma-3-12b-it-qat-q4_0-unquantized"),
            output_dir=Path(os.environ.get("LTX2_OUTPUT_DIR", "./outputs")),
            offload_mode=os.environ.get("LTX2_OFFLOAD_MODE", "NONE"),
            quantization=os.environ.get("LTX2_QUANTIZATION") or None,
            pipeline_name=os.environ.get("LTX2_PIPELINE", "TI2VidTwoStagesPipeline"),
        )


settings = Settings.from_env()
settings.output_dir.mkdir(parents=True, exist_ok=True)
