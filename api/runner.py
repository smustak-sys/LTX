"""Pipeline runner. Lazily imports ltx_pipelines so the API can boot without GPU/models."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

from .config import settings
from .schemas import GenerateRequest

logger = logging.getLogger(__name__)

_pipeline: Any = None
_pipeline_lock = Lock()


def _build_pipeline() -> Any:
    """Construct the configured pipeline. Imported lazily because the LTX stack
    pulls in torch/CUDA and is expensive to load."""
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.utils.types import OffloadMode

    pipeline_module = __import__(
        f"ltx_pipelines.{_pipeline_module_name(settings.pipeline_name)}",
        fromlist=[settings.pipeline_name],
    )
    pipeline_cls = getattr(pipeline_module, settings.pipeline_name)

    quantization = None
    if settings.quantization == "fp8-cast":
        quantization = QuantizationPolicy.fp8_cast()
    elif settings.quantization == "fp8-scaled-mm":
        quantization = QuantizationPolicy.fp8_scaled_mm()

    valid_modes = {m.name for m in OffloadMode}
    if settings.offload_mode not in valid_modes:
        msg = f"LTX2_OFFLOAD_MODE={settings.offload_mode!r} invalid. Must be one of {sorted(valid_modes)}."
        raise ValueError(msg)

    offload_mode = OffloadMode[settings.offload_mode]

    if settings.pipeline_name == "DistilledPipeline":
        return pipeline_cls(
            distilled_checkpoint_path=settings.checkpoint_path,
            gemma_root=settings.gemma_root,
            spatial_upsampler_path=settings.spatial_upsampler_path,
            loras=[],
            quantization=quantization,
            offload_mode=offload_mode,
        )

    distilled_lora = [LoraPathStrengthAndSDOps(path=settings.distilled_lora_path, strength=1.0, sd_ops=())]

    return pipeline_cls(
        checkpoint_path=settings.checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=settings.spatial_upsampler_path,
        gemma_root=settings.gemma_root,
        loras=(),
        quantization=quantization,
        offload_mode=offload_mode,
    )


def _pipeline_module_name(class_name: str) -> str:
    mapping = {
        "TI2VidTwoStagesPipeline": "ti2vid_two_stages",
        "TI2VidTwoStagesHQPipeline": "ti2vid_two_stages_hq",
        "TI2VidOneStagePipeline": "ti2vid_one_stage",
        "DistilledPipeline": "distilled",
        "ICLoraPipeline": "ic_lora",
        "KeyframeInterpolationPipeline": "keyframe_interpolation",
        "A2VidPipelineTwoStage": "a2vid_two_stage",
        "RetakePipeline": "retake",
        "HDRICLoraPipeline": "hdr_ic_lora",
    }
    if class_name not in mapping:
        msg = f"Unknown pipeline class: {class_name}"
        raise ValueError(msg)
    return mapping[class_name]


def get_pipeline() -> Any:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            logger.info("Building pipeline %s …", settings.pipeline_name)
            _pipeline = _build_pipeline()
        return _pipeline


def run_generation(req: GenerateRequest, output_path: Path) -> None:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    pipeline = get_pipeline()
    tiling_config = TilingConfig.default()
    images = [
        ImageConditioningInput(path=img.path, frame_idx=img.frame_idx, strength=img.strength, crf=img.crf)
        for img in req.images
        if img.path and Path(img.path).is_file()
    ]
    if req.images and not images:
        msg = (
            f"images provided but none of the paths exist on disk: "
            f"{[i.path for i in req.images]}. Use absolute paths inside the pod."
        )
        raise FileNotFoundError(msg)

    if settings.pipeline_name == "DistilledPipeline":
        video, audio = pipeline(
            prompt=req.prompt,
            seed=req.seed,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            frame_rate=req.frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=req.enhance_prompt,
        )
    else:
        video, audio = pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            frame_rate=req.frame_rate,
            num_inference_steps=req.num_inference_steps,
            video_guider_params=MultiModalGuiderParams(cfg_scale=req.video_cfg_scale),
            audio_guider_params=MultiModalGuiderParams(cfg_scale=req.audio_cfg_scale),
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=req.enhance_prompt,
            max_batch_size=req.max_batch_size,
        )

    encode_video(
        video=video,
        fps=req.frame_rate,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=get_video_chunks_number(req.num_frames, tiling_config),
    )
