"""
Sonic: Shifting Focus to Global Audio Perception in Portrait Animation.

Public API:
  load_models(...)            -> Sonic  – instantiate and return the Sonic pipeline
  generate_video(models, source_image, driving_audio, save_path, **kwargs) -> int
                                        – generate an audio-driven portrait video
  generate(args)              -> int    – convenience: load_models + generate_video

CLI:
  python generate_sonic.py --source_image face.jpg --driving_audio speech.wav --output out.mp4

Note: importing this module inserts the Sonic repo root into sys.path so that
`from src.xxx` imports resolve correctly regardless of working directory.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

_SONIC_ROOT = Path(__file__).resolve().parent
if str(_SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SONIC_ROOT))

from sonic import Sonic

_DEFAULT_MODEL_DIR = os.environ.get("SONIC_MODEL_DIR", str(_SONIC_ROOT))
_DEFAULT_CONFIG_PATH = os.environ.get(
    "SONIC_CONFIG_PATH",
    str(_SONIC_ROOT / "config" / "inference" / "sonic.yaml"),
)


def load_models(
    model_dir: str = None,
    config_path: str = None,
    device_id: int = 0,
    enable_interpolate_frame: bool = True,
) -> Sonic:
    """Load and return the Sonic pipeline.

    Args:
        model_dir:               Root directory containing Sonic checkpoints.
                                 Must contain: checkpoints/Sonic/{unet,audio2token,audio2bucket}.pth,
                                 checkpoints/stable-video-diffusion-img2vid-xt/,
                                 checkpoints/whisper-tiny/, checkpoints/yoloface_v5m.pt,
                                 checkpoints/RIFE/.
                                 Defaults to $SONIC_MODEL_DIR or the Sonic repo root.
        config_path:             Path to inference YAML config.
                                 Defaults to $SONIC_CONFIG_PATH or config/inference/sonic.yaml
                                 next to this file.
        device_id:               CUDA device index (-1 for CPU).
        enable_interpolate_frame: Use RIFE frame interpolation to double FPS (default True).

    Returns:
        Sonic instance ready for inference.
    """
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    return Sonic(
        device_id=device_id,
        enable_interpolate_frame=enable_interpolate_frame,
        model_dir=model_dir,
        config_path=config_path,
    )


def generate_video(
    models: Sonic,
    source_image: str,
    driving_audio: str,
    save_path: str,
    *,
    min_resolution: int = 512,
    inference_steps: int = 25,
    dynamic_scale: float = 1.0,
    keep_resolution: bool = False,
    crop: bool = False,
    expand_ratio: float = 0.5,
    seed: int = None,
) -> int:
    """Generate an audio-driven portrait animation video.

    Args:
        models:           Sonic instance returned by ``load_models()``.
        source_image:     Path to source portrait image (.jpg/.png).
        driving_audio:    Path to driving audio (.wav).
        save_path:        Output .mp4 path.
        min_resolution:   Minimum side length of the output video (default 512).
        inference_steps:  Number of denoising steps (default 25).
        dynamic_scale:    Motion bucket scale controlling animation amplitude (default 1.0).
        keep_resolution:  Preserve the original image resolution in the output (default False).
        crop:             Crop the source image to the detected face region before processing.
        expand_ratio:     Face bbox expansion ratio used when crop=True (default 0.5).
        seed:             Random seed for reproducibility.

    Returns:
        0 on success, -1 if no face is detected.
    """
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if crop:
        face_info = models.preprocess(source_image, expand_ratio=expand_ratio)
        if face_info['face_num'] == 0:
            return -1
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        models.crop_image(source_image, tmp.name, face_info['crop_bbox'])
        source_image = tmp.name

    return models.process(
        source_image,
        driving_audio,
        save_path,
        min_resolution=min_resolution,
        inference_steps=inference_steps,
        dynamic_scale=dynamic_scale,
        keep_resolution=keep_resolution,
        seed=seed,
    )


def generate(args) -> int:
    """Convenience wrapper: load models then generate one video."""
    models = load_models(
        model_dir=args.model_dir,
        config_path=args.config_path,
        device_id=args.device_id,
        enable_interpolate_frame=not args.no_interpolate,
    )
    return generate_video(
        models=models,
        source_image=args.source_image,
        driving_audio=args.driving_audio,
        save_path=args.output,
        min_resolution=args.min_resolution,
        inference_steps=args.inference_steps,
        dynamic_scale=args.dynamic_scale,
        keep_resolution=args.keep_resolution,
        crop=args.crop,
        expand_ratio=args.expand_ratio,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Sonic audio-driven portrait animation inference")

    parser.add_argument("--source_image", type=str, required=True,
                        help="Source portrait image (.jpg/.png)")
    parser.add_argument("--driving_audio", type=str, required=True,
                        help="Driving audio file (.wav)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output video path (.mp4)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Checkpoint root directory (default: $SONIC_MODEL_DIR or repo root)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to inference YAML config (default: config/inference/sonic.yaml)")
    parser.add_argument("--device_id", type=int, default=0,
                        help="CUDA device index (-1 for CPU)")
    parser.add_argument("--no_interpolate", action="store_true",
                        help="Disable RIFE frame interpolation")
    parser.add_argument("--min_resolution", type=int, default=512,
                        help="Minimum output resolution (default 512)")
    parser.add_argument("--inference_steps", type=int, default=25,
                        help="Number of denoising steps (default 25)")
    parser.add_argument("--dynamic_scale", type=float, default=1.0,
                        help="Motion bucket scale (default 1.0)")
    parser.add_argument("--keep_resolution", action="store_true",
                        help="Preserve original image resolution in output")
    parser.add_argument("--crop", action="store_true",
                        help="Crop source image to detected face region before processing")
    parser.add_argument("--expand_ratio", type=float, default=0.5,
                        help="Face bbox expansion ratio when --crop is set (default 0.5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
