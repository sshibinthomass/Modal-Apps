import argparse
import os
import time
from pathlib import Path
import modal

APP_NAME = "openai-to-3d-fast"
CLASS_NAME = "OpenAIImageTo3D"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "img.png"
DEFAULT_OUTPUT = Path(__file__).with_name("openai-to-3d-fast-output.glb")
MIN_GLB_BYTES = 1_000

def validate_glb(glb_bytes: bytes) -> None:
    if len(glb_bytes) < MIN_GLB_BYTES:
        raise ValueError(f"Generated GLB is too small: {len(glb_bytes)} bytes")
    if glb_bytes[:4] != b"glTF":
        raise ValueError("Generated output is not a binary GLB file")

def run_pipeline(
    image_path: Path,
    output_path: Path,
    prompt: str | None = None,
    seed: int = 42,
    pipeline_type: str = "512",
    decimation_target: int = 300_000,
    texture_size: int = 1024,
    remesh: bool = False,
    webp: bool = False,
    use_bf16: bool = False,
) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    model_cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    model = model_cls()

    image_bytes = image_path.read_bytes()
    started_at = time.perf_counter()
    print(f"Calling remote OpenAI-to-3D pipeline class on Modal (prompt: '{prompt}')...")
    glb_bytes = model.generate.remote(
        image_bytes,
        prompt=prompt,
        seed=seed,
        pipeline_type=pipeline_type,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        webp=webp,
        use_bf16=use_bf16,
    )
    elapsed = time.perf_counter() - started_at
    validate_glb(glb_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb_bytes)
    print(f"Remote inference returned in {elapsed:.2f}s")
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenAI + TRELLIS.2-Fast pipeline against deployed Modal app."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", type=str, default="sofa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default="512",
    )
    parser.add_argument("--decimation-target", type=int, default=300_000)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--remesh", action="store_true")
    parser.add_argument("--webp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output = run_pipeline(
        image_path=args.image,
        output_path=args.output,
        prompt=args.prompt,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=args.remesh,
        webp=args.webp,
        use_bf16=args.bf16,
    )
    print(f"Saved generated GLB to {output} ({os.path.getsize(output)} bytes)")
