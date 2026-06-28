import argparse
import os
from pathlib import Path

import modal


APP_NAME = "trellis-2-4b"
CLASS_NAME = "Trellis2Model"
DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "llm-hosting" / "demo.png"
DEFAULT_OUTPUT = Path(__file__).with_name("trellis-2-4b-output.glb")
MIN_GLB_BYTES = 1_000


def validate_glb(glb_bytes: bytes) -> None:
    if len(glb_bytes) < MIN_GLB_BYTES:
        raise ValueError(f"Generated GLB is too small: {len(glb_bytes)} bytes")
    if glb_bytes[:4] != b"glTF":
        raise ValueError("Generated output is not a binary GLB file")


def generate_3d(
    image_path: Path,
    output_path: Path,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    model_cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    model = model_cls()

    image_bytes = image_path.read_bytes()
    glb_bytes = model.generate.remote(
        image_bytes,
        seed=seed,
        pipeline_type=pipeline_type,
    )
    validate_glb(glb_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb_bytes)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run image-to-3D inference against the deployed TRELLIS.2 Modal app."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default="1024_cascade",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = generate_3d(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
    )
    print(f"Saved TRELLIS.2 GLB to {output} ({os.path.getsize(output)} bytes)")
