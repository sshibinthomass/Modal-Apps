import io
import os
import tempfile
import uuid
from pathlib import Path

import modal
from modal import Image, Volume


app = modal.App("hunyuan3d-2")

GPU = "H100"
MODEL_NAME = "tencent/Hunyuan3D-2.1"
HF_CACHE_DIR = "/hf-cache"
HY3DGEN_CACHE_DIR = "/hy3dgen-cache"
APP_DIR = "/app"
MIN_CONTAINERS = 0
SCALEDOWN_WINDOW = 300
TEXTURE_SUPPORT_ENABLED = False

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)
hy3dgen_cache_volume = Volume.from_name("hy3dgen-cache", create_if_missing=True)
rembg_cache_volume = Volume.from_name("rembg-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface-secret")]

image = (
    Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "clang",
        "cmake",
        "git",
        "libegl1",
        "libegl1-mesa-dev",
        "libgl1",
        "libgles2",
        "libgles2-mesa-dev",
        "libglib2.0-0",
        "libglvnd0",
        "libglvnd-dev",
        "libgomp1",
        "libjpeg-dev",
        "libopengl0",
        "libxrender1",
        "ninja-build",
        "pkg-config",
        "python3-dev",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "HY3DGEN_MODELS": HY3DGEN_CACHE_DIR,
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
            "PYTHONPATH": f"{APP_DIR}:{APP_DIR}/hy3dshape:{APP_DIR}/hy3dpaint",
            "PYOPENGL_PLATFORM": "egl",
            "TORCH_CUDA_ARCH_LIST": "9.0",
        }
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "accelerate==1.1.1",
        "configargparse==1.7",
        "diffusers==0.30.0",
        "einops==0.8.0",
        "huggingface-hub==0.30.2",
        "imageio==2.36.0",
        "ninja==1.11.1.1",
        "numpy==1.24.4",
        "omegaconf==2.3.0",
        "onnxruntime==1.16.3",
        "opencv-python-headless==4.10.0.84",
        "pybind11==2.13.4",
        "pygltflib==1.16.3",
        "pymeshlab==2022.2.post3",
        "rembg==2.0.65",
        "safetensors==0.4.4",
        "scikit-image==0.24.0",
        "scipy==1.14.1",
        "tb_nightly==2.18.0a20240726",
        "timm",
        "torchdiffeq",
        "tqdm==4.66.5",
        "transformers==4.46.0",
        "trimesh==4.4.7",
        "xatlas==0.0.9",
        "pyyaml==6.0.2",
    )
    .run_commands(
        f"git clone --recursive https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git {APP_DIR}"
    )
)


@app.cls(
    image=image,
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        HY3DGEN_CACHE_DIR: hy3dgen_cache_volume,
        "/root/.u2net": rembg_cache_volume,
    },
)
class Hunyuan3DModel:
    @modal.enter()
    def setup(self):
        import sys

        sys.path.insert(0, f"{APP_DIR}/hy3dshape")
        sys.path.insert(0, f"{APP_DIR}/hy3dpaint")
        sys.path.insert(0, APP_DIR)

        try:
            from torchvision_fix import apply_fix

            apply_fix()
        except Exception as exc:
            print(f"Warning: torchvision compatibility fix was not applied: {exc}")

        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        print(f"Loading Hunyuan3D shape pipeline from {MODEL_NAME}...")
        self.pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_NAME
        )
        self.pipeline_texgen = None

    def load_texture_pipeline(self):
        if not TEXTURE_SUPPORT_ENABLED:
            raise RuntimeError(
                "Texture generation is not enabled in this lean Modal image. "
                "The verified path currently returns a valid Hunyuan3D-2.1 shape GLB. "
                "Enable the full hy3dpaint dependency stack before calling texture=True."
            )
        if self.pipeline_texgen is not None:
            return

        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        print("Loading Hunyuan3D paint pipeline...")
        paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
        paint_config.realesrgan_ckpt_path = f"{APP_DIR}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        paint_config.multiview_cfg_path = f"{APP_DIR}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        paint_config.custom_pipeline = f"{APP_DIR}/hy3dpaint/hunyuanpaintpbr"
        self.pipeline_texgen = Hunyuan3DPaintPipeline(paint_config)

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        texture: bool = False,
        remove_background: bool = True,
    ) -> bytes:
        import sys

        sys.path.insert(0, f"{APP_DIR}/hy3dshape")
        sys.path.insert(0, f"{APP_DIR}/hy3dpaint")
        sys.path.insert(0, APP_DIR)

        from PIL import Image as PILImage
        from hy3dshape.rembg import BackgroundRemover

        input_image = PILImage.open(io.BytesIO(image_bytes))
        if remove_background:
            print("Removing background from input image...")
            input_image = BackgroundRemover()(input_image.convert("RGB"))
        else:
            input_image = input_image.convert("RGBA")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = uuid.uuid4().hex
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / f"{run_id}_input.png"
            shape_glb_path = tmpdir_path / f"{run_id}_shape.glb"
            textured_obj_path = tmpdir_path / f"{run_id}_textured.obj"
            textured_glb_path = tmpdir_path / f"{run_id}_textured.glb"

            input_image.save(input_path)

            print("Generating Hunyuan3D-2.1 shape mesh...")
            mesh = self.pipeline_shapegen(image=input_image)[0]
            mesh.export(shape_glb_path)

            output_path = shape_glb_path
            if texture:
                self.load_texture_pipeline()
                print("Generating Hunyuan3D-2.1 PBR texture...")
                self.pipeline_texgen(
                    mesh_path=str(shape_glb_path),
                    image_path=str(input_path),
                    output_mesh_path=str(textured_obj_path),
                    save_glb=True,
                )
                if not textured_glb_path.exists():
                    raise RuntimeError(
                        f"Texture pipeline did not create expected GLB: {textured_glb_path}"
                    )
                output_path = textured_glb_path

            output_bytes = output_path.read_bytes()
            if output_bytes[:4] != b"glTF":
                raise RuntimeError(f"Generated file is not a binary GLB: {output_path}")
            return output_bytes


@app.local_entrypoint()
def main(
    image_path: str = "llm-hosting/demo.png",
    output_path: str = "hunyuan3d-2.1-test.glb",
    texture: bool = False,
):
    model = Hunyuan3DModel()
    with open(image_path, "rb") as input_file:
        image_bytes = input_file.read()

    print(f"Testing Hunyuan3D-2.1 on Modal with {image_path}...")
    glb_bytes = model.generate.remote(image_bytes, texture=texture)

    with open(output_path, "wb") as output_file:
        output_file.write(glb_bytes)
    print(f"Saved generated 3D model to {output_path} ({len(glb_bytes)} bytes)")
