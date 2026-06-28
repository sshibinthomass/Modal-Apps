import io
import os
import tempfile
from pydantic import BaseModel
from fastapi import Response

import modal
from modal import Image, Volume

# Define the single Modal App
app = modal.App("image-to-3d")

GPU_T4 = "T4"
GPU_H100 = "H100"
MODEL_RMBG = "briaai/RMBG-2.0"
MODEL_TRELLIS = "microsoft/TRELLIS.2-4B"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CACHE_DIR = "/cache"
TRELLIS_DIR = "/app"
MIN_CONTAINERS = 0
SCALEDOWN_WINDOW = 300

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])]

# Image definition for RMBG background removal
image_rmbg = Image.debian_slim().pip_install(
    "huggingface", "torch", "torchvision", "transformers", "kornia", "pillow", "fastapi[standard]", "pydantic", "timm"
)

# Image definition for TRELLIS 3D generation
image_trellis = (
    Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "clang",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libjpeg-dev",
        "ninja-build",
    )
    .env(
        {
            "ATTN_BACKEND": "flash-attn",
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HUB_CACHE": CACHE_DIR,
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTHONPATH": TRELLIS_DIR,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_CUDA_ARCH_LIST": "9.0",
        }
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "setuptools",
        "easydict",
        "gradio==6.0.1",
        "huggingface_hub",
        "imageio",
        "imageio-ffmpeg",
        "kornia",
        "lpips",
        "ninja",
        "opencv-python-headless",
        "pandas",
        "safetensors",
        "timm",
        "tqdm",
        "transformers",
        "trimesh",
        "wheel",
        "zstandard",
        "pydantic",
        "fastapi[standard]",
    )
    .pip_install(
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
    )
    .pip_install("flash-attn==2.7.3", extra_options="--no-build-isolation")
    .run_commands(
        f"git clone --recursive https://github.com/microsoft/TRELLIS.2.git {TRELLIS_DIR}"
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/app/trellis2/modules/image_feature_extractor.py')\n"
        "text = path.read_text()\n"
        "old = \"        for i, layer_module in enumerate(self.model.layer):\\n\"\n"
        "new = (\n"
        "    \"        layers = getattr(self.model, 'layer', None)\\n\"\n"
        "    \"        if layers is None:\\n\"\n"
        "    \"            layers = self.model.model.layer\\n\"\n"
        "    \"\\n\"\n"
        "    \"        for i, layer_module in enumerate(layers):\\n\"\n"
        ")\n"
        "if old not in text:\n"
        "    raise RuntimeError('Expected DINOv3 layer loop not found in TRELLIS image_feature_extractor.py')\n"
        "path.write_text(text.replace(old, new))\n"
        "PY"
    )
    .run_commands(
        "git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast",
        "pip install /tmp/nvdiffrast --no-build-isolation",
    )
    .run_commands(
        "git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec",
        "pip install /tmp/nvdiffrec --no-build-isolation",
    )
    .run_commands(
        "git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh",
        "pip install /tmp/CuMesh --no-build-isolation",
    )
    .run_commands(
        "git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM",
        "pip install /tmp/FlexGEMM --no-build-isolation",
    )
    .run_commands(
        f"cp -r {TRELLIS_DIR}/o-voxel /tmp/o-voxel",
        "pip install /tmp/o-voxel --no-build-isolation",
    )
)


def assert_huggingface_access() -> None:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing from the Modal secret 'huggingface-secret'. "
            "Create it with: modal secret create huggingface-secret HF_TOKEN=hf_... --force"
        )

    gated_files = [
        (DINO_REPO, "config.json"),
        (MODEL_RMBG, "config.json"),
    ]
    for repo_id, filename in gated_files:
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        except Exception as exc:
            raise RuntimeError(
                "Pipeline requires Hugging Face access to gated dependency "
                f"{repo_id}. Visit https://huggingface.co/{repo_id}, request or "
                "accept access with the account that owns HF_TOKEN, then recreate "
                "the Modal secret with that token."
            ) from exc


class PipelineRequest(BaseModel):
    image_base64: str
    seed: int = 42
    pipeline_type: str = "1024_cascade"
    decimation_target: int = 1_000_000
    texture_size: int = 4096


@app.cls(
    image=image_rmbg.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU_T4,
    timeout=600,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={CACHE_DIR: hf_cache_volume},
)
class RMBGModel:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoModelForImageSegmentation

        self.model = AutoModelForImageSegmentation.from_pretrained(
            MODEL_RMBG, 
            trust_remote_code=True
        )
        self.model.to("cuda")
        self.model.eval()

    @modal.method()
    def generate(self, image_bytes: bytes) -> bytes:
        import torch
        from PIL import Image as PILImage
        from torchvision import transforms

        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size

        # RMBG-2.0 is trained on 1024x1024 input size
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_images = transform_image(image).unsqueeze(0).to("cuda")

        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)

        mask = pred_pil.resize(original_size)
        
        output_image = image.copy().convert("RGBA")
        output_image.putalpha(mask)

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        return buffer.getvalue()


@app.cls(
    image=image_trellis,
    secrets=secrets,
    gpu=GPU_H100,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Trellis2Model:
    @modal.enter()
    def setup(self):
        self.pipeline = None

    def load_pipeline(self):
        if self.pipeline is not None:
            return

        import torch
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        assert_huggingface_access()
        torch.set_float32_matmul_precision("high")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_TRELLIS)
        self.pipeline.cuda()

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        seed: int = 42,
        pipeline_type: str = "1024_cascade",
        decimation_target: int = 1_000_000,
        texture_size: int = 4096,
    ) -> bytes:
        from PIL import Image as PILImage
        import o_voxel

        self.load_pipeline()
        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")
        mesh = self.pipeline.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
        )[0]
        mesh.simplify(16_777_216)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as output:
            output_path = output.name
        try:
            glb.export(output_path, extension_webp=True)
            with open(output_path, "rb") as output:
                return output.read()
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


@app.cls(
    image=image_rmbg, # Use rmbg image environment (contains FastAPI, pillow, etc.)
    secrets=secrets,
    timeout=1800,
)
class ImageTo3D:
    @modal.enter()
    def setup(self):
        # Reference the models within the same app
        self.remover = RMBGModel()
        self.trellis = Trellis2Model()

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        seed: int = 42,
        pipeline_type: str = "1024_cascade",
        decimation_target: int = 1_000_000,
        texture_size: int = 4096,
    ) -> bytes:
        print("Pipeline Step 1: Removing background...")
        clean_bytes = self.remover.generate.remote(image_bytes)

        print("Pipeline Step 2: Generating 3D asset from transparent image...")
        glb_bytes = self.trellis.generate.remote(
            clean_bytes,
            seed=seed,
            pipeline_type=pipeline_type,
            decimation_target=decimation_target,
            texture_size=texture_size,
        )
        return glb_bytes

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_api(self, request: PipelineRequest) -> Response:
        import base64
        image_bytes = base64.b64decode(request.image_base64)
        glb_bytes = self.generate.local(
            image_bytes=image_bytes,
            seed=request.seed,
            pipeline_type=request.pipeline_type,
            decimation_target=request.decimation_target,
            texture_size=request.texture_size
        )
        return Response(content=glb_bytes, media_type="model/gltf-binary")


@app.local_entrypoint()
def main(
    image_path: str = "llm-inference/img.png",
    output_path: str = "image-to-3d-output.glb",
    seed: int = 42,
):
    if not os.path.exists(image_path):
        # Fallback check
        fallback_path = "../llm-inference/img.png"
        if os.path.exists(fallback_path):
            image_path = fallback_path
        else:
            print(f"Error: Input image {image_path} does not exist.")
            return

    model = ImageTo3D()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Executing Image-to-3D pipeline for {image_path}...")
    glb_bytes = model.generate.remote(image_bytes, seed=seed)

    with open(output_path, "wb") as f:
        f.write(glb_bytes)
    print(f"Successfully saved combined pipeline output to {output_path} ({len(glb_bytes)} bytes)")
