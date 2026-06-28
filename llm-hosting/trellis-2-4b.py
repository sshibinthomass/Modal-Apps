import io
import os
import tempfile
from pydantic import BaseModel
from fastapi import Response

import modal
from modal import Image, Volume



app = modal.App("trellis-2-4b")

GPU = "H100"
MODEL_NAME = "microsoft/TRELLIS.2-4B"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REMBG_REPO = "briaai/RMBG-2.0"
CACHE_DIR = "/cache"
TRELLIS_DIR = "/app"
MIN_CONTAINERS = 0
SCALEDOWN_WINDOW = 300

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])]

image = (
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
        (REMBG_REPO, "config.json"),
    ]
    for repo_id, filename in gated_files:
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        except Exception as exc:
            raise RuntimeError(
                "TRELLIS.2 requires Hugging Face access to gated dependency "
                f"{repo_id}. Visit https://huggingface.co/{repo_id}, request or "
                "accept access with the account that owns HF_TOKEN, then recreate "
                "the Modal secret with that token."
            ) from exc


class GenerateRequest(BaseModel):
    image_base64: str
    seed: int = 42
    pipeline_type: str = "1024_cascade"
    decimation_target: int = 1_000_000
    texture_size: int = 4096



@app.cls(
    image=image,
    secrets=secrets,
    gpu=GPU,
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
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_NAME)
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

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_api(self, request: GenerateRequest) -> Response:
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
    image_path: str = "llm-hosting/demo.png",
    output_path: str = "trellis-2-4b-test.glb",
    seed: int = 42,
):
    model = Trellis2Model()
    with open(image_path, "rb") as input_file:
        image_bytes = input_file.read()

    print(f"Generating TRELLIS.2 3D asset from {image_path} on Modal...")
    glb_bytes = model.generate.remote(image_bytes, seed=seed)

    with open(output_path, "wb") as output_file:
        output_file.write(glb_bytes)
    print(f"Saved generated GLB to {output_path} ({len(glb_bytes)} bytes)")
