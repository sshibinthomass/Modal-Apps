import io
import os
import tempfile
import time
from pydantic import BaseModel
from fastapi import Response
from fastapi.responses import JSONResponse

import modal
from modal import Image, Volume

app = modal.App("openai-to-3d-fast")

GPU_H100 = "H100"
MODEL_NAME = "microsoft/TRELLIS.2-4B"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REMBG_REPO = "briaai/RMBG-2.0"
CACHE_DIR = "/cache"
TRELLIS_DIR = "/app"
MIN_CONTAINERS = 0
SCALEDOWN_WINDOW = 120

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)
secrets = [
    modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
    modal.Secret.from_name("openai-secret", required_keys=["OPENAI_API_KEY"])
]

image_light = Image.debian_slim().pip_install(
    "huggingface", "pillow", "fastapi[standard]", "pydantic", "openai"
)

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


def prepare_image_for_openai(image_bytes: bytes) -> bytes:
    from PIL import Image as PILImage
    import io

    img = PILImage.open(io.BytesIO(image_bytes))
    w, h = img.size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    if new_w == 0:
        new_w = 16
    if new_h == 0:
        new_h = 16

    if (new_w, new_h) != (w, h) or img.format != "PNG":
        img = img.convert("RGB")
        img = img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
    
    out_buf = io.BytesIO()
    img.save(out_buf, format="PNG")
    return out_buf.getvalue()


class TrellisGenerateRequest(BaseModel):
    image_base64: str
    seed: int = 42
    pipeline_type: str = "512"
    decimation_target: int = 300_000
    texture_size: int = 1024
    remesh: bool = False
    webp: bool = False
    use_bf16: bool = False


class PipelineRequest(BaseModel):
    image_base64: str
    prompt: str | None = None
    seed: int = 42
    pipeline_type: str = "512"
    decimation_target: int = 300_000
    texture_size: int = 1024
    remesh: bool = False
    webp: bool = False
    use_bf16: bool = False


@app.cls(
    image=image_trellis,
    secrets=secrets,
    gpu=GPU_H100,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Trellis2FastModel:
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_NAME)
        self.pipeline.cuda()

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        seed: int = 42,
        pipeline_type: str = "512",
        decimation_target: int = 300_000,
        texture_size: int = 1024,
        remesh: bool = False,
        webp: bool = False,
        use_bf16: bool = False,
    ) -> bytes:
        from PIL import Image as PILImage
        import o_voxel
        import torch

        total_start = time.perf_counter()
        self.load_pipeline()
        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")

        inference_start = time.perf_counter()
        if use_bf16:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                mesh = self.pipeline.run(
                    image,
                    seed=seed,
                    pipeline_type=pipeline_type,
                )[0]
        else:
            with torch.inference_mode():
                mesh = self.pipeline.run(
                    image,
                    seed=seed,
                    pipeline_type=pipeline_type,
                )[0]
        print(f"TRELLIS fast inference took {time.perf_counter() - inference_start:.2f}s")

        postprocess_start = time.perf_counter()
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
            remesh=remesh,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        print(
            f"TRELLIS fast postprocess took {time.perf_counter() - postprocess_start:.2f}s"
        )

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as output:
            output_path = output.name
        try:
            glb.export(output_path, extension_webp=webp)
            with open(output_path, "rb") as output:
                output_bytes = output.read()
            print(f"TRELLIS fast total request took {time.perf_counter() - total_start:.2f}s")
            return output_bytes
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_api(self, request: TrellisGenerateRequest) -> Response:
        import base64
        image_bytes = base64.b64decode(request.image_base64)
        glb_bytes = self.generate.local(
            image_bytes=image_bytes,
            seed=request.seed,
            pipeline_type=request.pipeline_type,
            decimation_target=request.decimation_target,
            texture_size=request.texture_size,
            remesh=request.remesh,
            webp=request.webp,
            use_bf16=request.use_bf16
        )
        return Response(content=glb_bytes, media_type="model/gltf-binary")


@app.cls(
    image=image_light,
    secrets=secrets,
    timeout=1800,
)
class OpenAIImageTo3D:
    @modal.enter()
    def setup(self):
        self.trellis = Trellis2FastModel()

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        prompt: str | None = None,
        seed: int = 42,
        pipeline_type: str = "512",
        decimation_target: int = 300_000,
        texture_size: int = 1024,
        remesh: bool = False,
        webp: bool = False,
        use_bf16: bool = False,
        openai_api_key: str | None = None
    ) -> bytes:
        import base64
        from openai import OpenAI

        # 1. Resolve API key
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the request or create an 'openai-secret' in Modal."
            )

        # 2. Prepare the input image for the OpenAI Images Edit API (PNG, divisible by 16 dimensions)
        prepared_png_bytes = prepare_image_for_openai(image_bytes)
        image_file = io.BytesIO(prepared_png_bytes)
        image_file.name = "input.png"

        # 3. Call OpenAI gpt-image-2-2026-04-21 model via the edit endpoint
        client = OpenAI(api_key=api_key)
        
        if prompt:
            prompt_text = (
                f"Extract the {prompt} from the image. "
                f"Place the {prompt} in a frontal-side position suitable for 3D generation, "
                f"and make the background solid pure white. The final output must contain only a single {prompt}, "
                f"in high quality (HQ), extremely sharp, with clear details and studio lighting, optimized for 3D reconstruction."
            )
            print(f"Calling OpenAI gpt-image-2-2026-04-21 to extract '{prompt}'...")
        else:
            prompt_text = (
                "Extract the main, most prominent object from the image. "
                "Place it in a frontal-side position suitable for 3D generation, "
                "and make the background solid pure white. The final output must contain only a single object, "
                "in high quality (HQ), extremely sharp, with clear details and studio lighting, optimized for 3D reconstruction."
            )
            print("Calling OpenAI gpt-image-2-2026-04-21 to extract the main object...")

        try:
            response = client.images.edit(
                model="gpt-image-2-2026-04-21",
                image=image_file,
                prompt=prompt_text,
                n=1
            )
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

        img_data = response.data[0]
        if getattr(img_data, "b64_json", None):
            extracted_image_bytes = base64.b64decode(img_data.b64_json)
        elif getattr(img_data, "url", None):
            import urllib.request
            print(f"Downloading generated image from {img_data.url}...")
            try:
                with urllib.request.urlopen(img_data.url) as response_http:
                    extracted_image_bytes = response_http.read()
            except Exception as e:
                raise RuntimeError(f"Failed to download image from URL: {str(e)}")
        else:
            raise RuntimeError("Neither b64_json nor url was returned in OpenAI response.")

        print("Successfully generated image via OpenAI. Feeding directly to Trellis...")
        glb_bytes = self.trellis.generate.remote(
            extracted_image_bytes,
            seed=seed,
            pipeline_type=pipeline_type,
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
            webp=webp,
            use_bf16=use_bf16
        )
        return glb_bytes

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_api(self, request: PipelineRequest) -> Response:
        import base64
        image_bytes = base64.b64decode(request.image_base64)
        glb_bytes = self.generate.local(
            image_bytes=image_bytes,
            prompt=request.prompt,
            seed=request.seed,
            pipeline_type=request.pipeline_type,
            decimation_target=request.decimation_target,
            texture_size=request.texture_size,
            remesh=request.remesh,
            webp=request.webp,
            use_bf16=request.use_bf16
        )
        return Response(content=glb_bytes, media_type="model/gltf-binary")

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def start_api(self, request: PipelineRequest) -> JSONResponse:
        import base64
        image_bytes = base64.b64decode(request.image_base64)
        call = self.generate.spawn(
            image_bytes=image_bytes,
            prompt=request.prompt,
            seed=request.seed,
            pipeline_type=request.pipeline_type,
            decimation_target=request.decimation_target,
            texture_size=request.texture_size,
            remesh=request.remesh,
            webp=request.webp,
            use_bf16=request.use_bf16
        )
        return JSONResponse({"call_id": call.object_id})

    @modal.fastapi_endpoint(method="GET", requires_proxy_auth=True)
    def result_api(self, call_id: str):
        call = modal.FunctionCall.from_id(call_id)
        try:
            glb_bytes = call.get(timeout=0)
        except TimeoutError:
            return JSONResponse({"status": "running"}, status_code=202)
        return Response(content=glb_bytes, media_type="model/gltf-binary")


@app.local_entrypoint()
def main(
    image_path: str = "llm-hosting/demo.png",
    output_path: str = "openai-to-3d-fast-output.glb",
    prompt: str = None,
    seed: int = 42,
):
    if not os.path.exists(image_path):
        # Fallback check
        fallback_path = "llm-inference/img.png"
        if os.path.exists(fallback_path):
            image_path = fallback_path
        else:
            print(f"Error: Input image {image_path} does not exist.")
            return

    model = OpenAIImageTo3D()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Executing OpenAI-to-3D direct pipeline for {image_path}...")
    glb_bytes = model.generate.remote(
        image_bytes,
        prompt=prompt,
        seed=seed
    )

    with open(output_path, "wb") as f:
        f.write(glb_bytes)
    print(f"Successfully saved combined pipeline output to {output_path} ({len(glb_bytes)} bytes)")
