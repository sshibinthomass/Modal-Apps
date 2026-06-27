import io
import modal
from modal import Volume, Image

# Setup - define our infrastructure with code!

app = modal.App("hunyuan3d-2")

# We build the image starting from a CUDA devel image to compile custom extensions
image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "ninja-build", "libgl1-mesa-glx", "libglib2.0-0", "libgomp1", "clang", "libopengl0")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0"})
    .pip_install(
        "numpy<2.0.0",
        "setuptools<70.0.0",
        "wheel",
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "git clone --recursive https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /app"
    )
    .run_commands(
        "cd /app && pip install -r requirements.txt",
    )
    .run_commands(
        "cd /app && pip install -e .",
    )
    .run_commands(
        "cd /app/hy3dgen/texgen/custom_rasterizer && python3 setup.py install",
    )
    .run_commands(
        "cd /app/hy3dgen/texgen/differentiable_renderer && python3 setup.py install",
    )
    .pip_install("numpy<2.0.0")
    .env({"PYTHONPATH": "/app"})
)

# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "huggingface-secret" with "huggingface" or "hf-secret"
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "A100-80GB"
MODEL_NAME = "tencent/Hunyuan3D-2"
CACHE_DIR = "/cache"

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
MIN_CONTAINERS = 0
SCALEDOWN_WINDOW = 120  # Keep the container warm for 2 minutes (120 seconds) after the last request

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Hunyuan3DModel:
    @modal.enter()
    def setup(self):
        import sys
        sys.path.insert(0, "/app")
        
        # Apply torchvision compatibility fix if it exists in the cloned repository
        try:
            from torchvision_fix import apply_fix
            apply_fix()
            print("Successfully applied torchvision fix.")
        except ImportError:
            print("torchvision_fix not found, skipping...")

        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        print("Loading Hunyuan3D shape generation pipeline...")
        self.pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_NAME
        )
        print("Loading Hunyuan3D texture generation pipeline...")
        self.pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
            MODEL_NAME
        )

    @modal.method()
    def generate(self, image_bytes: bytes) -> bytes:
        import sys
        sys.path.insert(0, "/app")
        
        from PIL import Image
        from hy3dgen.rembg import BackgroundRemover

        # 1. Load the input image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # 2. Run background removal to isolate the foreground object
        print("Removing background from the input image...")
        rembg = BackgroundRemover()
        img = rembg(img)

        # 3. Generate Geometry (Shape generation)
        # Returns a list of generated trimesh/scene objects, take the first one
        print("Generating 3D mesh geometry...")
        mesh = self.pipeline_shapegen(image=img)[0]

        # 4. Generate Texture (Paint generation)
        print("Generating mesh texture...")
        mesh = self.pipeline_texgen(mesh, image=img)

        # 5. Export mesh to GLB bytes
        print("Exporting generated 3D model to GLB bytes...")
        glb_bytes = mesh.export(file_type='glb')
        return glb_bytes


@app.local_entrypoint()
def main():
    model = Hunyuan3DModel()
    print("Testing Hunyuan3D-2 model on Modal...")
    
    # Download the official demo image
    import urllib.request
    url = "https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2/main/assets/demo.png"
    demo_img_path = "demo.png"
    print(f"Downloading test image from {url}...")
    urllib.request.urlretrieve(url, demo_img_path)
    
    with open(demo_img_path, "rb") as f:
        image_bytes = f.read()
        
    print("Running remote inference on Modal...")
    glb_bytes = model.generate.remote(image_bytes)
    
    output_filename = "hunyuan3d-2-test.glb"
    with open(output_filename, "wb") as f_out:
        f_out.write(glb_bytes)
    print(f"Saved generated 3D model to {output_filename}")
