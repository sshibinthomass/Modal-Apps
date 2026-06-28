import io
import os
from pydantic import BaseModel
from fastapi import Response

import modal
from modal import Image, Volume

# Setup - define our infrastructure with code!

app = modal.App("rmbg-2-0")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "torchvision", "transformers", "kornia", "pillow", "fastapi[standard]", "pydantic", "timm"
)

class GenerateRequest(BaseModel):
    image_base64: str


# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "huggingface-secret" with "huggingface" or "hf-secret"
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
MODEL_NAME = "briaai/RMBG-2.0"
CACHE_DIR = "/cache"

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
MIN_CONTAINERS = 0
SCALEDOWN_WINDOW = 120  # Keep the container warm for 2 minutes (120 seconds) after the last request

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
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

        # Load the model
        self.model = AutoModelForImageSegmentation.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        self.model.to("cuda")
        self.model.eval()

    @modal.method()
    def generate(self, image_bytes: bytes) -> bytes:
        import torch
        from PIL import Image as PILImage
        from torchvision import transforms

        # 1. Load image
        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size

        # 2. Preprocess image
        # RMBG-2.0 is trained on 1024x1024 input size
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Add batch dimension and move to CUDA
        input_images = transform_image(image).unsqueeze(0).to("cuda")

        # 3. Predict mask
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)

        # 4. Resize mask back to original size and apply as alpha
        mask = pred_pil.resize(original_size)
        
        # Create output image with transparency
        output_image = image.copy().convert("RGBA")
        output_image.putalpha(mask)

        # 5. Convert to PNG bytes
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        return buffer.getvalue()

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_api(self, request: GenerateRequest) -> Response:
        import base64
        image_bytes = base64.b64decode(request.image_base64)
        output_bytes = self.generate.local(image_bytes=image_bytes)
        return Response(content=output_bytes, media_type="image/png")



@app.local_entrypoint()
def main(
    image_path: str = "llm-hosting/demo.png",
    output_path: str = "rmbg-2-0-test.png",
):
    import os
    model = RMBGModel()
    
    if not os.path.exists(image_path):
        # Fallback to check llm-inference/img.png if demo.png is not found
        fallback_path = "llm-inference/img.png"
        if os.path.exists(fallback_path):
            image_path = fallback_path
        else:
            print(f"Error: {image_path} does not exist. Please place an image there.")
            return

    with open(image_path, "rb") as input_file:
        image_bytes = input_file.read()

    print(f"Running background removal on {image_path} on Modal...")
    output_bytes = model.generate.remote(image_bytes)

    with open(output_path, "wb") as output_file:
        output_file.write(output_bytes)
    print(f"Saved generated transparent image to {output_path} ({len(output_bytes)} bytes)")
