import io
import os
from pydantic import BaseModel
from fastapi import Response

import modal
from modal import Image, Volume

# Setup - define our infrastructure with code!

app = modal.App("rmbg-2-0")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "torchvision", "transformers", "kornia", "pillow", "fastapi[standard]", "pydantic", "timm", "openai"
)

class GenerateRequest(BaseModel):
    image_base64: str


class ExtractObjectRequest(BaseModel):
    image_base64: str
    target_object: str | None = None
    openai_api_key: str | None = None


# This collects the secret from Modal.
secrets = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("openai-secret")
]

GPU = "T4"
MODEL_NAME = "briaai/RMBG-2.0"
CACHE_DIR = "/cache"

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
MIN_CONTAINERS = 0
SCALEDOWN_WINDOW = 120  # Keep the container warm for 2 minutes (120 seconds) after the last request

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


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

    @modal.method()
    def extract_and_remove_background(
        self,
        image_bytes: bytes,
        target_object: str | None = None,
        openai_api_key: str | None = None
    ) -> bytes:
        import io
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
        
        # Build prompt: extract, orient to frontal-side position, white background, HQ details
        if target_object:
            prompt = (
                f"Extract the {target_object} from the image. "
                f"Place the {target_object} in a frontal-side position suitable for 3D generation, "
                f"and make the background solid pure white. The final output must contain only a single {target_object}, "
                f"in high quality (HQ), extremely sharp, with clear details and studio lighting, optimized for 3D reconstruction."
            )
            print(f"Calling OpenAI gpt-image-2-2026-04-21 to extract '{target_object}'...")
        else:
            prompt = (
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
                prompt=prompt,
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

        # 4. Remove background using local RMBG-2.0 model
        print("Removing background using RMBG-2.0...")
        return self.generate.local(image_bytes=extracted_image_bytes)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def extract_and_remove_background_api(self, request: ExtractObjectRequest) -> Response:
        import base64
        image_bytes = base64.b64decode(request.image_base64)
        output_bytes = self.extract_and_remove_background.local(
            image_bytes=image_bytes,
            target_object=request.target_object,
            openai_api_key=request.openai_api_key
        )
        return Response(content=output_bytes, media_type="image/png")



@app.local_entrypoint()
def main(
    image_path: str = "llm-hosting/demo.png",
    output_path: str = "rmbg-2-0-test.png",
    prompt: str = None,
    extract: bool = False,
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

    if prompt or extract:
        print(f"Running extract and background removal on {image_path} (target object: '{prompt}') on Modal...")
        output_bytes = model.extract_and_remove_background.remote(
            image_bytes=image_bytes,
            target_object=prompt
        )
    else:
        print(f"Running background removal on {image_path} on Modal...")
        output_bytes = model.generate.remote(image_bytes)

    with open(output_path, "wb") as output_file:
        output_file.write(output_bytes)
    print(f"Saved generated transparent image to {output_path} ({len(output_bytes)} bytes)")
