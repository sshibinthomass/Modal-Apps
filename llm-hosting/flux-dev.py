import io
import modal
from modal import Volume, Image

# Setup - define our infrastructure with code!

app = modal.App("flux-dev-2")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "torchvision", "transformers", "diffusers", "accelerate", "sentencepiece", "protobuf"
)

# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "huggingface-secret" with "huggingface" or "hf-secret"
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "A100-80GB"
MODEL_NAME = "black-forest-labs/FLUX.2-dev"
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
class FluxModel:
    @modal.enter()
    def setup(self):
        import torch
        from diffusers import Flux2Pipeline

        # Load the pipeline in bfloat16 for optimal memory/speed on A100-80GB
        self.pipe = Flux2Pipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )

    @modal.method()
    def generate(self, prompt: str, num_inference_steps: int = 28, guidance_scale: float = 3.5) -> bytes:
        # Run FLUX.2-dev inference
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
        ).images[0]

        # Convert image to PNG bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


@app.local_entrypoint()
def main():
    model = FluxModel()
    print("Testing FLUX.Dev-2 model on Modal...")
    
    prompt = "A cinematic shot of a futuristic city with flying cars at sunset, high detail."
    print(f"Prompt: {prompt}")
    
    # Run the remote function
    image_bytes = model.generate.remote(prompt)
    
    # Save the output image locally
    output_filename = "flux-dev-2-test.png"
    with open(output_filename, "wb") as f:
        f.write(image_bytes)
    print(f"Saved generated image to {output_filename}")
