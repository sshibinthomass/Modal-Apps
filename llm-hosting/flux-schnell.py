import io
import modal
from modal import Volume, Image

# Setup - define our infrastructure with code!

app = modal.App("flux-1-schnell")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "torchvision", "transformers", "diffusers", "accelerate", "sentencepiece", "protobuf"
)

# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "huggingface-secret" with "huggingface" or "hf-secret"
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "L4"
MODEL_NAME = "black-forest-labs/FLUX.1-schnell"
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
        from diffusers import FluxTransformer2DModel, FluxPipeline

        # Load the main 12B transformer in native FP8 (float8_e4m3fn) to save memory
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_NAME,
            subfolder="transformer",
            torch_dtype=torch.float8_e4m3fn
        )
        
        # Load the rest of the pipeline in bfloat16
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_NAME,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        
        # Enable model CPU offloading to guarantee no peak memory spikes during generation
        self.pipe.enable_model_cpu_offload()

    @modal.method()
    def generate(self, prompt: str, num_inference_steps: int = 4, guidance_scale: float = 0.0) -> bytes:
        # Run FLUX.1-schnell inference
        image = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=256,
        ).images[0]

        # Convert image to PNG bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


@app.local_entrypoint()
def main():
    model = FluxModel()
    print("Testing FLUX.1-schnell model on Modal...")
    
    prompt = "A cinematic shot of a futuristic city with flying cars at sunset, high detail."
    print(f"Prompt: {prompt}")
    
    # Run the remote function
    image_bytes = model.generate.remote(prompt)
    
    # Save the output image locally
    output_filename = "flux-schnell-test.png"
    with open(output_filename, "wb") as f:
        f.write(image_bytes)
    print(f"Saved generated image to {output_filename}")
