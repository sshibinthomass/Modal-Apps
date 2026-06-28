from pydantic import BaseModel
from typing import Union, List, Dict, Any
import modal
from modal import Image, Volume

# Setup - define our infrastructure with code!

app = modal.App("gemma-4-12b-it")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "torchvision", "transformers", "accelerate", "librosa", "pydantic", "fastapi[standard]"
)

class GenerateRequest(BaseModel):
    prompt: Union[str, List[Dict[str, Any]]]
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    enable_thinking: bool = False


# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "huggingface-secret" with "huggingface" or "hf-secret"
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "A100-80GB"
MODEL_NAME = "google/gemma-4-12B-it"
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
class GemmaModel:
    @modal.enter()
    def setup(self):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_NAME,
            dtype="auto",
            device_map="auto",
        )

    @modal.method()
    def generate(
        self,
        prompt: str | list[dict],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = False,
    ) -> str:
        import torch

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=False,
        )
        parsed = self.processor.parse_response(response)
        if isinstance(parsed, dict):
            return parsed.get("answer") or parsed.get("content") or str(parsed)
        return str(parsed)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_api(self, request: GenerateRequest) -> str:
        return self.generate.local(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            enable_thinking=request.enable_thinking
        )



@app.local_entrypoint()
def main():
    model = GemmaModel()
    print("Testing Gemma 4 12B IT model on Modal...")

    prompt = "Tell me a short joke about saving GPU memory."
    print(f"Prompt: {prompt}")

    response = model.generate.remote(prompt)
    print("\nResponse:")
    print(response)
