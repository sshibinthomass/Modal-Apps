import modal
from modal import Volume, Image

# Setup - define our infrastructure with code!

app = modal.App("llama-3-2-3b-instruct")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", "accelerate"
)

# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "huggingface-secret" with "huggingface" or "hf-secret"
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
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
class LlamaModel:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        # Quant Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=quant_config, device_map="auto"
        )

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        import torch
        from transformers import set_seed

        # Format input using chat template if it is not already formatted
        if isinstance(prompt, list):
            # Prompt is a list of messages (chat history)
            formatted_prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            # Simple string prompt - wrap it as a single user message for Instruct models
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        set_seed(42)
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated response (excluding the prompt)
        generated_tokens = outputs[0][inputs.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


@app.local_entrypoint()
def main():
    model = LlamaModel()
    print("Testing Llama-3.2-3B-Instruct model on Modal...")
    
    # Test with a simple prompt
    prompt = "Tell me a short joke about programming."
    print(f"Prompt: {prompt}")
    
    response = model.generate.remote(prompt)
    print("\nResponse:")
    print(response)
