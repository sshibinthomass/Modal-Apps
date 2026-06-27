import os
import requests

# Helper to load environment variables from .env file if it exists
def load_dotenv():
    # Search for .env in parent or current directory
    for path in [".env", "../.env"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        # Strip optional quotes
                        v = v.strip().strip("'\"")
                        os.environ.setdefault(k.strip(), v)
            break

load_dotenv()

# The endpoint URL from your Modal deployment (shown in the dashboard image)
ENDPOINT_URL = "https://sshibinthomass--ep-gemma-4-31b-it-server.eu-west.modal.direct"
MODEL_NAME = "google/gemma-4-31b-it"

MODAL_KEY = os.environ.get("MODAL_KEY")
MODAL_SECRET = os.environ.get("MODAL_SECRET")


def infer_requests(prompt: str, stream: bool = False) -> str | None:
    """
    Perform inference using the standard requests library (zero extra SDK dependencies).
    """
    url = f"{ENDPOINT_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Modal-Key": MODAL_KEY,
        "Modal-Secret": MODAL_SECRET,
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": stream
    }

    if stream:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        print("Streaming response:")
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                        content = data["choices"][0]["delta"].get("content", "")
                        print(content, end="", flush=True)
                    except Exception:
                        pass
        print()
        return None
    else:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


def infer_openai_sdk(prompt: str, stream: bool = False) -> str | None:
    """
    Perform inference using the official openai Python library.
    To install: pip install openai
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed. Run `pip install openai` to use this method.")
        return None

    # Initialize the client pointing to our Modal endpoint
    # The OpenAI SDK requires api_key, but the authentication is handled via the default_headers
    client = OpenAI(
        base_url=f"{ENDPOINT_URL}/v1",
        api_key="modal",
        default_headers={
            "Modal-Key": MODAL_KEY,
            "Modal-Secret": MODAL_SECRET,
        }
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=stream
    )

    if stream:
        print("Streaming response (SDK):")
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
        print()
        return None
    else:
        return response.choices[0].message.content


if __name__ == "__main__":
    import sys
    
    # Check if keys are loaded
    if not MODAL_KEY or not MODAL_SECRET:
        print("=" * 80)
        print("WARNING: MODAL_KEY and MODAL_SECRET are not set.")
        print("Please configure them in your .env file or environment variables.")
        print("=" * 80)
        print()
        
    prompt = "Explain the difference between supervised and unsupervised learning in 2 sentences."
    print(f"Prompt: {prompt}\n")
    
    # Example using requests (non-streaming)
    try:
        print("--- Testing via requests (non-streaming) ---")
        reply = infer_requests(prompt, stream=False)
        print(f"Response: {reply}\n")
    except Exception as e:
        print(f"Error calling endpoint via requests: {e}\n")

    # Example using requests (streaming)
    try:
        print("--- Testing via requests (streaming) ---")
        infer_requests(prompt, stream=True)
        print()
    except Exception as e:
        print(f"Error calling endpoint via requests streaming: {e}\n")
