"""
Verifies the vLLM server is reachable, the target model is loaded, and inference works.
Safe to run in Google Colab — does NOT call sys.exit() when used as a module.
"""

import sys
import requests

SMOKE_TEST_PROMPT = "Where is the capital of United States? Short answer, just the city name."


def check_server_reachable(vllm_base_url: str) -> bool:
    """Ping the vLLM /health endpoint to confirm the server is up."""
    url = f"{vllm_base_url}/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"  [OK]     vLLM server is reachable at {vllm_base_url}")
            return True
        print(f"  [ERROR]  vLLM server returned HTTP {response.status_code} at {url}")
        return False
    except requests.ConnectionError:
        print(f"  [ERROR]  Cannot connect to vLLM server at {vllm_base_url} — is it running?")
        return False
    except requests.Timeout:
        print(f"  [ERROR]  vLLM server timed out at {url}")
        return False


def check_model_loaded(vllm_base_url: str, model_name: str) -> bool:
    """Confirm the target model appears in the vLLM /v1/models list."""
    url = f"{vllm_base_url}/v1/models"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        model_ids = [m.get("id", "") for m in response.json().get("data", [])]
        if model_name in model_ids:
            print(f"  [OK]     Model '{model_name}' is loaded.")
            return True
        print(f"  [ERROR]  Model '{model_name}' not found. Available: {model_ids or 'none'}")
        return False
    except Exception as e:
        print(f"  [ERROR]  Could not retrieve model list: {e}")
        return False


def check_inference(vllm_base_url: str, model_name: str) -> bool:
    """Send a simple prompt to the model and verify a non-empty response is returned.
    Handles reasoning models (e.g. Qwen3) where content may be None and the answer
    lives in reasoning_content instead."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{vllm_base_url}/v1", api_key="not-required")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": SMOKE_TEST_PROMPT}],
            temperature=0.1,
            max_tokens=4096,
        )
        message = response.choices[0].message
        content = message.content or ""
        # vLLM exposes reasoning under .reasoning; some builds use .reasoning_content
        reasoning_content = (
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
            or ""
        )
        answer = (content or reasoning_content).strip()

        if answer:
            print(f"  [OK]     Inference works. Response: \"{answer[:80]}\"")
            return True

        # Debug: dump the raw message so we can see what the model actually returned
        print(f"  [ERROR]  Model returned an empty response.")
        print(f"  [DEBUG]  finish_reason : {response.choices[0].finish_reason}")
        print(f"  [DEBUG]  content       : {repr(content)}")
        print(f"  [DEBUG]  reasoning     : {repr(reasoning_content)}")
        print(f"  [DEBUG]  full message  : {message}")
        return False
    except Exception as e:
        print(f"  [ERROR]  Inference failed: {e}")
        return False


def main(vllm_base_url: str, model_name: str) -> bool:
    """
    Run all vLLM checks in order.
    Returns True if all checks pass, False otherwise.
    Does NOT call sys.exit() — safe to use inside Google Colab notebooks.
    """
    print("Checking vLLM setup...\n")

    steps = [
        ("server reachable",     lambda: check_server_reachable(vllm_base_url)),
        ("model loaded",         lambda: check_model_loaded(vllm_base_url, model_name)),
        ("inference smoke test", lambda: check_inference(vllm_base_url, model_name)),
    ]

    for label, fn in steps:
        print(f"  Checking {label}...")
        if not fn():
            print(f"\n  vLLM check failed at step: {label}")
            return False

    print("\nAll vLLM checks passed. Ready to run.")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check vLLM server and model readiness.")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Model name to verify")
    args = parser.parse_args()

    if not main(vllm_base_url=args.url, model_name=args.model):
        sys.exit(1)
