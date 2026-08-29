import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENROUTER_API_KEY", "").strip()
print(f"OPENROUTER_API_KEY configured: {bool(key and len(key) >= 10)} (length: {len(key)})")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {key}",
    "HTTP-Referer": "https://vedaai.app",
    "X-Title": "VedaAI Examiner Assistant",
    "Content-Type": "application/json",
}

OPENROUTER_FREE_MODELS = [
    "dots-studio/dots-3-note-preview:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "thinkingmachines/inkling:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
]

async def test_openrouter_model(model: str):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with 'OK'."}],
        "max_tokens": 20,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=body, headers=headers)
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"  [SUCCESS] {model:<40} -> Status 200 | Response: '{content.strip()}'")
                return True
            else:
                err = r.text[:120].replace("\n", " ")
                print(f"  [FAILED ] {model:<40} -> Status {r.status_code} | Error: {err}")
                return False
    except Exception as e:
        print(f"  [ERROR  ] {model:<40} -> Exception: {e}")
        return False

async def main():
    print("\n--- Testing OpenRouter Free Models ---")

    # Also discover live free models from OpenRouter API
    live_free_models = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res_m = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
            if res_m.status_code == 200:
                data = res_m.json().get("data", [])
                for item in data:
                    mid = item.get("id", "")
                    pricing = item.get("pricing", {})
                    # Check if free by ID suffix or pricing
                    if mid.endswith(":free") or (pricing.get("prompt") == "0" and pricing.get("completion") == "0"):
                        live_free_models.append(mid)
    except Exception as e:
        print(f"Live discovery error: {e}")

    print(f"Discovered {len(live_free_models)} live free models on OpenRouter.")

    all_to_test = []
    for m in OPENROUTER_FREE_MODELS + live_free_models:
        if m not in all_to_test:
            all_to_test.append(m)

    working = []
    for m in all_to_test:
        ok = await test_openrouter_model(m)
        if ok:
            working.append(m)

    print(f"\nWorking OpenRouter Free Models ({len(working)}): {working}")

if __name__ == "__main__":
    asyncio.run(main())
