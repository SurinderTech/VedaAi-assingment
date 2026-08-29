import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY", "").strip()

models_to_test = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
]

async def test_model(model: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "contents": [{"parts": [{"text": "Reply with 'OK'."}]}],
        "generationConfig": {"maxOutputTokens": 20, "temperature": 0.1}
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=body)
            if r.status_code == 200:
                text = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                print(f"  [SUCCESS] {model:<30} -> Status 200 | Response: '{text.strip()}'")
                return True
            else:
                err_body = r.text[:120].replace("\n", " ")
                print(f"  [FAILED ] {model:<30} -> Status {r.status_code} | Error: {err_body}")
                return False
    except Exception as e:
        print(f"  [ERROR  ] {model:<30} -> Exception: {e}")
        return False

async def main():
    print("\n--- Testing Gemini API Models ---")
    results = []
    for model in models_to_test:
        ok = await test_model(model)
        if ok:
            results.append(model)
    print(f"\nWorking Gemini Models: {results}")

if __name__ == "__main__":
    asyncio.run(main())
