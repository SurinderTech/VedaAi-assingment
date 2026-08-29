import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY", "").strip()

async def list_models():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        print("Status code:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            models = [m.get("name") for m in data.get("models", [])]
            print(f"Available Gemini API models ({len(models)}):")
            for m in models:
                if "generateContent" in str(m) or "gemini" in str(m).lower():
                    print("  ", m)
        else:
            print("Error response:", r.text)

if __name__ == "__main__":
    asyncio.run(list_models())
