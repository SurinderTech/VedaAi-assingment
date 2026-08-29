import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import llm_provider
from unittest.mock import patch

async def test_fallback():
    print("\n================================================================================")
    print(" TESTING GEMINI -> OPENROUTER FALLBACK ENGINE")
    print("================================================================================")

    async def mock_gemini_fail(*args, **kwargs):
        raise llm_provider.LLMError("Simulated Gemini Rate Limit (429)")

    with patch.dict(llm_provider._PROVIDERS, {"gemini": mock_gemini_fail}):
        start = time.time()
        res, provider_used = await llm_provider.llm_complete_json_with_provider(
            'Return valid JSON: {"status": "ok", "answer": 42}', purpose="fallback_test"
        )
        latency = round(time.time() - start, 3)

        print(f"Fallback Result Status: SUCCESS")
        print(f"Provider Used: {provider_used}")
        print(f"Latency: {latency}s")
        print(f"Parsed Response: {res}")
        assert provider_used == "openrouter", f"Expected openrouter, got {provider_used}"
        print("Fallback to OpenRouter: PASS")

if __name__ == "__main__":
    asyncio.run(test_fallback())
