import asyncio
import time
import base64
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core import config
from app.services import llm_provider
from app.services.llm_provider import (
    llm_complete,
    llm_complete_json,
    llm_complete_json_with_provider,
    llm_complete_multimodal,
    LLMError,
    VLMError,
)

# 1x1 Red Pixel PNG base64 for test payload
DUMMY_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

async def test_normal_llm():
    print("\n--------------------------------------------------------------------------------")
    print(" 1. NORMAL TEXT LLM PATH (llm_complete)")
    print("--------------------------------------------------------------------------------")
    start = time.time()
    res = await llm_complete("What is 1 + 1? Reply in one word.", purpose="general")
    latency = round(time.time() - start, 3)
    print(f"Text LLM Status: SUCCESS | Latency: {latency}s | Response: '{res.strip()[:60]}'")
    assert len(res.strip()) > 0
    return True

async def test_json_llm():
    print("\n--------------------------------------------------------------------------------")
    print(" 2. JSON LLM PATH (llm_complete_json_with_provider)")
    print("--------------------------------------------------------------------------------")
    start = time.time()
    parsed, provider = await llm_complete_json_with_provider(
        'Return valid JSON: {"status": "ok", "count": 42}', purpose="json_generation"
    )
    latency = round(time.time() - start, 3)
    print(f"JSON LLM Status: SUCCESS | Provider: {provider} | Latency: {latency}s | Result: {parsed}")
    assert isinstance(parsed, dict) and parsed.get("status") == "ok"
    return True

async def test_multimodal_vlm():
    print("\n--------------------------------------------------------------------------------")
    print(" 3. MULTIMODAL VLM PATH (llm_complete_multimodal)")
    print("--------------------------------------------------------------------------------")
    img_bytes = len(DUMMY_IMAGE_B64)
    print(f"Sending real image payload: image_bytes={img_bytes}, mime_type=image/png")
    start = time.time()
    res = await llm_complete_multimodal(
        prompt="Describe what color or image you see in one word.",
        image_b64=DUMMY_IMAGE_B64,
        mime_type="image/png",
        purpose="document_vision",
    )
    latency = round(time.time() - start, 3)
    print(f"Multimodal VLM Status: SUCCESS | Latency: {latency}s | Response: '{res.strip()[:60]}'")
    assert len(res.strip()) > 0
    return True

async def test_downgrade_prevention():
    print("\n--------------------------------------------------------------------------------")
    print(" 4. TEXT-ONLY DOWNGRADE PREVENTION TEST (VLMError Enforcement)")
    print("--------------------------------------------------------------------------------")
    
    async def mock_failing_multimodal(*args, **kwargs):
        raise LLMError("Simulated Multimodal Vision Failure")

    with patch.dict(llm_provider._PROVIDERS, {"gemini": mock_failing_multimodal, "openrouter": mock_failing_multimodal}):
        try:
            await llm_complete_multimodal(
                prompt="Analyze image",
                image_b64=DUMMY_IMAGE_B64,
                mime_type="image/png",
                purpose="document_vision",
            )
            print("FAILED: Expected VLMError, but call succeeded!")
            return False
        except VLMError as ve:
            print(f"SUCCESS: Caught expected VLMError -> '{ve}'")
            print("Text-only downgrade prevention: VERIFIED")
            return True
        except Exception as e:
            print(f"FAILED: Expected VLMError, got {type(e)} -> {e}")
            return False

async def main():
    print("\n================================================================================")
    print(" VEDAAI LLM/VLM PROVIDER BOUNDARY ACCEPTANCE SUITE")
    print("================================================================================")

    t_ok = await test_normal_llm()
    j_ok = await test_json_llm()
    v_ok = await test_multimodal_vlm()
    d_ok = await test_downgrade_prevention()

    print("\n================================================================================")
    print(" PROVIDER BOUNDARY TEST SUMMARY")
    print("================================================================================")
    print(f"NORMAL LLM PATH:        {'PASS' if t_ok else 'FAIL'}")
    print(f"JSON LLM PATH:          {'PASS' if j_ok else 'FAIL'}")
    print(f"MULTIMODAL VLM PATH:    {'PASS' if v_ok else 'FAIL'}")
    print(f"PREVENT TEXT DOWNGRADE: {'PASS' if d_ok else 'FAIL'}")
    print("================================================================================\n")

if __name__ == "__main__":
    asyncio.run(main())
