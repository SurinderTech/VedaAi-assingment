import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core import config
from app.services import llm_provider
from app.services.llm_provider import _call_gemini, _call_openrouter, is_gemini_configured, is_openrouter_configured, LLMError

async def test_gemini_standalone():
    print("\n================================================================================")
    print(" 1. STANDALONE GEMINI PROVIDER TEST")
    print("================================================================================")
    configured = is_gemini_configured()
    print(f"Gemini Configured: {configured}")
    print(f"Default Model Configured: {config.settings.GEMINI_MODEL}")
    if not configured:
        print("RESULT: FAILED (API Key Not Configured)")
        return False

    start = time.time()
    try:
        res = await _call_gemini("What is 2 + 2? Reply in one word.", purpose="provider_diagnostic")
        latency = round(time.time() - start, 3)
        print(f"Request Status: SUCCESS")
        print(f"Latency: {latency}s")
        print(f"Response Received: '{res.strip()[:100]}'")
        return True
    except Exception as e:
        latency = round(time.time() - start, 3)
        print(f"Request Status: FAILED")
        print(f"Latency: {latency}s")
        print(f"Actual Error: {e}")
        return False

async def test_openrouter_standalone():
    print("\n================================================================================")
    print(" 2. STANDALONE OPENROUTER PROVIDER TEST")
    print("================================================================================")
    configured = is_openrouter_configured()
    print(f"OpenRouter Configured: {configured}")
    print(f"Default Model Configured: {config.settings.OPENROUTER_MODEL}")
    if not configured:
        print("RESULT: FAILED (API Key Not Configured)")
        return False

    start = time.time()
    try:
        res = await _call_openrouter("What is 2 + 2? Reply in one word.", purpose="provider_diagnostic")
        latency = round(time.time() - start, 3)
        print(f"Request Status: SUCCESS")
        print(f"Latency: {latency}s")
        print(f"Response Received: '{res.strip()[:100]}'")
        return True
    except Exception as e:
        latency = round(time.time() - start, 3)
        print(f"Request Status: FAILED")
        print(f"Latency: {latency}s")
        print(f"Actual Error: {e}")
        return False

async def test_provider_chain():
    print("\n================================================================================")
    print(" 3. COMPLETE PROVIDER CHAIN TEST (GEMINI -> OPENROUTER FALLBACK)")
    print("================================================================================")
    start = time.time()
    try:
        res, provider_used = await llm_provider.llm_complete_json_with_provider(
            '{"question": "2+2", "answer": 4}', purpose="chain_diagnostic"
        )
        latency = round(time.time() - start, 3)
        print(f"Chain Status: SUCCESS")
        print(f"Provider Used: {provider_used}")
        print(f"Latency: {latency}s")
        print(f"Parsed JSON Result: {res}")
        return True
    except Exception as e:
        latency = round(time.time() - start, 3)
        print(f"Chain Status: FAILED")
        print(f"Latency: {latency}s")
        print(f"Actual Error: {e}")
        return False

async def main():
    print("\n================================================================================")
    print(" VEDAAI LLM PROVIDER DIAGNOSTIC (GEMINI & OPENROUTER)")
    print("================================================================================")

    g_ok = await test_gemini_standalone()
    or_ok = await test_openrouter_standalone()
    chain_ok = await test_provider_chain()

    print("\n================================================================================")
    print(" DIAGNOSTIC SUMMARY")
    print("================================================================================")
    print(f"Gemini Standalone:      {'PASS' if g_ok else 'FAIL'}")
    print(f"OpenRouter Standalone:  {'PASS' if or_ok else 'FAIL'}")
    print(f"Provider Chain:         {'PASS' if chain_ok else 'FAIL'}")
    print("================================================================================\n")

if __name__ == "__main__":
    asyncio.run(main())
