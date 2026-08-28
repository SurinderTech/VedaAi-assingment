import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from app.services.llm_provider import _call_gemini, _call_openrouter, _call_xai_grok, _call_groq, LLMError
from app.core.config import settings

async def main():
    print(f"PRIMARY_LLM_PROVIDER: {settings.PRIMARY_LLM_PROVIDER}")
    print(f"GEMINI_API_KEY: {'configured (' + settings.GEMINI_API_KEY[:8] + '...)' if settings.GEMINI_API_KEY else 'EMPTY'}")
    print(f"XAI_API_KEY: {'configured (' + settings.XAI_API_KEY[:8] + '...)' if settings.XAI_API_KEY else 'EMPTY'}")
    print(f"OPENROUTER_API_KEY: {'configured (' + settings.OPENROUTER_API_KEY[:8] + '...)' if settings.OPENROUTER_API_KEY else 'EMPTY'}")
    print(f"GROQ_API_KEY: {'configured (' + settings.GROQ_API_KEY[:8] + '...)' if settings.GROQ_API_KEY else 'EMPTY'}")
    
    prompt = 'Return JSON: {"status": "ok"}'
    
    for name, fn in [("gemini", _call_gemini), ("xai", _call_xai_grok), ("openrouter", _call_openrouter), ("groq", _call_groq)]:
        try:
            res = await fn(prompt)
            print(f"[{name}] SUCCESS: {res[:100]}")
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__} -> {e}")

if __name__ == "__main__":
    asyncio.run(main())
