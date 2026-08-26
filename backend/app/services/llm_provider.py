"""
LLM provider abstraction (plan sections 31-34).

The rest of the app calls `llm_complete(prompt)` and never knows which
provider answered. Retries only on transient failures (429/5xx/timeout);
invalid-key/invalid-request errors are not retried.
"""
from __future__ import annotations
import httpx
from app.core.config import settings

TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


async def _call_gemini(prompt: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
    )
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=body)
        if r.status_code in TRANSIENT_STATUS:
            raise LLMError(f"gemini transient {r.status_code}")
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def _call_groq(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
    body = {"model": settings.GROQ_MODEL, "messages": [{"role": "user", "content": prompt}]}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=body, headers=headers)
        if r.status_code in TRANSIENT_STATUS:
            raise LLMError(f"groq transient {r.status_code}")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _call_openrouter(prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}
    body = {"model": settings.OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}]}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=body, headers=headers)
        if r.status_code in TRANSIENT_STATUS:
            raise LLMError(f"openrouter transient {r.status_code}")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


_PROVIDERS = {"gemini": _call_gemini, "groq": _call_groq, "openrouter": _call_openrouter}
_ORDER = ["gemini", "groq", "openrouter"]


async def llm_complete(prompt: str) -> str:
    order = [settings.PRIMARY_LLM_PROVIDER] + [p for p in _ORDER if p != settings.PRIMARY_LLM_PROVIDER]
    last_err = None
    for name in order:
        fn = _PROVIDERS.get(name)
        if not fn:
            continue
        try:
            return await fn(prompt)
        except Exception as e:  # noqa: BLE001 - deliberately broad, we fall through
            last_err = e
            continue
    raise LLMError(f"All providers failed. Last error: {last_err}")
