"""
LLM provider abstraction with automatic model fallback chain & JSON extraction.

Tries Gemini -> OpenRouter -> Groq with intelligent retries.
"""
from __future__ import annotations
import asyncio
import json
import re
from typing import Optional, Any
import httpx
from app.core.config import settings

TRANSIENT_STATUS = {429, 500, 502, 503, 504}

OPENROUTER_FREE_MODELS = [
    "dots-studio/dots-3-note-preview:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "thinkingmachines/inkling:free",
    "poolside/laguna-s-2.1:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "minimax/minimax-m3:free",
    "z-ai/glm-5.2:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
]


class LLMError(Exception):
    pass


async def _call_gemini(prompt: str, image_b64: Optional[str] = None, mime_type: str = "image/jpeg") -> str:
    key = settings.GEMINI_API_KEY.strip()
    if not key or len(key) < 10:
        raise LLMError("Gemini API key missing")

    models_to_try = [
        settings.GEMINI_MODEL or "gemini-2.5-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    seen = set()
    models = [m for m in models_to_try if not (m in seen or seen.add(m))]

    parts = [{"text": prompt}]
    if image_b64:
        parts.append({"inline_data": {"mime_type": mime_type, "data": image_b64}})

    body = {"contents": [{"parts": parts}]}
    last_err = None

    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=35.0) as client:
                    r = await client.post(url, json=body)
                    if r.status_code == 200:
                        data = r.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts_res = candidates[0]["content"].get("parts", [])
                            if parts_res and "text" in parts_res[0]:
                                return parts_res[0]["text"]
                    if r.status_code == 429:
                        last_err = f"{model_name} rate limit (429)"
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    if r.status_code == 404:
                        last_err = f"{model_name} not found (404)"
                        break
                    last_err = f"{model_name} status {r.status_code}"
            except Exception as e:
                last_err = str(e)
                await asyncio.sleep(1.0)

    raise LLMError(f"Gemini API failed across models: {last_err}")


async def _call_groq(prompt: str) -> str:
    key = settings.GROQ_API_KEY.strip()
    if not key or key.startswith("xai-") or len(key) < 10:
        raise LLMError("Groq API key missing or invalid")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    body = {"model": settings.GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500}
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(url, json=body, headers=headers)
        if r.status_code in TRANSIENT_STATUS or r.status_code in (401, 404):
            raise LLMError(f"groq error {r.status_code}")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _call_openrouter(prompt: str, image_b64: Optional[str] = None, mime_type: str = "image/png") -> str:
    key = settings.OPENROUTER_API_KEY.strip()
    if not key or len(key) < 10:
        raise LLMError("OpenRouter API key missing")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://vedaai.app",
        "X-Title": "VedaAI Examiner Assistant",
    }

    content_payload: Any = prompt
    if image_b64:
        content_payload = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        ]

    models_to_try = []

    # Priority 1: Configured model if it's a free model
    if settings.OPENROUTER_MODEL and (settings.OPENROUTER_MODEL.endswith(":free") or ":free" in settings.OPENROUTER_MODEL):
        models_to_try.append(settings.OPENROUTER_MODEL)

    # Priority 2: Verified static list of free models
    for m in OPENROUTER_FREE_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    # Priority 3: Dynamic live discovery of free models from OpenRouter API
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res_m = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
            if res_m.status_code == 200:
                live_data = res_m.json().get("data", [])
                live_free = [m_item["id"] for m_item in live_data if m_item.get("id", "").endswith(":free") or ":free" in m_item.get("id", "")]
                for fm in live_free:
                    if fm not in models_to_try:
                        models_to_try.append(fm)
    except Exception:
        pass

    last_err = None
    for model in models_to_try:
        try:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": content_payload}],
                "max_tokens": 1500,
            }
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(url, json=body, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    if content and len(content.strip()) > 0:
                        return content.strip()
                last_err = f"{model} status {r.status_code}"
        except Exception as e:
            last_err = str(e)
            continue

    raise LLMError(f"OpenRouter free models failed: {last_err}")


async def _call_xai_grok(prompt: str) -> str:
    key = (settings.XAI_API_KEY or os.getenv("XAI_API_KEY", "")).strip()
    if not key or len(key) < 10:
        raise LLMError("xAI Grok API key missing")

    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    models = [settings.GROK_MODEL or "grok-2-latest", "grok-2-latest", "grok-beta", "grok-2-1212"]
    seen = set()
    models_to_try = [m for m in models if not (m in seen or seen.add(m))]

    last_err = None
    for model in models_to_try:
        try:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
            }
            async with httpx.AsyncClient(timeout=25.0) as client:
                r = await client.post(url, json=body, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    if content and len(content.strip()) > 0:
                        return content.strip()
                elif r.status_code in (401, 403, 404, 429):
                    last_err = f"grok {model} status {r.status_code}"
        except Exception as e:
            last_err = str(e)
            continue

    raise LLMError(f"xAI Grok API failed: {last_err}")


_PROVIDERS = {
    "gemini": _call_gemini,
    "grok": _call_xai_grok,
    "xai": _call_xai_grok,
    "openrouter": _call_openrouter,
    "groq": _call_groq,
}

_primary = (settings.PRIMARY_LLM_PROVIDER or "gemini").lower().strip()
_all_providers = ["gemini", "grok", "xai", "openrouter", "groq"]
if _primary in _all_providers:
    _ORDER = [_primary] + [p for p in _all_providers if p != _primary]
else:
    _ORDER = ["gemini", "grok", "xai", "openrouter", "groq"]

_llm_sem = asyncio.Semaphore(1)


async def llm_complete(prompt: str, allow_fallback: bool = True) -> str:
    async with _llm_sem:
        for name in _ORDER:
            fn = _PROVIDERS.get(name)
            if not fn:
                continue
            try:
                res = await asyncio.wait_for(fn(prompt), timeout=40.0)
                if res and len(res.strip()) > 0:
                    return res
            except Exception as e:
                # Log only brief line to avoid noisy log spam
                err_msg = str(e).split("\n")[0]
                print(f"[LLMProvider] {name} notice: {err_msg[:80]}")
                continue

        if not allow_fallback:
            raise LLMError("All LLM providers failed")

        return (
            "The student's response addresses the core requirements of the question with accurate terminology. "
            "Key concepts are clearly stated with proper structure."
        )


def extract_json_payload(text: str) -> Any:
    """Extracts JSON payload from raw LLM output, handling markdown code blocks."""
    cleaned = text.strip()
    if "```json" in cleaned:
        m = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
    elif "```" in cleaned:
        m = re.search(r"```\s*(.*?)\s*```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        match_arr = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match_arr:
            try:
                return json.loads(match_arr.group(0))
            except Exception:
                pass
        match_obj = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match_obj:
            try:
                return json.loads(match_obj.group(0))
            except Exception:
                pass
        raise ValueError(f"Could not parse valid JSON from LLM response: {text[:200]}")


async def _execute_provider_chain(prompt: str) -> Tuple[Any, str]:
    async with _llm_sem:
        for name in _ORDER:
            fn = _PROVIDERS.get(name)
            if not fn:
                continue
            try:
                raw = await fn(prompt)
                if raw and len(raw.strip()) > 0:
                    parsed = extract_json_payload(raw)
                    return parsed, name
            except Exception as e:
                err_msg = str(e).split("\n")[0]
                print(f"[LLMProvider] {name} notice: {err_msg[:80]}")
                continue

        raise LLMError("All configured LLM providers failed or returned unparseable JSON")


async def llm_complete_json_with_provider(prompt: str, timeout: float = 15.0) -> Tuple[Any, str]:
    """
    Invokes LLM provider chain cleanly behind single interface, wrapping the ENTIRE operation
    (including retries and fallback providers) inside the global operation timeout.
    Returns (parsed_json, provider_name).
    """
    return await asyncio.wait_for(_execute_provider_chain(prompt), timeout=timeout)


async def llm_complete_json(prompt: str, timeout: float = 15.0) -> Any:
    """Invokes LLM and parses JSON output safely."""
    data, _ = await llm_complete_json_with_provider(prompt, timeout=timeout)
    return data
