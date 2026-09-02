"""
LLM provider abstraction with automatic provider & model fallback chain & JSON extraction.

Primary: Gemini
Secondary: OpenRouter (with automatic free model retries)

Supports explicit separation of:
1. Normal text LLM calls (llm_complete)
2. JSON generation calls (llm_complete_json / llm_complete_json_with_provider)
3. Explicit multimodal VLM calls (llm_complete_multimodal)
"""
from __future__ import annotations
import asyncio
import json
import re
import os
from typing import Optional, Any, Tuple, Dict, List
import httpx
from app.core.config import settings

TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# Standard free model fallbacks on OpenRouter
OPENROUTER_FREE_MODELS = [
    "minimax/minimax-m3:free",
    "dots-studio/dots-3-note-preview:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "poolside/laguna-s-2.1:free",
    "nousresearch/deephermes-3-llama-3-1-8b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "openrouter/free",
]


class LLMError(Exception):
    pass


class VLMError(LLMError):
    pass


def is_gemini_configured() -> bool:
    key = settings.GEMINI_API_KEY.strip()
    return bool(key and len(key) >= 10)


def is_openrouter_configured() -> bool:
    key = settings.OPENROUTER_API_KEY.strip()
    return bool(key and len(key) >= 10)


async def _call_gemini_with_metadata(
    prompt: str,
    image_b64: Optional[str] = None,
    mime_type: str = "image/jpeg",
    purpose: str = "general",
) -> Tuple[str, str, int, str]:
    """Calls Gemini API, returning (text_out, model_name, retry_count, finish_reason)."""
    key = settings.GEMINI_API_KEY.strip()
    if not key or len(key) < 10:
        err_msg = "Gemini API key missing or invalid"
        print(f"[LLM ERROR] provider=gemini model=N/A status=401 error={err_msg} endpoint=generativelanguage.googleapis.com stage={purpose}")
        raise LLMError(err_msg)

    models_to_try = [
        settings.GEMINI_MODEL or "gemini-3.6-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
    ]
    seen = set()
    models = [m for m in models_to_try if not (m in seen or seen.add(m))]

    parts = [{"text": prompt}]
    image_present = bool(image_b64 and len(image_b64) > 0)
    image_bytes = len(image_b64) if image_present else 0

    if image_present:
        clean_b64 = image_b64.replace("\n", "").replace("\r", "").strip()
        parts.append({"inline_data": {"mime_type": mime_type, "data": clean_b64}})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"maxOutputTokens": 8192, "temperature": 0.1},
    }
    last_err = None
    total_retries = 0

    for model_name in models:
        print(
            f"[LLM CALL] purpose={purpose} provider=gemini model={model_name} "
            f"multimodal={image_present} image_present={image_present} "
            f"image_bytes={image_bytes} mime_type={mime_type if image_present else 'N/A'}"
        )
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}",
                        json=body,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        candidates = data.get("candidates", [])
                        finish_reason = "STOP"
                        if candidates and len(candidates) > 0:
                            finish_reason = candidates[0].get("finishReason", "STOP")
                            if "content" in candidates[0]:
                                parts_res = candidates[0]["content"].get("parts", [])
                                if parts_res and "text" in parts_res[0]:
                                    text_out = parts_res[0]["text"]
                                    if text_out and len(text_out.strip()) > 0:
                                        print(f"[LLM SUCCESS] purpose={purpose} provider=gemini model={model_name}")
                                        return text_out.strip(), model_name, total_retries, finish_reason

                    err_snippet = r.text[:300].replace("\n", " ").strip()
                    last_err = f"Status {r.status_code} - {err_snippet}"

                    if r.status_code in (400, 401, 403):
                        print(
                            f"[LLM ERROR] provider=gemini model={model_name} status={r.status_code} "
                            f"error={err_snippet[:150]} endpoint=generativelanguage.googleapis.com stage={purpose}"
                        )
                        raise LLMError(f"Gemini API authentication/permission error: {r.status_code} - {err_snippet}")

                    if r.status_code == 429:
                        err_lower = err_snippet.lower()
                        is_quota = any(q in err_lower for q in ["quota", "resource_exhausted", "free_tier", "daily_limit", "per_day"])
                        if is_quota:
                            print(
                                f"[LLM ERROR] provider=gemini model={model_name} status=429_QUOTA "
                                f"error={err_snippet[:150]} endpoint=generativelanguage.googleapis.com stage={purpose}"
                            )
                            # Project quota exhausted: do NOT cycle through remaining Gemini models sharing the same key!
                            raise LLMError(f"Gemini API quota exhausted: 429_QUOTA - {err_snippet}")
                        else:
                            print(
                                f"[LLM ERROR] provider=gemini model={model_name} status=429_RATE_LIMIT "
                                f"error={err_snippet[:150]} endpoint=generativelanguage.googleapis.com stage={purpose}"
                            )
                            if attempt < max_attempts - 1:
                                total_retries += 1
                                backoff = 1.5 * (attempt + 1)
                                print(f"[LLM RETRY] provider=gemini model={model_name} transient status=429_RATE_LIMIT. Retrying in {backoff:.1f}s (attempt {attempt + 1}/{max_attempts})...")
                                await asyncio.sleep(backoff)
                                continue

                    else:
                        print(
                            f"[LLM ERROR] provider=gemini model={model_name} status={r.status_code} "
                            f"error={err_snippet[:150]} endpoint=generativelanguage.googleapis.com stage={purpose}"
                        )

                    if r.status_code == 404:
                        break  # Move to next Gemini model immediately on 404

                    if r.status_code in TRANSIENT_STATUS:
                        if attempt < max_attempts - 1:
                            total_retries += 1
                            backoff = 1.5 * (attempt + 1)
                            print(f"[LLM RETRY] provider=gemini model={model_name} transient status={r.status_code}. Retrying in {backoff:.1f}s (attempt {attempt + 1}/{max_attempts})...")
                            await asyncio.sleep(backoff)
                            continue

            except LLMError:
                raise
            except Exception as e:
                err_str = str(e).split("\n")[0]
                last_err = err_str
                print(
                    f"[LLM ERROR] provider=gemini model={model_name} status=500 "
                    f"error={err_str[:150]} endpoint=generativelanguage.googleapis.com stage={purpose}"
                )
                if attempt < max_attempts - 1:
                    total_retries += 1
                    await asyncio.sleep(1.0)
                    continue

    raise LLMError(f"Gemini API failed across models: {last_err}")


async def _call_gemini(
    prompt: str,
    image_b64: Optional[str] = None,
    mime_type: str = "image/jpeg",
    purpose: str = "general",
) -> str:
    text, _, _, _ = await _call_gemini_with_metadata(prompt, image_b64, mime_type, purpose)
    return text


OPENROUTER_VISION_MODELS = [
    # Multimodal / Vision-capable free models on OpenRouter
    "minimax/minimax-m3:free",
    "dots-studio/dots-3-note-preview:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "thinkingmachines/inkling:free",
    "thinkingmachines/inkling-small:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free",
]


async def _call_openrouter_with_metadata(
    prompt: str,
    image_b64: Optional[str] = None,
    mime_type: str = "image/png",
    purpose: str = "general",
) -> Tuple[str, str, int, str]:
    """Calls OpenRouter API, returning (text_out, model_name, retry_count, finish_reason)."""
    key = settings.OPENROUTER_API_KEY.strip()
    if not key or len(key) < 10:
        err_msg = "OpenRouter API key missing or invalid"
        print(f"[LLM ERROR] provider=openrouter model=N/A status=401 error={err_msg} endpoint=openrouter.ai stage={purpose}")
        raise LLMError(err_msg)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://vedaai.app",
        "X-Title": "VedaAI Examiner Assistant",
        "Content-Type": "application/json",
    }

    image_present = bool(image_b64 and len(image_b64) > 0)
    image_bytes = len(image_b64) if image_present else 0

    content_payload: Any = prompt
    if image_present:
        clean_b64 = image_b64.replace("\n", "").replace("\r", "").strip()
        content_payload = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{clean_b64}"}},
        ]

    models_to_try = []

    if image_present:
        if settings.OPENROUTER_MODEL and settings.OPENROUTER_MODEL.strip():
            models_to_try.append(settings.OPENROUTER_MODEL.strip())
        for m in OPENROUTER_VISION_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)
    else:
        if settings.OPENROUTER_MODEL and settings.OPENROUTER_MODEL.strip():
            models_to_try.append(settings.OPENROUTER_MODEL.strip())
        for m in OPENROUTER_FREE_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

    last_err = None
    retries = 0
    max_tokens_val = 8192 if image_present else 4096

    for model in models_to_try:
        print(
            f"[LLM CALL] purpose={purpose} provider=openrouter model={model} "
            f"multimodal={image_present} image_present={image_present} "
            f"image_bytes={image_bytes} mime_type={mime_type if image_present else 'N/A'}"
        )
        try:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": content_payload}],
                "max_tokens": max_tokens_val,
            }
            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(url, json=body, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    choices = data.get("choices", [])
                    finish_reason = "STOP"
                    if choices:
                        finish_reason = choices[0].get("finish_reason", "STOP").upper()
                        if "message" in choices[0]:
                            content = choices[0]["message"].get("content", "")
                            if content and len(content.strip()) > 0:
                                print(f"[LLM SUCCESS] purpose={purpose} provider=openrouter model={model}")
                                return content.strip(), model, retries, finish_reason

                err_snippet = r.text[:200].replace("\n", " ").strip()
                last_err = f"{model} status {r.status_code} - {err_snippet}"
                print(
                    f"[LLM ERROR] provider=openrouter model={model} status={r.status_code} "
                    f"error={err_snippet[:150]} endpoint=openrouter.ai stage={purpose}"
                )
                retries += 1
        except Exception as e:
            err_str = str(e).split("\n")[0]
            last_err = f"{model} exception: {err_str}"
            print(
                f"[LLM ERROR] provider=openrouter model={model} status=500 "
                f"error={err_str[:150]} endpoint=openrouter.ai stage={purpose}"
            )
            retries += 1
            continue

    raise LLMError(f"OpenRouter all candidate models failed: {last_err}")


async def _call_openrouter(
    prompt: str,
    image_b64: Optional[str] = None,
    mime_type: str = "image/png",
    purpose: str = "general",
) -> str:
    text, _, _, _ = await _call_openrouter_with_metadata(prompt, image_b64, mime_type, purpose)
    return text


_PROVIDERS = {
    "gemini": _call_gemini,
    "openrouter": _call_openrouter,
}

_ORDER = ["gemini", "openrouter"]
_llm_sem = asyncio.Semaphore(4)


async def llm_complete(prompt: str, allow_fallback: bool = True, purpose: str = "general") -> str:
    """Executes prompt across Gemini -> OpenRouter provider chain (Text-only)."""
    async with _llm_sem:
        last_errs = []
        for name in _ORDER:
            fn = _PROVIDERS.get(name)
            if not fn:
                continue
            try:
                res = await asyncio.wait_for(fn(prompt, purpose=purpose), timeout=45.0)
                if res and len(res.strip()) > 0:
                    return res
            except Exception as e:
                err_msg = str(e).split("\n")[0]
                last_errs.append(f"{name}: {err_msg}")
                print(f"[LLM FALLBACK] Provider '{name}' failed for purpose '{purpose}'. Trying next provider...")
                continue

        combined_err = " | ".join(last_errs)
        raise LLMError(f"All configured LLM providers (Gemini -> OpenRouter) failed: {combined_err}")


async def llm_complete_multimodal_with_metadata(
    prompt: str,
    image_b64: str,
    mime_type: str = "image/png",
    purpose: str = "document_vision",
) -> Tuple[str, Dict[str, Any]]:
    """
    Explicit multimodal interface for VLM / Document Vision requests with metadata tracking.
    Enforces that image_b64 is non-empty and every provider attempt remains multimodal.
    NO SILENT DOWNGRADE TO TEXT-ONLY IS ALLOWED.
    """
    if not image_b64 or len(image_b64.strip()) == 0:
        raise VLMError("Multimodal interface requires a valid, non-empty base64 image payload.")

    async with _llm_sem:
        last_errs = []

        # Attempt Gemini
        if "gemini" in _PROVIDERS and is_gemini_configured():
            try:
                res_text, model_used, retries, finish_reason = await asyncio.wait_for(
                    _call_gemini_with_metadata(prompt, image_b64=image_b64, mime_type=mime_type, purpose=purpose),
                    timeout=60.0,
                )
                if res_text and len(res_text.strip()) > 0:
                    source = "VLM_RETRY_SUCCESS" if retries > 0 else "VLM_SUCCESS"
                    meta = {
                        "provider": "gemini",
                        "model": model_used,
                        "vlm_result": "SUCCESS",
                        "finish_reason": finish_reason,
                        "retry_count": retries,
                        "fallback_used": False,
                        "fallback_provider": "N/A",
                        "structure_source": source,
                    }
                    return res_text, meta
            except Exception as e:
                err_msg = str(e).split("\n")[0]
                last_errs.append(f"gemini: {err_msg}")
                print(f"[LLM FALLBACK] Multimodal provider 'gemini' failed for purpose '{purpose}'. Trying OpenRouter...")

        # Attempt OpenRouter
        if "openrouter" in _PROVIDERS and is_openrouter_configured():
            try:
                res_text, model_used, retries, finish_reason = await asyncio.wait_for(
                    _call_openrouter_with_metadata(prompt, image_b64=image_b64, mime_type=mime_type, purpose=purpose),
                    timeout=60.0,
                )
                if res_text and len(res_text.strip()) > 0:
                    meta = {
                        "provider": "openrouter",
                        "model": model_used,
                        "vlm_result": "SUCCESS",
                        "finish_reason": finish_reason,
                        "retry_count": retries,
                        "fallback_used": True,
                        "fallback_provider": "openrouter",
                        "structure_source": "OPENROUTER_VLM_SUCCESS",
                    }
                    return res_text, meta
            except Exception as e:
                err_msg = str(e).split("\n")[0]
                last_errs.append(f"openrouter: {err_msg}")

        combined_err = " | ".join(last_errs)
        raise VLMError(f"No configured vision-capable provider succeeded: {combined_err}")


async def llm_complete_multimodal(
    prompt: str,
    image_b64: str,
    mime_type: str = "image/png",
    purpose: str = "document_vision",
) -> str:
    text, _ = await llm_complete_multimodal_with_metadata(prompt, image_b64, mime_type, purpose)
    return text



def _repair_truncated_json(json_str: str) -> Any:
    """Attempts to repair and parse JSON truncated mid-output by closing open brackets/braces."""
    s = json_str.strip()
    if not s:
        raise ValueError("Empty string for JSON repair")

    for end_idx in range(len(s), max(0, len(s) - 500), -1):
        sub = s[:end_idx].strip()
        if sub.endswith(","):
            sub = sub[:-1].strip()
        open_braces = sub.count("{") - sub.count("}")
        open_brackets = sub.count("[") - sub.count("]")
        candidate = sub + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("Truncated JSON repair exhausted")


def extract_json_payload(text: str) -> Any:
    """Extracts JSON payload from raw LLM output, handling codeblocks and truncated outputs."""
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match_obj = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match_obj:
        try:
            return json.loads(match_obj.group(0))
        except Exception:
            pass

    match_arr = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match_arr:
        try:
            return json.loads(match_arr.group(0))
        except Exception:
            pass

    try:
        return _repair_truncated_json(cleaned)
    except Exception:
        pass

    raise ValueError(f"Could not parse valid JSON from LLM response: {text[:200]}")


async def _execute_provider_chain(prompt: str, purpose: str = "json_generation") -> Tuple[Any, str]:
    async with _llm_sem:
        last_errs = []
        for name in _ORDER:
            fn = _PROVIDERS.get(name)
            if not fn:
                continue
            try:
                raw = await fn(prompt, purpose=purpose)
                if raw and len(raw.strip()) > 0:
                    parsed = extract_json_payload(raw)
                    return parsed, name
            except Exception as e:
                err_msg = str(e).split("\n")[0]
                last_errs.append(f"{name}: {err_msg}")
                print(f"[LLM FALLBACK] Provider '{name}' failed for purpose '{purpose}'. Trying next provider...")
                continue

        combined_err = " | ".join(last_errs)
        raise LLMError(f"All configured LLM providers (Gemini -> OpenRouter) failed: {combined_err}")


async def llm_complete_json_with_provider(prompt: str, timeout: float = 45.0, purpose: str = "json_generation") -> Tuple[Any, str]:
    """
    Invokes LLM provider chain cleanly behind single interface.
    Returns (parsed_json, provider_name).
    """
    return await asyncio.wait_for(_execute_provider_chain(prompt, purpose=purpose), timeout=timeout)


async def llm_complete_json(prompt: str, timeout: float = 45.0, purpose: str = "json_generation") -> Any:
    """Invokes LLM and parses JSON output safely."""
    data, _ = await llm_complete_json_with_provider(prompt, timeout=timeout, purpose=purpose)
    return data
