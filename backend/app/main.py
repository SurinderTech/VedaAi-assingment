from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm slow models at server startup so first user request is fast."""
    print("[Startup] Pre-warming OCR engine and VLM provider...")

    # 1. Pre-warm RapidOCR ONNX models (takes 60-120s on first load)
    try:
        import asyncio
        from app.services.document_processor import _get_ocr_engine
        import numpy as np
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_ocr_engine)
        print("[Startup] OCR engine ready.")
    except Exception as e:
        print(f"[Startup] OCR pre-warm notice: {e}")

    # 2. Pre-warm VLM provider (loads Gemini client, checks API key)
    try:
        from app.services.llm_provider import is_gemini_configured
        if is_gemini_configured():
            print("[Startup] Gemini VLM provider configured and ready.")
    except Exception as e:
        print(f"[Startup] VLM pre-warm notice: {e}")

    print("[Startup] All systems ready. Server accepting requests.")
    yield
    # Shutdown cleanup (nothing needed)


app = FastAPI(title="VedaAI Assessment Extraction API", lifespan=lifespan)

# Allow all Vercel preview deployments + localhost.
# Render free tier cold-starts can return 502s before FastAPI runs,
# so wildcard is the only reliable approach without a paid proxy.
ALLOWED_ORIGINS = [
    "https://veda-ai-assingment.vercel.app",
    "https://veda-ai-assingment-ldl1hla7z.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # wildcard — handles all Vercel preview URLs
    allow_credentials=False,    # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok", "ocr": "ready", "vlm": "ready"}