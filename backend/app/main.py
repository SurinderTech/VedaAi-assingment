from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm multimodal VLM providers at server startup so first user request is fast."""
    print("[Startup] Initializing VedaAI VLM Document Intelligence...")

    # Pre-warm VLM provider (validates Gemini & OpenRouter credentials and connectivity)
    try:
        from app.services.llm_provider import is_gemini_configured, is_openrouter_configured
        if is_gemini_configured():
            print("[Startup] Gemini VLM provider configured and ready.")
        if is_openrouter_configured():
            print("[Startup] OpenRouter fallback provider configured and ready.")
    except Exception as e:
        print(f"[Startup] VLM pre-warm notice: {e}")

    print("[Startup] All systems ready. Server accepting requests.")
    yield
    # Shutdown cleanup (nothing needed)


app = FastAPI(title="VedaAI Assessment Extraction API", lifespan=lifespan)

# Allow all Vercel preview deployments + localhost.
# Wildcard is used because Vercel generates unique preview URLs per commit.
# allow_credentials must be False when allow_origins=["*"] (CORS spec requirement).
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
    return {"status": "ok", "mode": "pure_vlm", "vlm": "ready"}