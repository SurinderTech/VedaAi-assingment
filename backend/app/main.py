from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(title="VedaAI Assessment Extraction API")

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
    return {"status": "ok"}