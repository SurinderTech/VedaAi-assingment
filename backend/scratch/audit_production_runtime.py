"""
Detailed Production Pipeline & VLM Runtime Trace for Meghalaya Examination PDF.
Instruments and prints exact runtime values requested in the diagnostic mandate.
"""
import sys
import os
import io
import json
import httpx
import asyncio
from pathlib import Path

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.services.document_processor import process_document
from app.services.document_vision_provider import MultimodalDocumentVisionProvider
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService
from app.services.llm_provider import is_gemini_configured, is_openrouter_configured

pdf_path = Path("C:/Users/surin/AppData/Local/Temp/vedaai_uploads/8a27cbab5a31/question_paper.pdf")
if not pdf_path.exists():
    pdf_path = Path(__file__).parent / "test_corpus" / "multi_page_paper.pdf"

print("="*80)
print("PRODUCTION PATH RUNTIME TRACE REPORT")
print("="*80)

# 1. Configuration Audit
print("\n[1] CONFIGURATION AUDIT:")
print(f"  DOCUMENT_VLM_ENABLED (settings):            {settings.DOCUMENT_VLM_ENABLED}")
print(f"  DOCUMENT_VLM_PAGE_UNDERSTANDING (settings): {settings.DOCUMENT_VLM_PAGE_UNDERSTANDING}")
print(f"  PRIMARY_LLM_PROVIDER (settings):            {settings.PRIMARY_LLM_PROVIDER}")
print(f"  DOCUMENT_VLM_PROVIDER (settings):           {settings.DOCUMENT_VLM_PROVIDER}")
print(f"  DOCUMENT_VLM_MODEL (settings):              {settings.DOCUMENT_VLM_MODEL}")
print(f"  GEMINI_API_KEY present:                     {bool(settings.GEMINI_API_KEY)}")
print(f"  GEMINI_API_KEY length:                      {len(settings.GEMINI_API_KEY)}")
print(f"  GEMINI_API_KEY prefix:                      {settings.GEMINI_API_KEY[:6]}...")
print(f"  OPENROUTER_API_KEY present:                 {bool(settings.OPENROUTER_API_KEY)}")
print(f"  OPENROUTER_API_KEY length:                  {len(settings.OPENROUTER_API_KEY)}")
print(f"  OPENROUTER_API_KEY prefix:                  {settings.OPENROUTER_API_KEY[:10]}...")
print(f"  is_gemini_configured():                     {is_gemini_configured()}")
print(f"  is_openrouter_configured():                 {is_openrouter_configured()}")

# 2. Document Rendering
print("\n[2] DOCUMENT RENDERING:")
blocks, num_pages, page_sizes, page_images = process_document(str(pdf_path), ".pdf", force_ocr=False)
print(f"  Total pages rendered:                       {num_pages}")
print(f"  Total page images in dict:                  {len(page_images) if page_images else 0}")
print(f"  Total OCR blocks extracted:                 {len(blocks)}")

# 3. Direct Gemini API Authentication Probe
print("\n[3] DIRECT GEMINI API AUTHENTICATION PROBE:")
gemini_key = settings.GEMINI_API_KEY.strip()
probe_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
async def probe_gemini():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                probe_url,
                json={"contents": [{"parts": [{"text": "ping"}]}]}
            )
            print(f"  Probe HTTP Status:                          {r.status_code}")
            print(f"  Probe Response:                             {r.text[:200]}")
            return r.status_code
    except Exception as e:
        print(f"  Probe Exception:                            {e}")
        return 500

status_code = asyncio.run(probe_gemini())

# 4. Multimodal Vision Provider State
vision_provider = MultimodalDocumentVisionProvider()
print("\n[4] MULTIMODAL VISION PROVIDER STATE:")
print(f"  vision_provider.is_configured():            {vision_provider.is_configured()}")
print(f"  vision_provider.model_name:                 {vision_provider.model_name}")

print("\n" + "="*80)
