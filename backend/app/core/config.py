import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-2-latest")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3:free")

    PRIMARY_LLM_PROVIDER: str = os.getenv("PRIMARY_LLM_PROVIDER", "gemini")

    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

    # Mapping thresholds
    HIGH_CONFIDENCE: float = 0.85
    MEDIUM_CONFIDENCE: float = 0.55

    # Question extraction confidence thresholds (configurable via env / settings)
    QP_HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("QP_HIGH_CONFIDENCE_THRESHOLD", "0.85"))
    QP_LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("QP_LOW_CONFIDENCE_THRESHOLD", "0.20"))

    # Step 3 Intelligent Mapping Engine Weights & Thresholds (Initial Defaults)
    MAPPING_ANCHOR_WEIGHT: float = float(os.getenv("MAPPING_ANCHOR_WEIGHT", "0.40"))
    MAPPING_SEMANTIC_WEIGHT: float = float(os.getenv("MAPPING_SEMANTIC_WEIGHT", "0.30"))
    MAPPING_STRUCTURAL_WEIGHT: float = float(os.getenv("MAPPING_STRUCTURAL_WEIGHT", "0.15"))
    MAPPING_SPATIAL_WEIGHT: float = float(os.getenv("MAPPING_SPATIAL_WEIGHT", "0.10"))
    MAPPING_ORDER_WEIGHT: float = float(os.getenv("MAPPING_ORDER_WEIGHT", "0.05"))
    MAPPING_HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("MAPPING_HIGH_CONFIDENCE_THRESHOLD", "0.70"))
    MAPPING_REVIEW_THRESHOLD: float = float(os.getenv("MAPPING_REVIEW_THRESHOLD", "0.35"))
    MAPPING_AMBIGUITY_DELTA: float = float(os.getenv("MAPPING_AMBIGUITY_DELTA", "0.10"))
    # Step 4 Answer Evaluation & Grading Thresholds
    GRADING_HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("GRADING_HIGH_CONFIDENCE_THRESHOLD", "0.85"))
    GRADING_REVIEW_THRESHOLD: float = float(os.getenv("GRADING_REVIEW_THRESHOLD", "0.55"))
    GRADING_CONTRADICTION_THRESHOLD: float = float(os.getenv("GRADING_CONTRADICTION_THRESHOLD", "0.70"))

    # Configurable LLM Routing & Evaluation Settings
    GRADING_LLM_ENABLED: bool = os.getenv("GRADING_LLM_ENABLED", "true").lower() == "true"
    GRADING_LLM_AMBIGUITY_THRESHOLD: float = float(os.getenv("GRADING_LLM_AMBIGUITY_THRESHOLD", "0.65"))
    GRADING_LLM_CONFIDENCE_THRESHOLD: float = float(os.getenv("GRADING_LLM_CONFIDENCE_THRESHOLD", "0.85"))
    GRADING_LLM_MAX_TOKENS: int = int(os.getenv("GRADING_LLM_MAX_TOKENS", "500"))
    GRADING_LLM_TIMEOUT_SECONDS: float = float(os.getenv("GRADING_LLM_TIMEOUT_SECONDS", "15.0"))
    GRADING_LLM_MAX_CALLS_PER_DOCUMENT: int = int(os.getenv("GRADING_LLM_MAX_CALLS_PER_DOCUMENT", "20"))

    # Step 8 Embedding & Semantic Intelligence Settings
    EMBEDDING_ENGINE_ENABLED: bool = os.getenv("EMBEDDING_ENGINE_ENABLED", "true").lower() == "true"
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_NORMALIZE: bool = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
    EMBEDDING_TOP_K: int = int(os.getenv("EMBEDDING_TOP_K", "5"))
    EMBEDDING_CACHE_ENABLED: bool = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    SEMANTIC_AMBIGUITY_MARGIN: float = float(os.getenv("SEMANTIC_AMBIGUITY_MARGIN", "0.05"))

    # Step 11B Visual Document Verification Settings
    DOCUMENT_VLM_ENABLED: bool = os.getenv("DOCUMENT_VLM_ENABLED", "false").lower() == "true"
    DOCUMENT_VLM_PROVIDER: str = os.getenv("DOCUMENT_VLM_PROVIDER", "gemini")
    DOCUMENT_VLM_MODEL: str = os.getenv("DOCUMENT_VLM_MODEL", "gemini-2.5-flash")
    DOCUMENT_VLM_TIMEOUT: float = float(os.getenv("DOCUMENT_VLM_TIMEOUT", "30.0"))
    DOCUMENT_VLM_MAX_PAGES_PER_REQUEST: int = int(os.getenv("DOCUMENT_VLM_MAX_PAGES_PER_REQUEST", "5"))
    DOCUMENT_VLM_MAX_REGIONS_PER_REQUEST: int = int(os.getenv("DOCUMENT_VLM_MAX_REGIONS_PER_REQUEST", "20"))
    DOCUMENT_VLM_CONFIDENCE_THRESHOLD: float = float(os.getenv("DOCUMENT_VLM_CONFIDENCE_THRESHOLD", "0.80"))
    # Page-level VLM document understanding (primary intelligence mode)
    DOCUMENT_VLM_PAGE_UNDERSTANDING: bool = os.getenv("DOCUMENT_VLM_PAGE_UNDERSTANDING", "true").lower() == "true"
    # Step 11C Intelligent Extraction Settings
    INTELLIGENT_EXTRACTION_ENABLED: bool = os.getenv("INTELLIGENT_EXTRACTION_ENABLED", "true").lower() == "true"


settings = Settings()

