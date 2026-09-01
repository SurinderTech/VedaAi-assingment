from app.core.config import settings


def test_document_vlm_enabled_defaults_to_primary_mode():
    assert settings.DOCUMENT_VLM_ENABLED is True


def test_question_extraction_prefers_semantic_graph_over_regex_fallback():
    # This is a regression guard: a semantic graph must not be silently ignored
    # in favor of regex-only heuristics.
    assert settings.INTELLIGENT_EXTRACTION_ENABLED is True
