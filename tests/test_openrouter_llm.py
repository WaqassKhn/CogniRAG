"""
tests/test_openrouter_llm.py
────────────────────────────
Integration and unit tests for OpenRouterLLM client.
Tests model discovery, live filtering, daily rate-limit short-circuiting,
fast Gemini fallback, and streaming.
"""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from rag.openrouter_llm import OpenRouterLLM
from config import OPENROUTER_API_KEY


@pytest.fixture
def mock_openrouter_llm():
    """Returns an OpenRouterLLM client with mocked network interactions."""
    with patch.object(OpenRouterLLM, "_fetch_live_free_models", return_value={"openrouter/free", "google/gemma-4-31b-it:free"}):
        llm = OpenRouterLLM(api_key="sk-or-test-key")
        llm._gemini_fallback = MagicMock()
        llm._gemini_fallback.is_available.return_value = True
        llm._gemini_fallback.generate.return_value = "Gemini fallback response"
        return llm


def test_openrouter_daily_rate_limit_shortcircuits_to_gemini(mock_openrouter_llm):
    """
    When OpenRouter returns 'free-models-per-day' 429, the client must immediately
    short-circuit to Gemini rather than retrying every other model sequentially.
    """
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.text = '{"error":{"message":"Rate limit exceeded: free-models-per-day","code":429}}'
    http_err = httpx.HTTPStatusError("429 Rate Limit", request=MagicMock(), response=mock_resp)

    call_count = 0
    def mock_call(model, messages, temp, stream=False):
        nonlocal call_count
        call_count += 1
        raise http_err

    mock_openrouter_llm._call_model = mock_call

    result = mock_openrouter_llm.generate("Test prompt", task="answer")

    # Should only try 1 model before short-circuiting to Gemini
    assert call_count == 1
    assert result == "Gemini fallback response"
    mock_openrouter_llm._gemini_fallback.generate.assert_called_once()


def test_openrouter_filters_dead_models(mock_openrouter_llm):
    """Verifies that models not in _live_free_models are skipped."""
    mock_openrouter_llm._live_free_models = {"google/gemma-4-31b-it:free"}
    mock_openrouter_llm._call_model = MagicMock(return_value="Answer from live model")

    result = mock_openrouter_llm.generate("What is revenue?", task="answer")

    assert result == "Answer from live model"
    assert mock_openrouter_llm.last_model_used == "google/gemma-4-31b-it:free"
    # openrouter/free should have been filtered out since it wasn't in _live_free_models
    assert mock_openrouter_llm._call_model.call_args[0][0] == "google/gemma-4-31b-it:free"


def test_openrouter_successful_generation(mock_openrouter_llm):
    """Verifies normal generation path when model call succeeds."""
    mock_openrouter_llm._call_model = MagicMock(return_value="Successful grounded response")

    ans = mock_openrouter_llm.generate("Explain project status", task="answer")
    assert ans == "Successful grounded response"


def test_openrouter_stream_fallback(mock_openrouter_llm):
    """Verifies streaming gracefully yields Gemini response when OpenRouter models fail."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.read.return_value = b'{"error":{"message":"Rate limit exceeded: free-models-per-day"}}'

    with patch("httpx.Client.stream") as mock_stream_ctx:
        mock_stream_ctx.return_value.__enter__.return_value = mock_resp
        chunks = list(mock_openrouter_llm.generate_stream("Stream this", task="answer"))

        assert len(chunks) == 1
        assert chunks[0] == "Gemini fallback response"


@pytest.mark.skipif(not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
def test_openrouter_model_discovery():
    """Live test checking free model list structure."""
    llm = OpenRouterLLM()
    live_models = llm._live_free_models
    assert isinstance(live_models, set)
    active = llm.get_active_models()
    assert "answer" in active
    assert "decompose" in active
    assert "judge" in active
