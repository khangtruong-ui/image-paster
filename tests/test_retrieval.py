"""Unit tests for Image Retrieval module."""

from pathlib import Path
from image_paster.dsl.ir import SourceReqsIR, AppearanceIR
from image_paster.retrieval.base import ImageRetriever, ImageCandidate, RetrievalResult
from image_paster.retrieval.mock import MockRetriever
from image_paster.retrieval.duckduckgo import DuckDuckGoRetriever


def test_query_synthesis():
    retriever = MockRetriever()
    source = SourceReqsIR(
        viewpoint="side",
        isolated="preferred",
        full_body="required",
        resolution="high",
    )
    appearance = AppearanceIR(color="gray")
    query = retriever.generate_query("elephant", source, appearance)

    assert "gray" in query
    assert "elephant" in query
    assert "side view" in query
    assert "full body" in query
    assert "isolated white background" in query
    assert "high resolution" in query


def test_mock_retriever_generates_candidates(tmp_path):
    retriever = MockRetriever(cache_dir=tmp_path)
    res = retriever.retrieve("tree", max_results=2)

    assert res.object_name == "tree"
    assert len(res.candidates) == 2
    for cand in res.candidates:
        assert cand.local_cached_path is not None
        assert Path(cand.local_cached_path).exists()
        assert cand.width == 400
        assert cand.height == 400


def test_duckduckgo_fallback_to_mock(tmp_path):
    mock = MockRetriever(cache_dir=tmp_path)
    # DuckDuckGoRetriever with mock as fallback
    ddg = DuckDuckGoRetriever(cache_dir=tmp_path, fallback_retriever=mock, max_retries=0)
    res = ddg.retrieve("elephant", max_results=2)
    assert res.object_name == "elephant"
    assert len(res.candidates) >= 1


def test_query_synthesis_with_explicit_search_query():
    retriever = MockRetriever()
    source = SourceReqsIR(
        query="red africa elephant",
        isolated="preferred",
    )
    query = retriever.generate_query("elephant", source)
    assert "red africa elephant" in query
    assert "isolated" in query


def test_duckduckgo_returns_real_candidates_with_mocked_ddgs(tmp_path, monkeypatch):
    """Verify DuckDuckGoRetriever returns real http URLs from ddgs.images without falling back to mock."""
    from unittest.mock import MagicMock
    import image_paster.retrieval.duckduckgo as ddg_mod

    fake_ddgs_instance = MagicMock()
    fake_ddgs_instance.images.return_value = [
        {
            "title": "Elephant in Africa",
            "image": "https://example.com/elephant1.jpg",
            "thumbnail": "https://example.com/thumb1.jpg",
            "url": "https://example.com/source1",
            "width": 800,
            "height": 600,
        },
        {
            "title": "African Elephant",
            "image": "https://example.com/elephant2.jpg",
            "thumbnail": "https://example.com/thumb2.jpg",
            "url": "https://example.com/source2",
            "width": 1024,
            "height": 768,
        },
    ]

    class FakeDDGSContext:
        def __enter__(self):
            return fake_ddgs_instance
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(ddg_mod, "DDGS", FakeDDGSContext)

    retriever = DuckDuckGoRetriever(cache_dir=tmp_path, download_immediately=False)
    res = retriever.retrieve("elephant", max_results=2)

    assert res.error is None
    assert len(res.candidates) == 2
    assert res.candidates[0].image_url == "https://example.com/elephant1.jpg"
    assert res.candidates[0].source_url == "https://example.com/source1"
    assert res.candidates[1].image_url == "https://example.com/elephant2.jpg"
    assert not any(c.image_url.startswith("mock://") for c in res.candidates)
