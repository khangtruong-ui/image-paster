"""Image retrieval package."""

from image_paster.retrieval.base import (
    ImageCandidate,
    RetrievalResult,
    RetrievalError,
    ImageRetriever,
)
from image_paster.retrieval.duckduckgo import DuckDuckGoRetriever
from image_paster.retrieval.mock import MockRetriever

__all__ = [
    "ImageCandidate",
    "RetrievalResult",
    "RetrievalError",
    "ImageRetriever",
    "DuckDuckGoRetriever",
    "MockRetriever",
]
