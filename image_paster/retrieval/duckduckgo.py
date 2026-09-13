"""DuckDuckGo Image Retrieval implementation."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, List

from image_paster.dsl.ir import SourceReqsIR, AppearanceIR
from image_paster.retrieval.base import (
    ImageRetriever,
    ImageCandidate,
    RetrievalResult,
    RetrievalError,
)

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


class DuckDuckGoRetriever(ImageRetriever):
    """Retrieves candidate images using DuckDuckGo image search."""

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        max_retries: int = 2,
        download_immediately: bool = True,
        fallback_retriever: Optional[ImageRetriever] = None,
    ):
        super().__init__(cache_dir)
        self.max_retries = max_retries
        self.download_immediately = download_immediately
        self.fallback_retriever = fallback_retriever

    def retrieve(
        self,
        object_name: str,
        source_reqs: Optional[SourceReqsIR] = None,
        appearance: Optional[AppearanceIR] = None,
        max_results: int = 5,
    ) -> RetrievalResult:
        query = self.generate_query(object_name, source_reqs, appearance)

        if DDGS is None:
            if self.fallback_retriever:
                return self.fallback_retriever.retrieve(object_name, source_reqs, appearance, max_results)
            return RetrievalResult(
                object_name=object_name,
                query=query,
                error="Neither 'ddgs' nor 'duckduckgo_search' is installed.",
            )

        candidates: List[ImageCandidate] = []
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Use DDGS context manager
                with DDGS() as ddgs:
                    # layout/type hints based on source_reqs
                    image_type = "transparent" if (source_reqs and source_reqs.isolated in ("preferred", "required")) else None
                    results = list(
                        ddgs.images(
                            keywords=query,
                            max_results=max_results,
                            type_image=image_type,
                        )
                    )

                for idx, r in enumerate(results):
                    img_url = r.get("image") or r.get("thumbnail") or ""
                    cand = ImageCandidate(
                        object_name=object_name,
                        query=query,
                        image_url=img_url,
                        source_url=r.get("url") or "",
                        ranking=idx + 1,
                        metadata={
                            "title": r.get("title", ""),
                            "width": r.get("width"),
                            "height": r.get("height"),
                        },
                    )
                    if self.download_immediately and img_url:
                        self.download_image(cand)
                    candidates.append(cand)

                if candidates:
                    return RetrievalResult(object_name=object_name, query=query, candidates=candidates)

            except Exception as e:
                last_exception = e
                time.sleep(0.5 * (attempt + 1))

        # If DDG fails (e.g. 403 Rate Limit), fall back if available
        if self.fallback_retriever:
            fallback_res = self.fallback_retriever.retrieve(object_name, source_reqs, appearance, max_results)
            if fallback_res.candidates:
                fallback_res.query = query
                return fallback_res

        return RetrievalResult(
            object_name=object_name,
            query=query,
            candidates=candidates,
            error=str(last_exception) if last_exception else "No images found",
        )
