"""Base abstractions and metadata models for image retrieval."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import hashlib
from PIL import Image

from image_paster.dsl.ir import SourceReqsIR, AppearanceIR


@dataclass
class ImageCandidate:
    """Metadata for a retrieved candidate image."""
    object_name: str
    query: str
    image_url: str
    source_url: str
    ranking: int
    retrieval_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    local_cached_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Result of retrieving candidate images for an object."""
    object_name: str
    query: str
    candidates: List[ImageCandidate] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_name": self.object_name,
            "query": self.query,
            "error": self.error,
            "candidates": [c.to_dict() for c in self.candidates],
        }


class RetrievalError(Exception):
    """Structured retrieval error."""

    def __init__(self, object_name: str, query: str, reason: str):
        self.object_name = object_name
        self.query = query
        self.reason = reason
        super().__init__(f"RetrievalError(object='{object_name}', query='{query}'): {reason}")


class ImageRetriever(ABC):
    """Abstract interface for replaceable image retrieval engines."""

    def __init__(self, cache_dir: Optional[Path | str] = None):
        self.cache_dir = Path(cache_dir or ".cache/image_paster/retrieval")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_query(
        self,
        object_name: str,
        source_reqs: Optional[SourceReqsIR] = None,
        appearance: Optional[AppearanceIR] = None,
    ) -> str:
        """Synthesize an object-specific retrieval query from semantic requirements."""
        clean_name = object_name.replace("_", " ")
        query_parts = []

        # Color
        if appearance and appearance.color:
            query_parts.append(appearance.color)

        query_parts.append(clean_name)

        if source_reqs:
            # Viewpoint
            if source_reqs.viewpoint:
                if source_reqs.viewpoint == "side":
                    query_parts.append("side view")
                elif source_reqs.viewpoint == "frontal":
                    query_parts.append("front view")
                elif source_reqs.viewpoint == "three_quarter":
                    query_parts.append("three quarter view")
                elif source_reqs.viewpoint == "top_down":
                    query_parts.append("top down view")
                else:
                    query_parts.append(f"{source_reqs.viewpoint} view")

            # Full body
            if source_reqs.full_body in ("required", "preferred"):
                query_parts.append("full body")

            # Isolated
            if source_reqs.isolated in ("required", "preferred"):
                query_parts.append("isolated white background transparent png")

            # Resolution
            if source_reqs.resolution == "high":
                query_parts.append("high resolution")

        return " ".join(query_parts)

    @abstractmethod
    def retrieve(
        self,
        object_name: str,
        source_reqs: Optional[SourceReqsIR] = None,
        appearance: Optional[AppearanceIR] = None,
        max_results: int = 5,
    ) -> RetrievalResult:
        """Retrieve candidate images for a given scene object."""
        pass

    def download_image(self, candidate: ImageCandidate) -> Optional[str]:
        """Download candidate image to local cache if not already cached."""
        if candidate.local_cached_path and Path(candidate.local_cached_path).exists():
            return candidate.local_cached_path

        url_hash = hashlib.md5(candidate.image_url.encode("utf-8")).hexdigest()[:12]
        dest_filename = f"{candidate.object_name}_{candidate.ranking}_{url_hash}.png"
        dest_path = self.cache_dir / dest_filename

        if dest_path.exists():
            candidate.local_cached_path = str(dest_path)
            return str(dest_path)

        # Download with timeout and user-agent
        req = urllib.request.Request(
            candidate.image_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImagePaster/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                temp_path = dest_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    f.write(data)
                # Verify valid image
                with Image.open(temp_path) as img:
                    candidate.width, candidate.height = img.size
                    img.convert("RGBA").save(dest_path, "PNG")
                if temp_path.exists():
                    temp_path.unlink()
                candidate.local_cached_path = str(dest_path)
                return str(dest_path)
        except Exception as e:
            return None
