"""Mock image retriever for offline testing and ablation studies."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2

from image_paster.dsl.ir import SourceReqsIR, AppearanceIR
from image_paster.retrieval.base import ImageRetriever, ImageCandidate, RetrievalResult


class MockRetriever(ImageRetriever):
    """Generates synthetic candidate images locally for testing without network calls."""

    def __init__(self, cache_dir: Optional[Path | str] = None):
        super().__init__(cache_dir)

    def retrieve(
        self,
        object_name: str,
        source_reqs: Optional[SourceReqsIR] = None,
        appearance: Optional[AppearanceIR] = None,
        max_results: int = 3,
    ) -> RetrievalResult:
        query = self.generate_query(object_name, source_reqs, appearance)
        candidates: List[ImageCandidate] = []

        # Color mapping heuristic
        color_bgr = (180, 180, 180)  # Default gray
        color_name = (appearance.color if appearance and appearance.color else "").lower()
        if "gray" in color_name or "elephant" in object_name:
            color_bgr = (150, 150, 150)
        elif "green" in color_name or "tree" in object_name or "flower" in object_name:
            color_bgr = (34, 139, 34)
        elif "red" in color_name or "panda" in object_name:
            color_bgr = (40, 50, 200)
        elif "brown" in color_name or "wood" in object_name or "chair" in object_name:
            color_bgr = (42, 75, 139)
        elif "blue" in color_name:
            color_bgr = (200, 100, 50)
        elif "yellow" in color_name:
            color_bgr = (0, 215, 255)

        for i in range(max_results):
            filename = f"mock_{object_name}_{i+1}.png"
            filepath = self.cache_dir / filename

            # Generate synthetic 400x400 RGBA image
            h, w = 400, 400
            canvas = np.zeros((h, w, 4), dtype=np.uint8)

            # Draw representative shapes
            if "tree" in object_name:
                # Trunk
                cv2.rectangle(canvas, (170, 220), (230, 390), (19, 69, 139, 255), -1)
                # Foliage
                cv2.circle(canvas, (200, 170), 120, (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
                cv2.circle(canvas, (140, 190), 80, (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
                cv2.circle(canvas, (260, 190), 80, (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
            elif "chair" in object_name:
                # Seat
                cv2.rectangle(canvas, (100, 200), (300, 240), (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
                # Backrest
                cv2.rectangle(canvas, (100, 70), (140, 200), (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
                # Legs
                cv2.rectangle(canvas, (110, 240), (135, 380), (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
                cv2.rectangle(canvas, (265, 240), (290, 380), (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
            else:
                # Generic animal / object oval shape
                cv2.ellipse(canvas, (200, 220), (130, 90), 0, 0, 360, (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
                # Head
                cv2.circle(canvas, (290, 170), 60, (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
                # Legs
                cv2.rectangle(canvas, (130, 280), (160, 380), (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)
                cv2.rectangle(canvas, (230, 280), (260, 380), (color_bgr[0], color_bgr[1], color_bgr[2], 255), -1)

            cv2.imwrite(str(filepath), canvas)

            cand = ImageCandidate(
                object_name=object_name,
                query=query,
                image_url=f"mock://{object_name}/{i+1}",
                source_url=f"mock://source/{object_name}/{i+1}",
                ranking=i + 1,
                local_cached_path=str(filepath),
                width=w,
                height=h,
                metadata={"synthetic": True},
            )
            candidates.append(cand)

        return RetrievalResult(object_name=object_name, query=query, candidates=candidates)
