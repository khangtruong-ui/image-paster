"""Declarative constraint validation and resolution."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class ConstraintViolation(Exception):
    """Structured constraint violation error."""
    subject: str
    constraint: str
    target: Optional[str]
    reason: str
    details: Dict[str, Any]

    def __str__(self) -> str:
        tgt = f" on '{self.target}'" if self.target else ""
        return f"ConstraintViolation({self.subject}.{self.constraint}{tgt}): {self.reason} - {self.details}"


class ConstraintSolver:
    """Evaluates and enforces scene constraints on computed layout boxes."""

    @staticmethod
    def evaluate(
        constraints: List[Any],
        boxes: Dict[str, Tuple[int, int, int, int]],  # (x, y, w, h)
        z_indices: Dict[str, int],
        ground_y: int,
        canvas_w: int = 1024,
        canvas_h: int = 1024,
    ) -> List[ConstraintViolation]:
        """Check all constraints against computed layouts. Returns list of violations."""
        violations: List[ConstraintViolation] = []

        for cstr in constraints:
            subj = cstr.subject
            c_type = cstr.constraint.lower()
            tgt = cstr.target

            if subj not in boxes:
                continue

            sx, sy, sw, sh = boxes[subj]
            s_bottom = sy + sh
            s_area = sw * sh

            if c_type in ("must_touch", "standing_on", "touching"):
                if tgt == "ground":
                    # Check distance from bottom to ground line
                    dist = abs(s_bottom - ground_y)
                    if dist > 35:  # tolerance in pixels
                        violations.append(ConstraintViolation(
                            subject=subj,
                            constraint=c_type,
                            target=tgt,
                            reason=f"Object bottom ({s_bottom}) does not touch ground line ({ground_y})",
                            details={"distance": dist, "ground_y": ground_y, "bottom": s_bottom},
                        ))
                elif tgt in boxes:
                    tx, ty, tw, th = boxes[tgt]
                    # Check if subject bottom touches top of target (e.g. chair top)
                    dist = abs(s_bottom - ty)
                    if dist > 40:
                        violations.append(ConstraintViolation(
                            subject=subj,
                            constraint=c_type,
                            target=tgt,
                            reason=f"Object bottom ({s_bottom}) does not touch target top ({ty})",
                            details={"distance": dist, "target_top": ty, "bottom": s_bottom},
                        ))

            elif c_type == "must_be_inside":
                if tgt in boxes:
                    tx, ty, tw, th = boxes[tgt]
                    if not (sx >= tx and sy >= ty and (sx + sw) <= (tx + tw) and (sy + sh) <= (ty + th)):
                        violations.append(ConstraintViolation(
                            subject=subj,
                            constraint=c_type,
                            target=tgt,
                            reason=f"Object '{subj}' is not contained inside '{tgt}'",
                            details={"subject_box": (sx, sy, sw, sh), "target_box": (tx, ty, tw, th)},
                        ))
                elif tgt == "canvas":
                    if sx < 0 or sy < 0 or (sx + sw) > canvas_w or (sy + sh) > canvas_h:
                        violations.append(ConstraintViolation(
                            subject=subj,
                            constraint=c_type,
                            target=tgt,
                            reason=f"Object '{subj}' exceeds canvas bounds",
                            details={"box": (sx, sy, sw, sh), "canvas": (canvas_w, canvas_h)},
                        ))

            elif c_type in ("must_be_larger_than", "larger_than"):
                if tgt in boxes:
                    tx, ty, tw, th = boxes[tgt]
                    t_area = tw * th
                    if s_area <= t_area:
                        violations.append(ConstraintViolation(
                            subject=subj,
                            constraint=c_type,
                            target=tgt,
                            reason=f"Subject area ({s_area}) is not larger than target area ({t_area})",
                            details={"subject_area": s_area, "target_area": t_area},
                        ))

            elif c_type in ("must_be_smaller_than", "smaller_than"):
                if tgt in boxes:
                    tx, ty, tw, th = boxes[tgt]
                    t_area = tw * th
                    if s_area >= t_area:
                        violations.append(ConstraintViolation(
                            subject=subj,
                            constraint=c_type,
                            target=tgt,
                            reason=f"Subject area ({s_area}) is not smaller than target area ({t_area})",
                            details={"subject_area": s_area, "target_area": t_area},
                        ))

            elif c_type in ("must_occlude", "occludes"):
                if tgt in boxes:
                    tx, ty, tw, th = boxes[tgt]
                    # Check overlap of bounding boxes
                    overlap_x = max(0, min(sx + sw, tx + tw) - max(sx, tx))
                    overlap_y = max(0, min(sy + sh, ty + th) - max(sy, ty))
                    overlap_area = overlap_x * overlap_y
                    sz = z_indices.get(subj, 0)
                    tz = z_indices.get(tgt, 0)

                    if sz <= tz:
                        violations.append(ConstraintViolation(
                            subject=subj,
                            constraint=c_type,
                            target=tgt,
                            reason=f"Subject '{subj}' (z={sz}) is behind target '{tgt}' (z={tz}); cannot occlude it",
                            details={"subject_z": sz, "target_z": tz},
                        ))
                    elif overlap_area <= 0:
                        violations.append(ConstraintViolation(
                            subject=subj,
                            constraint=c_type,
                            target=tgt,
                            reason=f"Subject '{subj}' does not spatially overlap target '{tgt}'",
                            details={"overlap_area": overlap_area},
                        ))

            elif c_type in ("must_not_overlap", "no_overlap"):
                if tgt in boxes:
                    tx, ty, tw, th = boxes[tgt]
                    overlap_x = max(0, min(sx + sw, tx + tw) - max(sx, tx))
                    overlap_y = max(0, min(sy + sh, ty + th) - max(sy, ty))
                    if overlap_x > 0 and overlap_y > 0:
                        violations.append(ConstraintViolation(
                            subject=subj,
                            constraint=c_type,
                            target=tgt,
                            reason=f"Subject '{subj}' overlaps target '{tgt}'",
                            details={"overlap": (overlap_x, overlap_y)},
                        ))

        return violations
