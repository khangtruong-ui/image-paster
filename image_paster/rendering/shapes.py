"""Basic geometric shapes and text rendering for Scene DSL cutouts."""

from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from image_paster.segmentation.base import SegmentationResult


NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 200, 0),
    "dark_green": (0, 100, 0),
    "lime": (50, 255, 50),
    "blue": (0, 0, 255),
    "sky_blue": (135, 206, 235),
    "navy": (0, 0, 128),
    "yellow": (255, 255, 0),
    "gold": (255, 215, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "magenta": (255, 0, 255),
    "pink": (255, 192, 203),
    "cyan": (0, 255, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "brown": (139, 69, 19),
    "teal": (0, 128, 128),
    "violet": (238, 130, 238),
}


def parse_color(color_spec: Any, default: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Tuple[int, int, int, int]:
    """Parse color into RGBA tuple (0-255)."""
    if color_spec is None:
        return default

    if isinstance(color_spec, (tuple, list)):
        if len(color_spec) >= 4:
            a = color_spec[3]
            if isinstance(a, float) and a <= 1.0:
                a_val = int(a * 255)
            else:
                a_val = int(a)
            return (int(color_spec[0]), int(color_spec[1]), int(color_spec[2]), a_val)
        elif len(color_spec) == 3:
            return (int(color_spec[0]), int(color_spec[1]), int(color_spec[2]), default[3])
        elif len(color_spec) == 1:
            val = int(color_spec[0])
            return (val, val, val, default[3])

    if isinstance(color_spec, str):
        c_clean = color_spec.strip().lower().strip("\"'")
        if c_clean in NAMED_COLORS:
            r, g, b = NAMED_COLORS[c_clean]
            return (r, g, b, default[3])

        if c_clean.startswith("#"):
            hex_code = c_clean.lstrip("#")
            if len(hex_code) == 3:
                r = int(hex_code[0] * 2, 16)
                g = int(hex_code[1] * 2, 16)
                b = int(hex_code[2] * 2, 16)
                return (r, g, b, default[3])
            elif len(hex_code) == 6:
                r = int(hex_code[0:2], 16)
                g = int(hex_code[2:4], 16)
                b = int(hex_code[4:6], 16)
                return (r, g, b, default[3])
            elif len(hex_code) == 8:
                r = int(hex_code[0:2], 16)
                g = int(hex_code[2:4], 16)
                b = int(hex_code[4:6], 16)
                a = int(hex_code[6:8], 16)
                return (r, g, b, a)

        # Regex for rgba?(r, g, b[, a])
        m = re.match(r"^rgba?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)$", c_clean)
        if m:
            r = int(float(m.group(1)))
            g = int(float(m.group(2)))
            b = int(float(m.group(3)))
            a = default[3]
            if m.group(4) is not None:
                alpha_str = m.group(4).strip()
                alpha_float = float(alpha_str)
                if "." in alpha_str and alpha_float <= 1.0:
                    a = int(alpha_float * 255)
                else:
                    a = int(alpha_float)
            return (r, g, b, min(255, max(0, a)))

        # Fallback comma-separated
        if "," in c_clean:
            parts = [p.strip().lstrip("rgb(").rstrip(")") for p in c_clean.split(",")]
            try:
                nums = [float(p) for p in parts if p]
                if len(nums) >= 3:
                    r = int(nums[0])
                    g = int(nums[1])
                    b = int(nums[2])
                    a = default[3]
                    if len(nums) > 3:
                        if nums[3] <= 1.0:
                            a = int(nums[3] * 255)
                        else:
                            a = int(nums[3])
                    return (r, g, b, min(255, max(0, a)))
            except ValueError:
                pass

    return default


def _make_segmentation_result(rgba: np.ndarray, name: str = "shape") -> SegmentationResult:
    mask = (rgba[:, :, 3] > 10).astype(np.uint8) * 255
    h_out, w_out = rgba.shape[:2]
    return SegmentationResult(
        object_name=name,
        original_image=rgba[:, :, :3].copy(),
        mask=mask,
        extracted_rgba=rgba,
        bbox=(0, 0, w_out, h_out),
        score=1.0,
        rejected=False,
    )


def render_circle(
    radius: int = 50,
    color: Any = "yellow",
    fill: Any = None,
    stroke: Any = None,
    stroke_color: Any = None,
    stroke_width: int = 2,
    thickness: int = 2,
    opacity: float = 1.0,
    name: str = "circle",
    **kwargs,
) -> SegmentationResult:
    """Render a clean anti-aliased circle cutout as a SegmentationResult."""
    rad = max(4, int(radius))
    size = 2 * rad
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fill_color = parse_color(fill or color)
    if opacity < 1.0:
        fill_color = (fill_color[0], fill_color[1], fill_color[2], int(fill_color[3] * opacity))

    outline_arg = stroke_color or stroke
    outline_color = parse_color(outline_arg) if outline_arg else None
    if outline_color and opacity < 1.0:
        outline_color = (outline_color[0], outline_color[1], outline_color[2], int(outline_color[3] * opacity))

    th = int(kwargs.get("stroke_width", stroke_width if stroke_width != 2 or thickness == 2 else thickness))
    bbox = [0, 0, size - 1, size - 1]
    draw.ellipse(bbox, fill=fill_color, outline=outline_color, width=max(1, th) if outline_color else 0)
    rgba = np.array(img, dtype=np.uint8)
    return _make_segmentation_result(rgba, name=name)


def render_rectangle(
    width: int = 120,
    height: int = 80,
    color: Any = "blue",
    fill: Any = None,
    stroke: Any = None,
    stroke_color: Any = None,
    stroke_width: int = 2,
    thickness: int = 2,
    corner_radius: int = 0,
    opacity: float = 1.0,
    name: str = "rectangle",
    **kwargs,
) -> SegmentationResult:
    """Render a clean anti-aliased rectangle cutout as a SegmentationResult."""
    w = max(4, int(width))
    h = max(4, int(height))

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fill_color = parse_color(fill or color)
    if opacity < 1.0:
        fill_color = (fill_color[0], fill_color[1], fill_color[2], int(fill_color[3] * opacity))

    outline_arg = stroke_color or stroke
    outline_color = parse_color(outline_arg) if outline_arg else None
    if outline_color and opacity < 1.0:
        outline_color = (outline_color[0], outline_color[1], outline_color[2], int(outline_color[3] * opacity))

    th = int(kwargs.get("stroke_width", stroke_width if stroke_width != 2 or thickness == 2 else thickness))
    cr = int(kwargs.get("corner_radius", corner_radius))
    bbox = [0, 0, w - 1, h - 1]
    if cr > 0:
        draw.rounded_rectangle(
            bbox,
            radius=min(cr, w // 2, h // 2),
            fill=fill_color,
            outline=outline_color,
            width=max(1, th) if outline_color else 0,
        )
    else:
        draw.rectangle(
            bbox,
            fill=fill_color,
            outline=outline_color,
            width=max(1, th) if outline_color else 0,
        )
    rgba = np.array(img, dtype=np.uint8)
    return _make_segmentation_result(rgba, name=name)


def render_triangle(
    width: int = 120,
    base: Optional[int] = None,
    height: int = 100,
    color: Any = "red",
    fill: Any = None,
    stroke: Any = None,
    stroke_color: Any = None,
    stroke_width: int = 2,
    thickness: int = 2,
    points: Optional[List[Any]] = None,
    opacity: float = 1.0,
    name: str = "triangle",
    **kwargs,
) -> SegmentationResult:
    """Render an anti-aliased triangle cutout as a SegmentationResult."""
    w = max(6, int(base if base is not None else width))
    h = max(6, int(height))

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fill_color = parse_color(fill or color)
    if opacity < 1.0:
        fill_color = (fill_color[0], fill_color[1], fill_color[2], int(fill_color[3] * opacity))

    outline_arg = stroke_color or stroke
    outline_color = parse_color(outline_arg) if outline_arg else None
    if outline_color and opacity < 1.0:
        outline_color = (outline_color[0], outline_color[1], outline_color[2], int(outline_color[3] * opacity))

    th = int(kwargs.get("stroke_width", stroke_width if stroke_width != 2 or thickness == 2 else thickness))

    pts = kwargs.get("points", points)
    if pts and len(pts) >= 3:
        poly_pts = [(float(p[0]), float(p[1])) for p in pts[:3]]
    else:
        poly_pts = [
            (w / 2.0, 0.0),
            (0.0, float(h - 1)),
            (float(w - 1), float(h - 1)),
        ]

    draw.polygon(poly_pts, fill=fill_color, outline=outline_color)
    if outline_color and th > 1:
        draw.line(poly_pts + [poly_pts[0]], fill=outline_color, width=th)

    rgba = np.array(img, dtype=np.uint8)
    return _make_segmentation_result(rgba, name=name)


def render_line(
    length: int = 200,
    thickness: int = 4,
    stroke_width: int = 4,
    color: Any = "white",
    angle: float = 0.0,
    opacity: float = 1.0,
    x1: Optional[float] = None,
    y1: Optional[float] = None,
    x2: Optional[float] = None,
    y2: Optional[float] = None,
    name: str = "line",
    **kwargs,
) -> SegmentationResult:
    """Render an anti-aliased straight line cutout as a SegmentationResult."""
    th = max(1, int(kwargs.get("stroke_width", stroke_width if stroke_width != 4 or thickness == 4 else thickness)))
    line_color = parse_color(kwargs.get("stroke_color", color))
    if opacity < 1.0:
        line_color = (line_color[0], line_color[1], line_color[2], int(line_color[3] * opacity))

    if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        span_w = int(abs(max_x - min_x))
        span_h = int(abs(max_y - min_y))
        pad = max(th, 2)
        w = max(span_w + 2 * pad, 4)
        h = max(span_h + 2 * pad, 4)
        p1 = (x1 - min_x + pad, y1 - min_y + pad)
        p2 = (x2 - min_x + pad, y2 - min_y + pad)
    else:
        l = max(6, int(length))
        rad = math.radians(angle)
        dx = l * math.cos(rad)
        dy = l * math.sin(rad)
        span_w = int(abs(dx))
        span_h = int(abs(dy))
        pad = max(th, 2)
        w = max(span_w + 2 * pad, 6)
        h = max(span_h + 2 * pad, 6)
        cx, cy = w / 2.0, h / 2.0
        p1 = (cx - dx / 2.0, cy - dy / 2.0)
        p2 = (cx + dx / 2.0, cy + dy / 2.0)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.line([p1, p2], fill=line_color, width=th)
    rgba = np.array(img, dtype=np.uint8)
    return _make_segmentation_result(rgba, name=name)


def render_curve(
    width: int = 200,
    height: int = 80,
    thickness: int = 4,
    stroke_width: int = 4,
    color: Any = "cyan",
    curvature: float = 1.0,
    points: Optional[List[Any]] = None,
    opacity: float = 1.0,
    name: str = "curve",
    **kwargs,
) -> SegmentationResult:
    """Render a smooth curved arc / bezier line cutout as a SegmentationResult."""
    th = max(1, int(kwargs.get("stroke_width", stroke_width if stroke_width != 4 or thickness == 4 else thickness)))
    line_color = parse_color(kwargs.get("stroke_color", color))
    if opacity < 1.0:
        line_color = (line_color[0], line_color[1], line_color[2], int(line_color[3] * opacity))

    pts = kwargs.get("points", points)
    if pts and len(pts) >= 2:
        pts_arr = np.array(pts, dtype=np.float32)
        min_x, min_y = np.min(pts_arr, axis=0)
        max_x, max_y = np.max(pts_arr, axis=0)
        span_w = int(max_x - min_x)
        span_h = int(max_y - min_y)
        pad = th + 6
        canvas_w = max(span_w + 2 * pad, 20)
        canvas_h = max(span_h + 2 * pad, 20)
        shifted = pts_arr - [min_x, min_y] + [pad, pad]
        if len(shifted) == 3:
            p0, p1, p2 = shifted[0], shifted[1], shifted[2]
            t = np.linspace(0, 1, max(50, canvas_w)).reshape(-1, 1)
            curve_pts = ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2
            pts_int = np.int32([curve_pts])
        else:
            pts_int = np.int32([shifted])
    else:
        w = max(20, int(width))
        h = max(20, int(height))
        pad = th + 6
        canvas_w = w + 2 * pad
        canvas_h = h + 2 * pad
        p0 = np.array([pad, pad + h * 0.85], dtype=np.float32)
        p1 = np.array([pad + w * 0.5, pad + h * (0.85 - 0.8 * curvature)], dtype=np.float32)
        p2 = np.array([pad + w, pad + h * 0.85], dtype=np.float32)
        t = np.linspace(0, 1, max(50, w)).reshape(-1, 1)
        curve_pts = ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2
        pts_int = np.int32([curve_pts])

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    bgr_a = (line_color[2], line_color[1], line_color[0], line_color[3])
    cv2.polylines(canvas, pts_int, isClosed=False, color=bgr_a, thickness=th, lineType=cv2.LINE_AA)

    rgba_out = np.zeros_like(canvas)
    rgba_out[:, :, 0] = canvas[:, :, 2]  # R
    rgba_out[:, :, 1] = canvas[:, :, 1]  # G
    rgba_out[:, :, 2] = canvas[:, :, 0]  # B
    rgba_out[:, :, 3] = canvas[:, :, 3]  # A
    return _make_segmentation_result(rgba_out, name=name)


def render_text(
    text: str = "Scene Text",
    content: Optional[str] = None,
    font_size: int = 36,
    size: Optional[int] = None,
    color: Any = "white",
    stroke: Any = None,
    stroke_color: Any = None,
    stroke_width: int = 0,
    background: Any = None,
    bg_color: Any = None,
    opacity: float = 1.0,
    name: str = "text",
    **kwargs,
) -> SegmentationResult:
    """Render crisp text as a SegmentationResult cutout using PIL."""
    text_content = str(content if content is not None else kwargs.get("content", text)).strip()
    if not text_content:
        text_content = " "

    fsize = max(10, int(size if size is not None else kwargs.get("size", font_size)))
    try:
        font = ImageFont.load_default(size=fsize)
    except Exception:
        font = ImageFont.load_default()

    swidth = int(kwargs.get("thickness", stroke_width))
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text_content, font=font, stroke_width=swidth)
    tw = max(10, bbox[2] - bbox[0])
    th = max(10, bbox[3] - bbox[1])

    pad_x = max(8, swidth * 2 + 4)
    pad_y = max(8, swidth * 2 + 4)
    canvas_w = tw + 2 * pad_x
    canvas_h = th + 2 * pad_y

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg = bg_color or background or kwargs.get("bg_color") or kwargs.get("background")
    if bg:
        bg_col = parse_color(bg)
        draw.rounded_rectangle([0, 0, canvas_w - 1, canvas_h - 1], radius=6, fill=bg_col)

    text_color = parse_color(color)
    if opacity < 1.0:
        text_color = (text_color[0], text_color[1], text_color[2], int(text_color[3] * opacity))

    outline_arg = stroke_color or stroke or kwargs.get("stroke_color")
    stroke_col = parse_color(outline_arg) if outline_arg else None
    draw.text(
        (pad_x - bbox[0], pad_y - bbox[1]),
        text_content,
        font=font,
        fill=text_color,
        stroke_width=swidth,
        stroke_fill=stroke_col,
    )
    rgba = np.array(img, dtype=np.uint8)
    return _make_segmentation_result(rgba, name=name)


def render_shape_cutout(shape_ir: Any, obj_ir: Optional[Any] = None) -> SegmentationResult:
    """Render any geometric shape or text IR into a complete SegmentationResult."""
    stype = getattr(shape_ir, "shape_type", "rectangle").lower()
    name = getattr(shape_ir, "name", "shape")
    props = getattr(shape_ir, "properties", {})

    # Appearance overrides from object IR
    opacity = 1.0
    if obj_ir and getattr(obj_ir, "appearance", None):
        if obj_ir.appearance.opacity is not None:
            opacity = float(obj_ir.appearance.opacity)

    # Resolve scale factor
    scale_factor = 1.0
    if obj_ir and getattr(obj_ir, "transformation", None):
        sc = obj_ir.transformation.scale
        if isinstance(sc, (int, float)):
            scale_factor = float(sc)
        elif isinstance(sc, str):
            scale_map = {"tiny": 0.5, "small": 0.75, "medium": 1.0, "large": 1.4, "huge": 2.0}
            scale_factor = scale_map.get(sc.lower(), 1.0)

    color = getattr(shape_ir, "color", None) or props.get("color", "white")
    fill = getattr(shape_ir, "fill", None) or props.get("fill")
    stroke = getattr(shape_ir, "stroke", None) or props.get("stroke") or props.get("border") or props.get("stroke_color")
    thickness = getattr(shape_ir, "thickness", 2) or props.get("thickness") or props.get("stroke_width", 2)

    if stype == "circle":
        radius = getattr(shape_ir, "radius", None) or props.get("radius", 50)
        radius = int(radius * scale_factor)
        return render_circle(radius=radius, color=color, fill=fill, stroke=stroke, thickness=thickness, opacity=opacity, name=name)

    elif stype == "rectangle":
        width = getattr(shape_ir, "width", None) or props.get("width", 120)
        height = getattr(shape_ir, "height", None) or props.get("height", 80)
        w = int(width * scale_factor)
        h = int(height * scale_factor)
        cr = int(props.get("corner_radius", 0))
        return render_rectangle(width=w, height=h, color=color, fill=fill, stroke=stroke, thickness=thickness, corner_radius=cr, opacity=opacity, name=name)

    elif stype == "triangle":
        width = getattr(shape_ir, "width", None) or props.get("width") or props.get("base", 120)
        height = getattr(shape_ir, "height", None) or props.get("height", 100)
        w = int(width * scale_factor)
        h = int(height * scale_factor)
        pts = getattr(shape_ir, "points", None) or props.get("points")
        return render_triangle(width=w, height=h, color=color, fill=fill, stroke=stroke, thickness=thickness, points=pts, opacity=opacity, name=name)

    elif stype == "line":
        x1 = props.get("x1")
        y1 = props.get("y1")
        x2 = props.get("x2")
        y2 = props.get("y2")
        if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
            return render_line(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), thickness=thickness, color=color, opacity=opacity, name=name)
        length = getattr(shape_ir, "length", None) or props.get("length", 200)
        l = int(length * scale_factor)
        ang = float(getattr(shape_ir, "angle", 0.0) or props.get("angle", 0.0))
        return render_line(length=l, thickness=thickness, color=color, angle=ang, opacity=opacity, name=name)

    elif stype == "curve":
        width = getattr(shape_ir, "width", None) or props.get("width", 200)
        height = getattr(shape_ir, "height", None) or props.get("height", 80)
        w = int(width * scale_factor)
        h = int(height * scale_factor)
        curv = float(getattr(shape_ir, "curvature", 1.0) or props.get("curvature", 1.0))
        pts = getattr(shape_ir, "points", None) or props.get("points")
        return render_curve(width=w, height=h, thickness=thickness, color=color, curvature=curv, points=pts, opacity=opacity, name=name)

    elif stype in ("text", "label", "caption"):
        content = getattr(shape_ir, "text", None) or props.get("content") or props.get("text") or name
        fsize = getattr(shape_ir, "font_size", 32) or props.get("font_size") or props.get("size", 32)
        fsize = int(fsize * scale_factor)
        bg = props.get("background") or props.get("bg_color")
        swidth = int(props.get("stroke_width", 0))
        return render_text(text=content, font_size=fsize, color=color, stroke=stroke, stroke_width=swidth, background=bg, opacity=opacity, name=name)

    elif stype == "ellipse":
        width = getattr(shape_ir, "width", None) or props.get("width", 140)
        height = getattr(shape_ir, "height", None) or props.get("height", 80)
        w = int(width * scale_factor)
        h = int(height * scale_factor)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        fill_col = parse_color(fill or color)
        out_col = parse_color(stroke) if stroke else None
        draw.ellipse([0, 0, w - 1, h - 1], fill=fill_col, outline=out_col, width=thickness if out_col else 0)
        rgba = np.array(img, dtype=np.uint8)
        return _make_segmentation_result(rgba, name=name)

    else:
        # Generic shape fallback (rounded rectangle)
        w = int(props.get("width", 100) * scale_factor)
        h = int(props.get("height", 100) * scale_factor)
        return render_rectangle(width=w, height=h, color=color, fill=fill, stroke=stroke, thickness=thickness, corner_radius=10, opacity=opacity, name=name)
