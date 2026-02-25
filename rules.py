"""
EAS 510 - Expert System Rules
Weights: Rule1=30pts, Rule2=30pts, Rule3=40pts (Total=100)
"""
import os

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image, UnidentifiedImageError  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_basic_image_info(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return {
                "width": int(width),
                "height": int(height),
                "format": (img.format or "").upper() or None,
                "mode": img.mode or None,
            }
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
        return {"width": None, "height": None, "format": None, "mode": None}


def _ratio(a, b):
    if a is None or b is None:
        return 0.0
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def _load_bgr(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _mode_group(mode):
    if not mode:
        return None
    m = str(mode).upper()
    if m in {"1", "L", "LA", "I", "F"}:
        return "GRAY"
    return "COLOR"


def _hsv_hist(img_bgr):
    """Compute normalized HSV H+S histogram for an image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


# ---------------------------------------------------------------------------
# Rule 1 – Metadata Analysis (30 pts)
# ---------------------------------------------------------------------------

def rule1_metadata(target_info, input_path, input_info=None):
    """
    Rule 1 (30 pts): Compare file size, dimensions, and basic properties.
    size_ratio → up to 15 pts
    dim_ratio  → up to 15 pts
    """
    input_size = os.stat(input_path).st_size
    target_size = target_info["size"]

    if input_info is None:
        input_info = get_basic_image_info(input_path)

    size_ratio = _ratio(input_size, target_size)

    tw = target_info.get("width")
    th = target_info.get("height")
    iw = input_info.get("width")
    ih = input_info.get("height")
    dim_ratio = (_ratio(iw, tw) + _ratio(ih, th)) / 2.0 if (tw and th and iw and ih) else 0.0

    mode_match = bool(
        _mode_group(input_info.get("mode"))
        and _mode_group(target_info.get("mode"))
        and _mode_group(input_info.get("mode")) == _mode_group(target_info.get("mode"))
    )

    evidence = f"Size ratio {size_ratio:.2f}"

    if not mode_match:
        return 0, False, evidence

    score = int(size_ratio * 15) + int(dim_ratio * 15)
    score = min(30, score)
    fired = score >= 10

    return score, fired, evidence


# ---------------------------------------------------------------------------
# Rule 2 – Color Histogram Analysis (30 pts)
# ---------------------------------------------------------------------------

def rule2_color_distribution(target_info, input_path):
    """
    Rule 2 (30 pts): Compare color distributions.

    For crops: a 25% crop only shows one region of the image, so comparing
    the full histograms gives low correlation. Fix: also compare the input's
    histogram against FOUR quadrant sub-regions of the target, then take
    the best match. This simulates 'is this crop from any part of the target?'

    Full image comparison → handles brightness/compression ✅
    Sub-region comparison → handles crop cases ✅
    Random images → neither full nor sub-region matches → 0% FP ✅
    """
    target_path = target_info["path"]

    img_t = _load_bgr(target_path)
    img_i = _load_bgr(input_path)
    if img_t is None or img_i is None:
        return 0, False, "Correlation 0.00"

    img_t_resized = cv2.resize(img_t, (256, 256), interpolation=cv2.INTER_AREA)
    img_i_resized = cv2.resize(img_i, (256, 256), interpolation=cv2.INTER_AREA)

    # --- Full image histogram comparison ---
    hist_t_full = _hsv_hist(img_t_resized)
    hist_i_full = _hsv_hist(img_i_resized)
    sim_full = float(cv2.compareHist(hist_t_full, hist_i_full, cv2.HISTCMP_CORREL))
    sim_full = max(0.0, min(1.0, sim_full))

    # --- Sub-region comparison for crop robustness ---
    # Divide target into 4 quadrants and compare each against input
    # A 25% crop likely came from one of these quadrants
    h, w = img_t_resized.shape[:2]
    quadrants = [
        img_t_resized[0:h//2,   0:w//2],    # top-left
        img_t_resized[0:h//2,   w//2:w],    # top-right
        img_t_resized[h//2:h,   0:w//2],    # bottom-left
        img_t_resized[h//2:h,   w//2:w],    # bottom-right
    ]

    best_quad_sim = 0.0
    for quad in quadrants:
        quad_resized = cv2.resize(quad, (128, 128), interpolation=cv2.INTER_AREA)
        hist_quad = _hsv_hist(quad_resized)
        sim_q = float(cv2.compareHist(hist_quad, hist_i_full, cv2.HISTCMP_CORREL))
        best_quad_sim = max(best_quad_sim, max(0.0, min(1.0, sim_q)))

    # Take the best of full vs sub-region
    best_sim = max(sim_full, best_quad_sim)

    score = int(best_sim * 30)
    fired = score >= 10
    evidence = f"Correlation {best_sim:.2f}"

    return score, fired, evidence


# ---------------------------------------------------------------------------
# Rule 3 – Template Matching (40 pts)
# cv2.matchTemplate() — size-aware for crop cases
# ---------------------------------------------------------------------------

def rule3_visual_similarity(target_info, input_path):
    
    target_path = target_info["path"]

    img_t = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    img_i = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_t is None or img_i is None:
        return 0, False, "Match score 0.00"

    best_score = 0.0

    h_t, w_t = img_t.shape
    h_i, w_i = img_i.shape

    # --- Attempt 1: SIZE-AWARE — preserves crop ratio ---
    # Compute how large input is relative to target
    area_ratio = (w_i * h_i) / max(w_t * h_t, 1)
    linear_ratio = max(0.2, min(0.85, area_ratio ** 0.5))
    # linear_ratio for 25% area crop = 0.5 (50% width and height)

    search_size = 500
    template_size = max(50, int(search_size * linear_ratio))

    search   = cv2.resize(img_t, (search_size, search_size), interpolation=cv2.INTER_AREA)
    template = cv2.resize(img_i, (template_size, template_size), interpolation=cv2.INTER_AREA)

    # Template must be strictly smaller than search
    if template_size < search_size:
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        best_score = max(best_score, float(max_val))

    # --- Attempt 2: same-size images (brightness, compression) ---
    # Resize input to 200x200, target to 240x240 (20% larger)
    template_2 = cv2.resize(img_i, (200, 200), interpolation=cv2.INTER_AREA)
    search_2   = cv2.resize(img_t, (240, 240), interpolation=cv2.INTER_AREA)
    result_2   = cv2.matchTemplate(search_2, template_2, cv2.TM_CCOEFF_NORMED)
    _, max_val_2, _, _ = cv2.minMaxLoc(result_2)
    best_score = max(best_score, float(max_val_2))

    best_score = max(0.0, min(1.0, best_score))

    score = int(best_score * 40)
    fired = score >= 15
    evidence = f"Match score {best_score:.2f}"

    return score, fired, evidence