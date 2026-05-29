"""
composition_analyzer.py
------------------------
Scores a 16:9 PNG image against 15 photographic composition templates.

Usage:
    from composition_analyzer import analyze_composition
    scores = analyze_composition("photo.png")
    # Returns dict of {composition_name: score (0.0 - 1.0)}
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_prep(image_path: str, target_w=960, target_h=540):
    """Load image, resize to 16:9, return BGR + grayscale + saliency map."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (target_w, target_h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    saliency = _saliency_map(gray)
    return img, gray, saliency


def _saliency_map(gray: np.ndarray) -> np.ndarray:
    """
    Simple frequency-domain saliency (spectral residual method).
    Returns float32 map normalized to [0, 1].
    """
    h, w = gray.shape
    f = np.fft.fft2(gray.astype(np.float32))
    log_amp = np.log(np.abs(f) + 1e-8)
    # Spectral residual = log amplitude - smoothed log amplitude
    smoothed = gaussian_filter(log_amp, sigma=3)
    residual = log_amp - smoothed
    # Reconstruct with original phase
    phase = np.angle(f)
    sal_f = np.exp(residual + 1j * phase)
    sal = np.abs(np.fft.ifft2(sal_f)) ** 2
    sal = gaussian_filter(sal, sigma=8)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.astype(np.float32)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Canny edge map normalized to [0,1]."""
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    return edges


def _template_score(saliency: np.ndarray, template: np.ndarray) -> float:
    """
    Score = weighted overlap between saliency and a composition template.
    Template is a float32 mask of the same shape, values in [0,1].
    Returns Pearson r mapped to [0,1], then boosted by mean overlap.
    """
    s = saliency.flatten()
    t = template.flatten()
    if t.sum() < 1:
        return 0.0
    # Pearson correlation
    try:
        r, _ = pearsonr(s, t)
    except Exception:
        r = 0.0
    r = float(np.clip(r, 0, 1))
    # Weighted overlap: how much saliency lands inside the template regions
    overlap = float((saliency * template).sum() / (template.sum() + 1e-8))
    # Combine: 60% correlation, 40% overlap, both already [0,1]
    score = 0.6 * r + 0.4 * overlap
    return float(np.clip(score, 0.0, 1.0))


def _line_density_score(edges: np.ndarray, angle_range, tolerance=10) -> float:
    """
    Use Hough transform to find how many lines fall within a given angle range.
    angle_range: (min_deg, max_deg) — angles measured from horizontal.
    Returns fraction of detected lines in that angular range.
    """
    lines = cv2.HoughLines(
        (edges * 255).astype(np.uint8), 1, np.pi / 180, threshold=80
    )
    if lines is None:
        return 0.0
    angles = np.degrees(lines[:, 0, 1])
    # Normalize angles to [0, 180)
    angles = angles % 180
    lo, hi = angle_range
    in_range = np.sum((angles >= lo - tolerance) & (angles <= hi + tolerance))
    score = in_range / max(len(angles), 1)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Template builders  (all return float32 arrays of shape H x W)
# ---------------------------------------------------------------------------

def _make_rule_of_thirds(H, W):
    mask = np.zeros((H, W), np.float32)
    # Tight gaussian blobs at the four intersection points
    sigma = min(H, W) * 0.03
    for y in [H // 3, 2 * H // 3]:
        for x in [W // 3, 2 * W // 3]:
            tmp = np.zeros((H, W), np.float32)
            tmp[y, x] = 1.0
            mask += gaussian_filter(tmp, sigma=sigma)
    # Faint grid lines as secondary cue
    thick = max(1, H // 12)
    for y in [H // 3, 2 * H // 3]:
        mask[y-thick, :] += 1.0
        mask[y+thick, :] + 1.0
    for x in [W // 3, 2 * W // 3]:
        mask[:, x-thick] += 1.0
        mask[:, x+thick] += 1.0
    return mask / (mask.max() + 1e-8)


def _make_golden_section(H, W):
    """Golden ratio grid: phi ≈ 0.618"""
    phi = 0.6180339887
    mask = np.zeros((H, W), np.float32)
    # Tight blobs at the four phi intersections
    sigma = min(H, W) * 0.03
    for fy in [phi, 1 - phi]:
        for fx in [phi, 1 - phi]:
            tmp = np.zeros((H, W), np.float32)
            tmp[int(H * fy), int(W * fx)] = 1.0
            mask += gaussian_filter(tmp, sigma=sigma)
    # Faint grid lines
    for frac in [phi, 1 - phi]:
        mask[int(H * frac), :] += 1
        mask[:, int(W * frac)] += 1
    return mask / (mask.max() + 1e-8)


def _make_golden_spiral(H, W):
    """
    Golden spiral as connected quarter-circle arcs.
    Repeatedly cuts squares from the rectangle; arc center is the
    inner corner shared between the square and the remaining rect.
    Each arc endpoint exactly meets the next arc's start point.
    """
    mask = np.zeros((H, W), np.float32)
    x1, y1, x2, y2 = 0.0, 0.0, float(W), float(H)
    for i in range(10):
        w = x2 - x1
        h = y2 - y1
        if w < 1 or h < 1:
            break
        phase = i % 4
        if phase == 0:      # cut square from left; arc center = bottom-right of square
            sq = h
            cx, cy = x1 + sq, y2
            sa, ea = math.pi, 1.5 * math.pi
            x1 += sq
        elif phase == 1:    # cut square from top; arc center = bottom-left of square
            sq = w
            cx, cy = x1, y1 + sq
            sa, ea = 1.5 * math.pi, 2.0 * math.pi
            y1 += sq
        elif phase == 2:    # cut square from right; arc center = top-left of square
            sq = h
            cx, cy = x2 - sq, y1
            sa, ea = 0.0, 0.5 * math.pi
            x2 -= sq
        else:               # cut square from bottom; arc center = top-right of square
            sq = w
            cx, cy = x2, y2 - sq
            sa, ea = 0.5 * math.pi, math.pi
            y2 -= sq
        pts = max(80, int(sq * 8))
        for j in range(pts + 1):
            a = sa + (ea - sa) * j / pts
            px = int(round(cx + sq * math.cos(a)))
            py = int(round(cy + sq * math.sin(a)))
            if 0 <= px < W and 0 <= py < H:
                mask[py, px] += 1.0
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.016)
    return mask / (mask.max() + 1e-8)


def _make_spiral_section(H, W):
    """Fibonacci / Spiral Section — rectangular partitioning (like golden spiral but via rects)."""
    mask = np.zeros((H, W), np.float32)
    phi = 1.6180339887
    x0, y0, w, h = 0, 0, W, H
    for _ in range(6):
        if w >= h:
            sq = h
            # vertical dividing line
            lx = int(x0 + w / phi)
            mask[:, lx] += 0.5 if 0 <= lx < W else 0
            w = W - lx
            x0 = lx
        else:
            sq = w
            ly = int(y0 + h / phi)
            mask[ly, :] += 0.5 if 0 <= ly < H else 0
            h = H - ly
            y0 = ly
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.03)
    return mask / (mask.max() + 1e-8)


def _make_golden_triangles(H, W):
    """
    Main diagonal (TL→BR) plus perpendicular lines dropped from each
    opposite corner to the diagonal — matching the reference image pattern.
    """
    mask = np.zeros((H, W), np.float32)
    def draw_line(y1, x1, y2, x2, weight=1.0):
        n = int(math.hypot(x2 - x1, y2 - y1) * 3)
        for t in np.linspace(0, 1, max(n, 2)):
            y = int(y1 + t * (y2 - y1))
            x = int(x1 + t * (x2 - x1))
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] += weight
    # Main diagonal
    draw_line(0, 0, H - 1, W - 1, 1.0)
    # Foot of perpendicular from top-right corner (0, W-1) onto diagonal y = (H/W)*x
    m = H / W
    xf1 = (W - 1) / (1 + m ** 2)
    yf1 = m * xf1
    draw_line(0, W - 1, int(yf1), int(xf1), 0.9)
    # Foot of perpendicular from bottom-left corner (H-1, 0) onto diagonal
    xf2 = m * (H - 1) / (1 + m ** 2)
    yf2 = m * xf2
    draw_line(H - 1, 0, int(yf2), int(xf2), 0.9)
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.018)
    return mask / (mask.max() + 1e-8)


def _make_harmonious_triangles(H, W):
    """
    Main diagonal + two lines from opposite corners perpendicular to diagonal.
    """
    mask = np.zeros((H, W), np.float32)
    def draw_line(y1, x1, y2, x2, weight=1.0):
        length = int(math.hypot(x2 - x1, y2 - y1)) * 2
        for t in np.linspace(0, 1, length):
            y = int(y1 + t * (y2 - y1))
            x = int(x1 + t * (x2 - x1))
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] += weight
    draw_line(0, 0, H - 1, W - 1)  # main diagonal
    # Perpendicular from top-right to main diagonal
    # Foot of perpendicular from (0, W-1) on y=x*(H/W):
    m = H / W
    xf = (W - 1 + m * 0) / (1 + m * m) * (1 + 0)
    xf = (W - 1) / (1 + m ** 2)
    yf = m * xf
    draw_line(0, W - 1, int(yf), int(xf), 0.8)
    # Perpendicular from bottom-left
    xf2 = (0 + m * (H - 1)) / (1 + m * m)
    yf2 = m * xf2
    draw_line(H - 1, 0, int(yf2), int(xf2), 0.8)
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.025)
    return mask / (mask.max() + 1e-8)


def _make_cross(H, W):
    mask = np.zeros((H, W), np.float32)
    mask[H // 2, :] = 1.0
    mask[:, W // 2] = 1.0
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.02)
    return mask / (mask.max() + 1e-8)


def _make_focal_mass(H, W):
    """Dense cluster near center-slightly-off-center."""
    mask = np.zeros((H, W), np.float32)
    tmp = np.zeros((H, W), np.float32)
    tmp[int(H * 0.45), int(W * 0.5)] = 1.0
    mask = gaussian_filter(tmp, sigma=min(H, W) * 0.15)
    return mask / (mask.max() + 1e-8)


def _make_v_arrangement(H, W):
    """V shape: two lines converging from top corners to center-bottom."""
    mask = np.zeros((H, W), np.float32)
    def draw_line(y1, x1, y2, x2):
        length = int(math.hypot(x2 - x1, y2 - y1)) * 2
        for t in np.linspace(0, 1, length):
            y = int(y1 + t * (y2 - y1))
            x = int(x1 + t * (x2 - x1))
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = 1.0
    draw_line(0, 0, H - 1, W // 2)
    draw_line(0, W - 1, H - 1, W // 2)
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.03)
    return mask / (mask.max() + 1e-8)


def _make_diagonal(H, W):
    """Strong single diagonal, top-left to bottom-right."""
    mask = np.zeros((H, W), np.float32)
    for t in np.linspace(0, 1, 3000):
        y, x = int(t * H), int(t * W)
        if 0 <= y < H and 0 <= x < W:
            mask[y, x] = 1.0
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.03)
    return mask / (mask.max() + 1e-8)


def _make_radial(H, W):
    """Lines radiating from center."""
    mask = np.zeros((H, W), np.float32)
    cy, cx = H // 2, W // 2
    for angle in np.linspace(0, 2 * math.pi, 12, endpoint=False):
        for r in range(1, int(math.hypot(H, W) // 2)):
            y = int(cy + r * math.sin(angle))
            x = int(cx + r * math.cos(angle))
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = 1.0
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.025)
    return mask / (mask.max() + 1e-8)


def _make_l_arrangement(H, W):
    """
    Centered L-shape: vertical stroke from ~20% to ~75% height,
    horizontal stroke extending right from its base — matching the reference image.
    """
    mask = np.zeros((H, W), np.float32)
    lw = max(2, min(H, W) // 80)
    cx   = int(W * 0.35)
    top_y  = int(H * 0.18)
    bot_y  = int(H * 0.76)
    right_x = int(W * 0.74)
    mask[top_y:bot_y, cx - lw:cx + lw] = 1.0
    mask[bot_y - lw:bot_y + lw, cx:right_x] = 1.0
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.025)
    return mask / (mask.max() + 1e-8)


def _make_compound_curve(H, W):
    """S-curve: sinusoidal path from top to bottom."""
    mask = np.zeros((H, W), np.float32)
    for i, y in enumerate(range(0, H)):
        t = y / H
        x = int(W * 0.5 + W * 0.2 * math.sin(2 * math.pi * t))
        if 0 <= x < W:
            mask[y, x] = 1.0
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.04)
    return mask / (mask.max() + 1e-8)


def _make_pyramid(H, W):
    """Inset triangle: apex at top-center, base near bottom — matching reference."""
    mask = np.zeros((H, W), np.float32)
    def draw_line(y1, x1, y2, x2):
        n = int(math.hypot(x2 - x1, y2 - y1) * 3)
        for t in np.linspace(0, 1, max(n, 2)):
            y = int(y1 + t * (y2 - y1))
            x = int(x1 + t * (x2 - x1))
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = 1.0
    apex_x, apex_y = W // 2, int(H * 0.12)
    base_y = int(H * 0.88)
    base_l, base_r = int(W * 0.12), int(W * 0.88)
    draw_line(apex_y, apex_x, base_y, base_l)
    draw_line(apex_y, apex_x, base_y, base_r)
    draw_line(base_y, base_l, base_y, base_r)
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.022)
    return mask / (mask.max() + 1e-8)


def _make_circular(H, W):
    """Circle centered in frame."""
    mask = np.zeros((H, W), np.float32)
    cy, cx = H // 2, W // 2
    r = min(H, W) // 3
    for angle in np.linspace(0, 2 * math.pi, 3000):
        y = int(cy + r * math.sin(angle))
        x = int(cx + r * math.cos(angle))
        if 0 <= y < H and 0 <= x < W:
            mask[y, x] = 1.0
    mask = gaussian_filter(mask, sigma=min(H, W) * 0.025)
    return mask / (mask.max() + 1e-8)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

TEMPLATE_BUILDERS = {
    "rule_of_thirds":       _make_rule_of_thirds,
    "golden_section":       _make_golden_section,
    "golden_triangles":     _make_golden_triangles,
    "spiral_section":       _make_spiral_section,
    "golden_spiral":        _make_golden_spiral,
    "harmonious_triangles": _make_harmonious_triangles,
    "cross":                _make_cross,
    "focal_mass":           _make_focal_mass,
    "v_arrangement":        _make_v_arrangement,
    "diagonal":             _make_diagonal,
    "radial":               _make_radial,
    "l_arrangement":        _make_l_arrangement,
    "compound_curve":       _make_compound_curve,
    "pyramid":              _make_pyramid,
    "circular":             _make_circular,
}


def analyze_composition(
    image_path: str,
    target_w: int = 960,
    target_h: int = 540,
    top_n: int = None,
) -> dict:
    """
    Analyze the photographic composition of a 16:9 PNG image.

    Parameters
    ----------
    image_path : str
        Path to the input PNG (any aspect ratio is accepted but 16:9 is ideal).
    target_w : int
        Internal processing width (default 960).
    target_h : int
        Internal processing height (default 540).
    top_n : int or None
        If set, return only the top-N scoring compositions.

    Returns
    -------
    dict
        {composition_name: score}  — scores in [0.0, 1.0], sorted descending.
    """
    img, gray, saliency = _load_and_prep(image_path, target_w, target_h)
    H, W = saliency.shape

    scores = {}
    for name, builder in TEMPLATE_BUILDERS.items():
        template = builder(H, W)
        scores[name] = _template_score(saliency, template)

    # Normalize so the max score = 1.0, rest are relative
    max_s = max(scores.values()) if scores else 1.0
    if max_s > 0:
        scores = {k: round(v / max_s, 4) for k, v in scores.items()}

    # Sort descending
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    if top_n is not None:
        scores = dict(list(scores.items())[:top_n])

    return scores


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python composition_analyzer.py <image.png> [top_n]")
        sys.exit(1)
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    result = analyze_composition(sys.argv[1], top_n=top_n)
    print(json.dumps(result, indent=2))