"""
render_compositions.py
-----------------------
Renders two grid images:
  1. Original photo with each composition overlay
  2. Saliency map with each composition overlay

Usage:
    python render_compositions.py <image.png>
    -> outputs: compositions_on_photo.png
                compositions_on_saliency.png
"""

import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Import all template builders and helpers from the analyzer
from composition_analyzer import (
    _load_and_prep,
    _template_score,
    TEMPLATE_BUILDERS,
)

PRETTY_NAMES = {
    "rule_of_thirds":       "Rule of Thirds",
    "golden_section":       "Golden Section",
    "golden_triangles":     "Golden Triangles",
    "spiral_section":       "Spiral Section",
    "golden_spiral":        "Golden Spiral",
    "harmonious_triangles": "Harmonious Triangles",
    "cross":                "Cross",
    "focal_mass":           "Focal Mass",
    "v_arrangement":        "V-Arrangement",
    "diagonal":             "Diagonal",
    "radial":               "Radial",
    "l_arrangement":        "L-Arrangement",
    "compound_curve":       "Compound Curve",
    "pyramid":              "Pyramid",
    "circular":             "Circular",
}

OVERLAY_CMAP = LinearSegmentedColormap.from_list(
    "overlay", [(0, 0, 0, 0), (1.0, 0.85, 0.0, 0.72)]  # transparent -> gold
)


def render_grid(image_path: str, target_w=960, target_h=540):
    img_bgr, gray, saliency = _load_and_prep(image_path, target_w, target_h)
    H, W = saliency.shape

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Saliency as RGB (viridis colormap feel, dark bg)
    sal_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    sal_rgb = plt.cm.inferno(sal_norm)[:, :, :3]  # HxWx3 float

    # Build templates and scores
    templates = {}
    scores = {}
    for name, builder in TEMPLATE_BUILDERS.items():
        t = builder(H, W)
        templates[name] = t
        scores[name] = _template_score(saliency, t)

    max_s = max(scores.values()) or 1.0
    norm_scores = {k: v / max_s for k, v in scores.items()}

    n = len(TEMPLATE_BUILDERS)
    ncols = 3
    nrows = math.ceil(n / ncols)

    template_list = list(templates.items())  # [(name, array), ...]

    def make_grid(background_imgs, title_str, out_path):
        """background_imgs: list of HxWx3 float arrays (one per composition)"""
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 5.2, nrows * 3.4 + 0.7),
            facecolor="#1a1a1a",
        )
        fig.suptitle(title_str, color="white", fontsize=16, fontweight="bold", y=0.995)

        axes_flat = axes.flatten()

        for idx, (name, template) in enumerate(template_list):
            ax = axes_flat[idx]
            bg = background_imgs[idx]

            # Show background
            ax.imshow(bg, aspect="auto")

            # Overlay composition template
            # t_norm = (template - template.min()) / (template.max() - template.min() + 1e-8)
            ax.imshow(template, cmap=OVERLAY_CMAP, alpha=0.85, aspect="auto")

            score = norm_scores[name]
            bar_color = plt.cm.RdYlGn(score)

            # Title bar
            ax.set_title(
                PRETTY_NAMES[name],
                color="white",
                fontsize=9.5,
                fontweight="bold",
                pad=3,
                loc="left",
            )

            # Score badge — bottom-right
            ax.text(
                0.98, 0.04,
                f"{score:.2f}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9, fontweight="bold",
                color="black",
                bbox=dict(
                    facecolor=bar_color,
                    edgecolor="none",
                    boxstyle="round,pad=0.3",
                    alpha=0.92,
                ),
            )

            # Score bar along bottom edge
            bar_w = score
            ax.add_patch(mpatches.FancyArrowPatch(
                (0, 0), (bar_w, 0),
                transform=ax.transAxes,
                color=bar_color,
                lw=4, arrowstyle="-",
                clip_on=False,
                zorder=10,
            ))

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
                spine.set_linewidth(0.8)

        # Hide unused axes
        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.995], h_pad=0.8, w_pad=0.5)
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out_path}")

    # --- Grid 1: original photo backgrounds ---
    photo_bgs = [img_rgb.astype(np.float32) / 255.0] * n
    make_grid(
        photo_bgs,
        "Composition overlays — original photo  (gold = template mask, score = saliency match)",
        "compositions_on_photo.png",
    )

    # --- Grid 2: saliency map backgrounds ---
    sal_bgs = [sal_rgb] * n
    make_grid(
        sal_bgs,
        "Composition overlays — saliency map  (gold = template mask, brighter = more visually salient)",
        "compositions_on_saliency.png",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_compositions.py <image.png>")
        sys.exit(1)
    render_grid(sys.argv[1])