import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── helper ──────────────────────────────────────────────
def md(src):
    return nbf.v4.new_markdown_cell(src)

def code(src):
    return nbf.v4.new_code_cell(src)

# ════════════════════════════════════════════════════════
# CELL 1 — Title
# ════════════════════════════════════════════════════════
cells.append(md("""# 🧺 Research-Grade Geometry-Aware Clothing Folding Planner
### Final Revised Version — v3.2 (Canny Enhanced + Drive Save + Deploy Export)

**Pipeline:**
```
Input Image
    ↓ load_image (np.fromfile + imdecode)
    ↓ validate_image
    ↓ GrabCut Segmentation (iterative rect + center refinement)
    ↓ Enhanced Canny Refinement
       (Denoising → Bilateral → CLAHE → Adaptive Canny → Morphology)
    ↓ Largest Contour Extraction (min area validation)
    ↓ Safe Orientation Normalization (PCA + angle clamping)
    ↓ Shape Analysis (centroid + solidity + aspect ratio)
    ↓ Class-Aware Fold Planning (geometry-aware ratios)
    ↓ Mask-Aware Dashed Arrow Rendering
    ↓ Sequential Fold Visualization + Auto Save to Drive
    ↓ Model Export (JSON + Pickle) for Web Deploy
```
"""))

# ════════════════════════════════════════════════════════
# CELL 2 — Install
# ════════════════════════════════════════════════════════
cells.append(md("## 📦 Cell 1 — Install Dependencies"))
cells.append(code("""\
!pip install opencv-python matplotlib pandas tqdm scikit-image -q

print("✅ Dependencies installed")
"""))

# ════════════════════════════════════════════════════════
# CELL 3 — Imports
# ════════════════════════════════════════════════════════
cells.append(md("## 📚 Cell 2 — Import Libraries"))
cells.append(code("""\
import os
import cv2
import math
import json
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm
from pathlib import Path
from skimage import morphology as skmorph

warnings.filterwarnings("ignore")

print("=" * 50)
print("LIBRARY VERSION")
print("=" * 50)
print(f"OpenCV  : {cv2.__version__}")
print(f"Numpy   : {np.__version__}")
print(f"Pandas  : {pd.__version__}")
print("=" * 50)
print("✅ Libraries ready")
"""))

# ════════════════════════════════════════════════════════
# CELL 4 — Mount Drive
# ════════════════════════════════════════════════════════
cells.append(md("## ☁️ Cell 3 — Mount Google Drive"))
cells.append(code("""\
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive mounted")
"""))

# ════════════════════════════════════════════════════════
# CELL 5 — Config
# ════════════════════════════════════════════════════════
cells.append(md("## ⚙️ Cell 4 — Dataset Configuration & Output Setup"))
cells.append(code("""\
# ── Paths ──────────────────────────────────────────────
DATASET_PATH = "/content/drive/MyDrive/kaggle/train"

PROCESS_CLASSES = [
    "baju_lengan_panjang",
    "baju_lengan_pendek",
    "celana_panjang",
    "celana_pendek"
]

BAJU_CLASSES   = ["baju_lengan_panjang", "baju_lengan_pendek"]
CELANA_CLASSES = ["celana_panjang", "celana_pendek"]

# ── Output folders ──────────────────────────────────────
OUTPUT_PATH        = "/content/drive/MyDrive/folding_output"
OUTPUT_IMAGES_PATH = os.path.join(OUTPUT_PATH, "images")
OUTPUT_MODEL_PATH  = os.path.join(OUTPUT_PATH, "model")

for folder in [OUTPUT_PATH, OUTPUT_IMAGES_PATH, OUTPUT_MODEL_PATH]:
    os.makedirs(folder, exist_ok=True)

# ── Config ──────────────────────────────────────────────
CONFIG = {
    # image
    "MIN_IMAGE_SIZE"         : 100,
    "MIN_CONTOUR_AREA_RATIO" : 0.05,

    # grabcut
    "GRABCUT_ITERATION"      : 10,       # naik dari 5 → lebih presisi
    "GRABCUT_MARGIN"         : 0.04,     # margin rect dari tepi

    # canny enhancement
    "BILATERAL_D"            : 9,
    "BILATERAL_SIGMA_COLOR"  : 80,
    "BILATERAL_SIGMA_SPACE"  : 80,
    "CLAHE_CLIP"             : 3.0,      # naik dari 2.0
    "CLAHE_GRID"             : (8, 8),
    "CANNY_SIGMA"            : 0.28,     # lebih ketat dari 0.33
    "MORPH_CLOSE_KERNEL"     : 9,        # naik dari 7
    "MORPH_DILATE_KERNEL"    : 17,       # naik dari 15
    "MORPH_ERODE_KERNEL"     : 11,       # naik dari 10

    # orientation
    "MAX_ROTATION_ANGLE"     : 20,

    # experiment
    "SAMPLES_PER_CLASS"      : 3,
    "SHOW_DEBUG"             : True,
    "RANDOM_SEED"            : 42
}

random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])

print(f"📁 Output       : {OUTPUT_PATH}")
print(f"🖼️  Images       : {OUTPUT_IMAGES_PATH}")
print(f"🤖 Model export : {OUTPUT_MODEL_PATH}")
print("✅ Config ready")
"""))

# ════════════════════════════════════════════════════════
# CELL 6 — Utils
# ════════════════════════════════════════════════════════
cells.append(md("## 🛠️ Cell 5 — Utility Functions"))
cells.append(code("""\
def debug_print(message):
    if CONFIG["SHOW_DEBUG"]:
        print(f"  [DEBUG] {message}")


def load_image(image_path):
    \"\"\"Load image safely using np.fromfile (handles unicode path).\"\"\"
    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Decode failed: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  [ERROR] Load failed: {image_path} → {e}")
        return None


def validate_image(image_rgb):
    \"\"\"Reject images smaller than MIN_IMAGE_SIZE on either axis.\"\"\"
    h, w = image_rgb.shape[:2]
    return h >= CONFIG["MIN_IMAGE_SIZE"] and w >= CONFIG["MIN_IMAGE_SIZE"]


print("✅ Utility functions ready")
"""))

# ════════════════════════════════════════════════════════
# CELL 7 — Enhanced Canny
# ════════════════════════════════════════════════════════
cells.append(md("""\
## 🔬 Cell 6 — Enhanced Canny Refinement

Perbaikan dari v3.1:
- **CLAHE clip lebih tinggi** → kontras lebih tajam
- **Fast Non-Local Means Denoising** sebelum bilateral → noise berkurang sebelum edge detection
- **Adaptive Canny sigma** lebih ketat (0.28) → edge lebih selektif
- **Remove small objects** via skimage → contour noise dibersihkan
- **Kernel morphology lebih besar** → boundary lebih rapat
"""))
cells.append(code("""\
def auto_canny(image, sigma=None):
    \"\"\"
    Otsu-guided Canny: threshold otomatis dari distribusi pixel.
    Sigma lebih kecil = threshold lebih ketat = edge lebih sedikit tapi lebih akurat.
    \"\"\"
    if sigma is None:
        sigma = CONFIG["CANNY_SIGMA"]

    # Otsu threshold sebagai median yang lebih robust
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val = _

    lower = int(max(0,   otsu_val * (1.0 - sigma)))
    upper = int(min(255, otsu_val * (1.0 + sigma)))

    # fallback ke median jika otsu gagal
    if lower == upper == 0:
        median = np.median(image)
        lower  = int(max(0,   (1.0 - sigma) * median))
        upper  = int(min(255, (1.0 + sigma) * median))

    debug_print(f"Canny thresholds: lower={lower}, upper={upper}")
    return cv2.Canny(image, lower, upper)


def enhance_gray(gray_image):
    \"\"\"
    Multi-step preprocessing untuk memaksimalkan kualitas Canny:
    1. Fast Non-Local Means  → hilangkan noise sebelum filtering
    2. Bilateral filter      → smooth + preserve edge
    3. CLAHE                 → tingkatkan kontras lokal
    \"\"\"
    # Step 1: Denoise
    denoised = cv2.fastNlMeansDenoising(
        gray_image, h=10, templateWindowSize=7, searchWindowSize=21
    )

    # Step 2: Bilateral
    bilateral = cv2.bilateralFilter(
        denoised,
        CONFIG["BILATERAL_D"],
        CONFIG["BILATERAL_SIGMA_COLOR"],
        CONFIG["BILATERAL_SIGMA_SPACE"]
    )

    # Step 3: CLAHE
    clahe     = cv2.createCLAHE(
        clipLimit=CONFIG["CLAHE_CLIP"],
        tileGridSize=CONFIG["CLAHE_GRID"]
    )
    enhanced  = clahe.apply(bilateral)

    return enhanced


def refine_silhouette_with_canny(grabcut_mask, gray_image):
    \"\"\"
    Refined Canny boundary fusion:
    1. Enhance gray
    2. Auto-Canny (Otsu-guided)
    3. Filter Canny edges ke zona boundary GrabCut saja
    4. Combine + morphological close
    5. Flood fill untuk isi lubang interior
    6. Remove small noise objects (skimage)
    \"\"\"
    enhanced     = enhance_gray(gray_image)
    canny_edges  = auto_canny(enhanced)

    # Boundary zone dari GrabCut mask
    k_dilate     = np.ones((CONFIG["MORPH_DILATE_KERNEL"],) * 2, np.uint8)
    k_erode      = np.ones((CONFIG["MORPH_ERODE_KERNEL"],)  * 2, np.uint8)

    boundary_zone   = cv2.subtract(
        cv2.dilate(grabcut_mask, k_dilate),
        cv2.erode(grabcut_mask,  k_erode)
    )

    canny_filtered  = cv2.bitwise_and(canny_edges, canny_edges, mask=boundary_zone)
    combined        = cv2.bitwise_or(grabcut_mask, canny_filtered)

    # Morph close
    k_close  = np.ones((CONFIG["MORPH_CLOSE_KERNEL"],) * 2, np.uint8)
    refined  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    # Flood fill dari sudut (hapus background sisa)
    flood      = refined.copy()
    h, w       = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv  = cv2.bitwise_not(flood)
    final_mask = cv2.bitwise_or(refined, flood_inv)

    # Remove small noise via skimage
    bool_mask  = final_mask > 0
    cleaned    = skmorph.remove_small_objects(bool_mask, min_size=500)
    cleaned    = skmorph.remove_small_holes(cleaned, area_threshold=1000)
    final_mask = (cleaned * 255).astype(np.uint8)

    return {
        "enhanced"      : enhanced,
        "canny_edges"   : canny_edges,
        "boundary_zone" : boundary_zone,
        "canny_filtered": canny_filtered,
        "refined_mask"  : final_mask
    }


print("✅ Enhanced Canny functions ready")
"""))

# ════════════════════════════════════════════════════════
# CELL 8 — Silhouette + Contour
# ════════════════════════════════════════════════════════
cells.append(md("## ✂️ Cell 7 — Silhouette Extraction & Contour"))
cells.append(code("""\
def extract_clothing_silhouette(image_rgb):
    \"\"\"
    GrabCut dengan 2 pass:
    - Pass 1: rect init (standar)
    - Pass 2: mask refinement dari hasil pass 1 → edge lebih rapat
    \"\"\"
    h, w        = image_rgb.shape[:2]
    mask        = np.zeros((h, w), np.uint8)
    bgd_model   = np.zeros((1, 65), np.float64)
    fgd_model   = np.zeros((1, 65), np.float64)
    m           = CONFIG["GRABCUT_MARGIN"]

    rect = (
        int(w * m),
        int(h * m),
        int(w * (1 - 2 * m)),
        int(h * (1 - 2 * m))
    )

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Pass 1
    cv2.grabCut(
        image_bgr, mask, rect,
        bgd_model, fgd_model,
        CONFIG["GRABCUT_ITERATION"],
        cv2.GC_INIT_WITH_RECT
    )

    # Pass 2: mask refinement
    cv2.grabCut(
        image_bgr, mask, rect,
        bgd_model, fgd_model,
        3,
        cv2.GC_INIT_WITH_MASK
    )

    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    gray        = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    refinement  = refine_silhouette_with_canny(binary_mask, gray)

    return {
        "grabcut_mask"  : binary_mask,
        "enhanced"      : refinement["enhanced"],
        "canny_edges"   : refinement["canny_edges"],
        "boundary_zone" : refinement["boundary_zone"],
        "canny_filtered": refinement["canny_filtered"],
        "refined_mask"  : refinement["refined_mask"]
    }


def extract_largest_contour(binary_mask):
    \"\"\"Return largest contour atau None kalau terlalu kecil.\"\"\"
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None

    largest  = max(contours, key=cv2.contourArea)
    h, w     = binary_mask.shape[:2]
    min_area = h * w * CONFIG["MIN_CONTOUR_AREA_RATIO"]
    area     = cv2.contourArea(largest)

    if area < min_area:
        debug_print(f"Contour too small: {area:.0f}px² (min {min_area:.0f})")
        return None

    return largest


def detect_hanger(contour, image_shape):
    \"\"\"Heuristic: deteksi hanger dari top-margin ratio dan top-width ratio.\"\"\"
    try:
        x, y, w, h  = cv2.boundingRect(contour)
        ih, iw      = image_shape[:2]
        top_margin  = y / ih
        pts         = contour.reshape(-1, 2)
        top_pts     = pts[pts[:, 1] < y + h * 0.10]
        if len(top_pts) == 0:
            return False
        top_width   = top_pts[:, 0].max() - top_pts[:, 0].min()
        top_ratio   = top_width / w if w > 0 else 1.0
        return top_margin < 0.05 and top_ratio < 0.15
    except Exception as e:
        debug_print(f"Hanger detection fallback: {e}")
        return False


print("✅ Silhouette & contour functions ready")
"""))

# ════════════════════════════════════════════════════════
# CELL 9 — Orientation + Shape
# ════════════════════════════════════════════════════════
cells.append(md("## 📐 Cell 8 — Orientation Normalization & Shape Analysis"))
cells.append(code("""\
def normalize_orientation(image_rgb, contour):
    \"\"\"PCA-based rotation dengan angle clamping.\"\"\"
    pts  = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)

    angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    h, w  = image_rgb.shape[:2]

    safe  = angle
    if safe >  90: safe -= 180
    if safe < -90: safe += 180

    if h > w and abs(safe) > 45:
        safe = safe - 90 if safe > 0 else safe + 90

    if abs(safe) > CONFIG["MAX_ROTATION_ANGLE"]:
        debug_print(f"Rotation clamped: {safe:.1f}° > {CONFIG['MAX_ROTATION_ANGLE']}°")
        return image_rgb.copy(), 0.0

    M       = cv2.getRotationMatrix2D((w // 2, h // 2), safe, 1.0)
    rotated = cv2.warpAffine(image_rgb, M, (w, h))
    return rotated, safe


def analyze_shape(contour):
    \"\"\"Return shape metrics: bbox, centroid, aspect_ratio, solidity.\"\"\"
    x, y, w, h = cv2.boundingRect(contour)
    M          = cv2.moments(contour)

    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    area      = cv2.contourArea(contour)
    hull      = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity  = area / hull_area if hull_area > 0 else 0
    aspect    = w / h if h > 0 else 0

    return {
        "x": x, "y": y,
        "width": w, "height": h,
        "bbox": (x, y, w, h),
        "centroid": (cx, cy),
        "aspect_ratio": aspect,
        "area": area,
        "solidity": solidity
    }


print("✅ Orientation & shape functions ready")
"""))

# ════════════════════════════════════════════════════════
# CELL 10 — Fold Rules + Planner
# ════════════════════════════════════════════════════════
cells.append(md("## 📏 Cell 9 — Fold Rules & Planner"))
cells.append(code("""\
FOLD_RULES = {
    "baju_lengan_panjang": {
        "vertical_left_ratio"     : 0.30,
        "vertical_right_ratio"    : 0.70,
        "horizontal_bottom_ratio" : 0.75
    },
    "baju_lengan_pendek": {
        "vertical_left_ratio"     : 0.33,
        "vertical_right_ratio"    : 0.67,
        "horizontal_bottom_ratio" : 0.72
    },
    "celana_panjang": {
        "vertical_center_ratio"   : 0.50,
        "horizontal_ratio"        : 0.55
    },
    "celana_pendek": {
        "vertical_center_ratio"   : 0.50,
        "horizontal_ratio"        : 0.45
    }
}


def get_geometry_aware_ratio(class_name, aspect_ratio):
    base = dict(FOLD_RULES[class_name])
    if class_name in BAJU_CLASSES:
        correction = (aspect_ratio - 1.0) * 0.03
        base["vertical_left_ratio"]  += correction
        base["vertical_right_ratio"] -= correction
    return base


def draw_dashed_arrow(image, pt1, pt2, mask, color,
                      thickness=2, dash_length=15, gap_length=8):
    x1, y1 = pt1
    x2, y2 = pt2
    dist   = np.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return

    ux, uy  = (x2 - x1) / dist, (y2 - y1) / dist
    current = 0.0
    drawing = True

    while current < dist - 25:
        if drawing:
            end = min(current + dash_length, dist - 25)
            sx, sy = int(x1 + current * ux), int(y1 + current * uy)
            ex, ey = int(x1 + end     * ux), int(y1 + end     * uy)
            if 0 <= sy < mask.shape[0] and 0 <= sx < mask.shape[1] and mask[sy, sx] > 0:
                cv2.line(image, (sx, sy), (ex, ey), color, thickness)
        current += dash_length if drawing else gap_length
        drawing  = not drawing

    arrow_start = (int(x2 - ux * 25), int(y2 - uy * 25))
    cv2.arrowedLine(image, arrow_start, (x2, y2), color, thickness, tipLength=0.5)


def generate_folding_guideline(image_rgb, mask, shape_info, class_name):
    if class_name not in FOLD_RULES:
        print(f"  [WARNING] Unknown class: {class_name}")
        return image_rgb.copy(), []

    output = image_rgb.copy()
    x, y, w, h = shape_info["bbox"]
    cx, cy     = shape_info["centroid"]
    rules      = get_geometry_aware_ratio(class_name, shape_info["aspect_ratio"])
    fold_lines = []

    if class_name in BAJU_CLASSES:
        left_x   = int(x + w * rules["vertical_left_ratio"])
        right_x  = int(x + w * rules["vertical_right_ratio"])
        bottom_y = int(y + h * rules["horizontal_bottom_ratio"])
        fold_lines = [
            {"from": (x,     cy), "to": (left_x,  cy),      "color": (255, 165, 0),  "label": "Fold Left"},
            {"from": (x + w, cy), "to": (right_x, cy),      "color": (0,   220, 220), "label": "Fold Right"},
            {"from": (cx, y + h), "to": (cx,      bottom_y),"color": (255,  50, 50),  "label": "Fold Bottom"},
        ]

    elif class_name in CELANA_CLASSES:
        center_x = int(x + w * rules["vertical_center_ratio"])
        fold_y   = int(y + h * rules["horizontal_ratio"])
        fold_lines = [
            {"from": (x,  cy),    "to": (center_x, cy),    "color": (255, 165, 0),  "label": "Fold Center"},
            {"from": (cx, y + h), "to": (cx,       fold_y),"color": (0,   220, 220), "label": "Fold Bottom"},
        ]

    for fold in fold_lines:
        draw_dashed_arrow(output, fold["from"], fold["to"], mask, fold["color"], thickness=3)

    return output, fold_lines


print("✅ Fold rules & planner ready")
"""))

# ════════════════════════════════════════════════════════
# CELL 11 — Full Pipeline
# ════════════════════════════════════════════════════════
cells.append(md("## 🔄 Cell 10 — Full Processing Pipeline"))
cells.append(code("""\
def process_image(image_path, class_name):
    \"\"\"End-to-end pipeline. Returns result dict atau None kalau gagal/skip.\"\"\"
    try:
        image_rgb = load_image(image_path)
        if image_rgb is None:
            return None

        if not validate_image(image_rgb):
            debug_print("SKIP: image too small")
            return None

        silhouette = extract_clothing_silhouette(image_rgb)

        contour = extract_largest_contour(silhouette["refined_mask"])
        if contour is None:
            debug_print("SKIP: contour not found")
            return None

        if detect_hanger(contour, image_rgb.shape):
            debug_print("SKIP: hanger detected")
            return None

        normalized, angle = normalize_orientation(image_rgb, contour)
        shape_info        = analyze_shape(contour)

        fold_output, fold_lines = generate_folding_guideline(
            normalized, silhouette["refined_mask"], shape_info, class_name
        )

        contour_image = image_rgb.copy()
        cv2.drawContours(contour_image, [contour], -1, (255, 0, 0), 2)

        return {
            "original"      : image_rgb,
            "grabcut_mask"  : silhouette["grabcut_mask"],
            "enhanced"      : silhouette["enhanced"],
            "canny_edges"   : silhouette["canny_edges"],
            "boundary_zone" : silhouette["boundary_zone"],
            "canny_filtered": silhouette["canny_filtered"],
            "refined_mask"  : silhouette["refined_mask"],
            "contour_image" : contour_image,
            "normalized"    : normalized,
            "fold_output"   : fold_output,
            "fold_lines"    : fold_lines,
            "shape_info"    : shape_info,
            "angle"         : angle
        }

    except Exception as e:
        print(f"  [ERROR] process_image failed: {e}")
        return None


print("✅ Full pipeline ready")
"""))

# ════════════════════════════════════════════════════════
# CELL 12 — Visualization + Save
# ════════════════════════════════════════════════════════
cells.append(md("## 🖼️ Cell 11 — Visualization & Auto Save to Drive"))
cells.append(code("""\
def visualize_result(result, class_name, filename, save=True):
    \"\"\"
    9-panel visualization (tambah panel Enhanced Gray vs v3.1 yang 8-panel).
    Otomatis save ke OUTPUT_IMAGES_PATH di Google Drive.
    \"\"\"
    if result is None:
        print("  [WARNING] Empty result, skip visualization")
        return

    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    fig.patch.set_facecolor("#1a1a2e")

    title_color = "white"
    fig.suptitle(
        f"{'='*20}  {class_name}  |  {filename}  {'='*20}",
        fontsize=15, color=title_color, fontweight="bold", y=1.01
    )

    panels = [
        (ax[0,0], result["original"],       "① Original",          None),
        (ax[0,1], result["grabcut_mask"],    "② GrabCut Mask",      "gray"),
        (ax[0,2], result["enhanced"],        "③ Enhanced Gray",     "gray"),
        (ax[0,3], result["canny_edges"],     "④ Canny Edges",       "gray"),
        (ax[1,0], result["boundary_zone"],   "⑤ Boundary Zone",     "gray"),
        (ax[1,1], result["canny_filtered"],  "⑥ Canny Filtered",    "gray"),
        (ax[1,2], result["refined_mask"],    "⑦ Refined Mask",      "gray"),
        (ax[1,3], result["fold_output"],
            f"⑧ Fold Planner  |  angle={result['angle']:.1f}°", None),
    ]

    for a, img, title, cmap in panels:
        a.imshow(img, cmap=cmap)
        a.set_title(title, color=title_color, fontsize=10, pad=6)
        a.axis("off")
        for spine in a.spines.values():
            spine.set_edgecolor("#444")

    # Fold legend
    colors  = [(255/255, 165/255, 0),     (0, 220/255, 220/255), (255/255, 50/255, 50/255)]
    labels  = ["Fold Left/Center", "Fold Right/Bottom", "Fold Bottom"]
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax[1,3].legend(
        handles=patches, loc="lower left",
        fontsize=8, framealpha=0.5,
        facecolor="#222", labelcolor="white"
    )

    plt.tight_layout(pad=1.5)

    if save:
        stem      = os.path.splitext(filename)[0]
        save_path = os.path.join(OUTPUT_IMAGES_PATH, f"{class_name}_{stem}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  💾 Saved → {save_path}")

    plt.show()
    plt.close(fig)


print("✅ Visualization & save function ready")
"""))

# ════════════════════════════════════════════════════════
# CELL 13 — Visual Experiment
# ════════════════════════════════════════════════════════
cells.append(md("## 🧪 Cell 12 — Visual Experiment (Sample per Class)"))
cells.append(code("""\
for class_name in PROCESS_CLASSES:

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(class_path):
        print(f"⚠️  Missing folder: {class_path}")
        continue

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(image_files) == 0:
        print(f"⚠️  No images in {class_name}")
        continue

    sample_files = random.sample(
        image_files,
        min(CONFIG["SAMPLES_PER_CLASS"], len(image_files))
    )

    print("\\n" + "╔" + "═" * 58 + "╗")
    print(f"║  CLASS: {class_name:<49}║")
    print("╚" + "═" * 58 + "╝")

    for filename in sample_files:
        image_path = os.path.join(class_path, filename)
        print(f"\\n▶ Processing: {filename}")

        result = process_image(image_path, class_name)
        if result is None:
            print("  ⚠️  Skipped")
            continue

        visualize_result(result, class_name, filename, save=True)

        print(f"  Angle        : {result['angle']:.2f}°")
        print(f"  Aspect Ratio : {result['shape_info']['aspect_ratio']:.3f}")
        print(f"  Area         : {result['shape_info']['area']:.0f} px²")
        print(f"  Solidity     : {result['shape_info']['solidity']:.3f}")
        print(f"  Fold Steps   : {len(result['fold_lines'])}")
"""))

# ════════════════════════════════════════════════════════
# CELL 14 — Batch Processing
# ════════════════════════════════════════════════════════
cells.append(md("## 🏭 Cell 13 — Batch Processing + Metadata CSV"))
cells.append(code("""\
def batch_process_dataset(save_images=False):
    \"\"\"
    Proses seluruh dataset.
    save_images=True → simpan fold output tiap gambar ke Drive
                        (lebih lambat, butuh storage lebih besar)
    \"\"\"
    metadata    = []
    total_ok    = 0
    total_skip  = 0
    total_error = 0

    for class_name in PROCESS_CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            continue

        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"\\n▶ Processing {class_name} ({len(image_files)} images)")

        for filename in tqdm(image_files, desc=class_name):
            try:
                image_path = os.path.join(class_path, filename)
                result     = process_image(image_path, class_name)

                if result is None:
                    total_skip += 1
                    continue

                metadata.append({
                    "filename"    : filename,
                    "class_name"  : class_name,
                    "angle"       : round(result["angle"], 4),
                    "width"       : result["shape_info"]["width"],
                    "height"      : result["shape_info"]["height"],
                    "aspect_ratio": round(result["shape_info"]["aspect_ratio"], 4),
                    "solidity"    : round(result["shape_info"]["solidity"], 4),
                    "fold_steps"  : len(result["fold_lines"])
                })

                if save_images:
                    stem      = os.path.splitext(filename)[0]
                    save_path = os.path.join(OUTPUT_IMAGES_PATH, f"{class_name}_{stem}.png")
                    fold_bgr  = cv2.cvtColor(result["fold_output"], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, fold_bgr)

                total_ok += 1

            except Exception as e:
                total_error += 1
                print(f"  [ERROR] {filename}: {e}")

    # ── Save metadata CSV ──────────────────────────────
    metadata_df = pd.DataFrame(metadata)
    csv_path    = os.path.join(OUTPUT_PATH, "metadata.csv")
    metadata_df.to_csv(csv_path, index=False)
    print(f"\\n💾 Metadata saved → {csv_path}")

    # ── Summary ────────────────────────────────────────
    print("\\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if len(metadata_df) > 0:
        print(metadata_df.groupby("class_name")[["fold_steps", "solidity", "aspect_ratio"]].describe().round(3))
    print(f"\\n✅ OK    : {total_ok}")
    print(f"⚠️  Skip  : {total_skip}")
    print(f"❌ Error : {total_error}")

    return metadata_df


# Jalankan batch (save_images=True kalau mau simpan semua fold output)
metadata_df = batch_process_dataset(save_images=False)
"""))

# ════════════════════════════════════════════════════════
# CELL 15 — Export Model
# ════════════════════════════════════════════════════════
cells.append(md("## 🤖 Cell 14 — Export Model untuk Web Deploy"))
cells.append(code("""\
def export_model():
    \"\"\"
    Export pipeline ke 3 format:
    1. fold_model.json    → ringan, bisa dibaca JS/backend apapun
    2. fold_pipeline.pkl  → load langsung di Flask/FastAPI
    3. requirements.txt   → dependency list untuk server
    \"\"\"

    # 1. JSON (config + rules)
    model_data = {
        "version"        : "v3.2",
        "config"         : CONFIG,
        "fold_rules"     : FOLD_RULES,
        "baju_classes"   : BAJU_CLASSES,
        "celana_classes" : CELANA_CLASSES,
        "process_classes": PROCESS_CLASSES,
        "description"    : "Geometry-Aware Clothing Fold Planner"
    }
    json_path = os.path.join(OUTPUT_MODEL_PATH, "fold_model.json")
    with open(json_path, "w") as f:
        json.dump(model_data, f, indent=2)
    print(f"💾 JSON model       → {json_path}")

    # 2. Pickle (full pipeline functions)
    pipeline = {
        "config"                     : CONFIG,
        "fold_rules"                 : FOLD_RULES,
        "baju_classes"               : BAJU_CLASSES,
        "celana_classes"             : CELANA_CLASSES,
        "process_image"              : process_image,
        "load_image"                 : load_image,
        "validate_image"             : validate_image,
        "extract_clothing_silhouette": extract_clothing_silhouette,
        "extract_largest_contour"    : extract_largest_contour,
        "analyze_shape"              : analyze_shape,
        "normalize_orientation"      : normalize_orientation,
        "generate_folding_guideline" : generate_folding_guideline,
    }
    pkl_path = os.path.join(OUTPUT_MODEL_PATH, "fold_pipeline.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"💾 Pickle pipeline  → {pkl_path}")

    # 3. Requirements
    requirements = \"\"\"# Folding Planner v3.2 — Web Deploy Requirements
opencv-python-headless==4.9.0.80
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
scikit-image==0.22.0
tqdm==4.66.4
flask==3.0.3
Pillow==10.3.0
gunicorn==22.0.0
\"\"\"
    req_path = os.path.join(OUTPUT_MODEL_PATH, "requirements.txt")
    with open(req_path, "w") as f:
        f.write(requirements)
    print(f"💾 requirements.txt → {req_path}")

    print("\\n✅ Model export selesai!")
    print(f"📁 Folder: {OUTPUT_MODEL_PATH}")
    print("\\n── File list ──────────────────────────────")
    for f in os.listdir(OUTPUT_MODEL_PATH):
        size = os.path.getsize(os.path.join(OUTPUT_MODEL_PATH, f))
        print(f"   {f:<30} {size/1024:.1f} KB")


export_model()
"""))

# ════════════════════════════════════════════════════════
# CELL 16 — Flask API skeleton
# ════════════════════════════════════════════════════════
cells.append(md("## 🌐 Cell 15 — Generate Flask API (app.py) untuk Web Deploy"))
cells.append(code("""\
flask_code = '''#!/usr/bin/env python3
\"\"\"
Folding Planner API — v3.2
Run: gunicorn app:app --bind 0.0.0.0:5000
\"\"\"

import os, io, pickle, base64, json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import cv2

app = Flask(__name__)

# ── Load pipeline ────────────────────────────────────────
with open("fold_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

process_image              = pipeline["process_image"]
load_image                 = pipeline["load_image"]
generate_folding_guideline = pipeline["generate_folding_guideline"]
CONFIG                     = pipeline["config"]
FOLD_RULES                 = pipeline["fold_rules"]

ALLOWED_CLASSES = (
    pipeline["baju_classes"] +
    pipeline["celana_classes"]
)


def decode_image(b64_string):
    \"\"\"Decode base64 image string → numpy RGB array.\"\"\"
    img_bytes = base64.b64decode(b64_string)
    img_arr   = np.frombuffer(img_bytes, np.uint8)
    img_bgr   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def encode_image(image_rgb):
    \"\"\"Encode numpy RGB array → base64 string.\"\"\"
    img_bgr   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "v3.2"})


@app.route("/predict", methods=["POST"])
def predict():
    \"\"\"
    POST body (JSON):
    {
        "image"     : "<base64 string>",
        "class_name": "baju_lengan_panjang"
    }

    Response:
    {
        "success"     : true,
        "fold_output" : "<base64 png>",
        "fold_lines"  : [...],
        "shape_info"  : {...},
        "angle"       : float
    }
    \"\"\"
    try:
        data       = request.get_json()
        class_name = data.get("class_name", "")
        b64_image  = data.get("image", "")

        if class_name not in ALLOWED_CLASSES:
            return jsonify({
                "success": False,
                "error"  : f"class_name must be one of {ALLOWED_CLASSES}"
            }), 400

        if not b64_image:
            return jsonify({"success": False, "error": "image is required"}), 400

        # Save temp file (process_image expects a path)
        img_rgb   = decode_image(b64_image)
        temp_path = "/tmp/input.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        result = process_image(temp_path, class_name)

        if result is None:
            return jsonify({"success": False, "error": "Processing failed or image skipped"}), 422

        # Serialize fold_lines (tuples → lists for JSON)
        fold_lines_json = [
            {
                "from" : list(fl["from"]),
                "to"   : list(fl["to"]),
                "color": list(fl["color"]),
                "label": fl.get("label", "")
            }
            for fl in result["fold_lines"]
        ]

        return jsonify({
            "success"    : True,
            "fold_output": encode_image(result["fold_output"]),
            "fold_lines" : fold_lines_json,
            "shape_info" : {
                k: (list(v) if isinstance(v, tuple) else float(v))
                for k, v in result["shape_info"].items()
                if k != "bbox"
            },
            "angle": round(float(result["angle"]), 4)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
'''

api_path = os.path.join(OUTPUT_MODEL_PATH, "app.py")
with open(api_path, "w") as f:
    f.write(flask_code)

print(f"💾 Flask API saved → {api_path}")
print("\\n── Cara deploy ────────────────────────────────────────")
print("1. Copy folder model/ ke server kamu")
print("2. pip install -r requirements.txt")
print("3. gunicorn app:app --bind 0.0.0.0:5000")
print("\\n── Contoh request ─────────────────────────────────────")
print(\"\"\"
import requests, base64

with open("baju.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:5000/predict", json={
    "image"     : b64,
    "class_name": "baju_lengan_panjang"
})

data = resp.json()
# data["fold_output"] → base64 PNG hasil fold
\"\"\")
"""))

# ════════════════════════════════════════════════════════
# CELL 17 — Summary
# ════════════════════════════════════════════════════════
cells.append(md("""\
## ✅ Cell 16 — Final Summary

### Yang sudah ditingkatkan di v3.2 vs v3.1

| Komponen | v3.1 | v3.2 |
|---|---|---|
| GrabCut iterations | 5 | 10 + pass 2 refinement |
| Preprocessing Canny | Bilateral + CLAHE | **Denoise + Bilateral + CLAHE** |
| Canny thresholds | Median-based | **Otsu-guided** (lebih robust) |
| CLAHE clip | 2.0 | **3.0** (kontras lebih tajam) |
| Morph close kernel | 7×7 | **9×9** |
| Post-processing | — | **remove_small_objects + remove_small_holes** |
| Output gambar | ❌ tidak tersimpan | ✅ auto-save ke Drive |
| Model export | ❌ | ✅ JSON + Pickle + Flask API |

### Struktur output Google Drive
```
MyDrive/folding_output/
├── images/              ← visualisasi per gambar
├── model/
│   ├── fold_model.json  ← config + rules
│   ├── fold_pipeline.pkl← full pipeline
│   ├── requirements.txt
│   └── app.py           ← Flask API siap deploy
└── metadata.csv         ← metrics seluruh dataset
```
"""))

# ════════════════════════════════════════════════════════
# Assemble & write
# ════════════════════════════════════════════════════════
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language"    : "python",
        "name"        : "python3"
    },
    "language_info": {
        "name"   : "python",
        "version": "3.10.0"
    },
    "colab": {
        "provenance": []
    }
}

# Detect environment and output directory
try:
    import google.colab  # noqa: F401
    # Prefer Google Drive if already mounted, fallback to /content/
    drive_dir = "/content/drive/MyDrive"
    output_dir = drive_dir if os.path.exists(drive_dir) else "/content"
except ImportError:
    output_dir = os.path.dirname(os.path.abspath(__file__))

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "folding_planner_v3_2.ipynb")

with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Notebook written → {output_path}")
