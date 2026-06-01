#!/usr/bin/env python3
"""
Folding Planner API — v3.2
Run: gunicorn app:app --bind 0.0.0.0:5000
"""

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
    """Decode base64 image string → numpy RGB array."""
    img_bytes = base64.b64decode(b64_string)
    img_arr   = np.frombuffer(img_bytes, np.uint8)
    img_bgr   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def encode_image(image_rgb):
    """Encode numpy RGB array → base64 string."""
    img_bgr   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "v3.2"})


@app.route("/predict", methods=["POST"])
def predict():
    """
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
    """
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
