"""
Flask backend for the Air Quality Image Classification dashboard.

Serves the premium 3D UI and a small JSON API that reads REAL generated result
files (results/…). When a result file does not yet exist the API returns a
clear "not available yet" status so the UI never shows fabricated metrics.

SECURITY NOTE: this is a LOCAL demo server with NO authentication. It binds to
127.0.0.1 by default. Do not expose it to a public network without adding
authentication, file-type/size limits (already applied), and rate limiting.
"""
from __future__ import annotations

import io
import sys
import uuid
from pathlib import Path

from flask import (Flask, jsonify, render_template, request,
                   send_from_directory, abort)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, utils  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload cap
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UPLOAD_DIR = config.RESULTS_DIR / "_uploads"


def _exists(p: Path) -> bool:
    return Path(p).exists()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    meta = utils.load_json(config.FINAL_MODEL_META_JSON)
    cm = utils.load_json(config.CLASS_MAP_JSON)
    metrics = utils.load_json(config.FINAL_METRICS_JSON)
    return jsonify({
        "model_ready": bool(meta) and _exists(Path(meta.get("model_path", "")))
        if meta else False,
        "evaluated": bool(metrics),
        "classes": (cm or {}).get("classes"),
        "selected_model": (meta or {}).get("selected_model"),
    })


@app.route("/api/dataset")
def dataset_info():
    audit = utils.load_json(config.DATASET_AUDIT_JSON)
    if not audit:
        return jsonify({"available": False, "message": "Run dataset audit."})
    return jsonify({"available": True, **audit})


@app.route("/api/model")
def model_info():
    meta = utils.load_json(config.FINAL_MODEL_META_JSON)
    metrics = utils.load_json(config.FINAL_METRICS_JSON)
    if not meta:
        return jsonify({"available": False,
                        "message": "No model selected yet."})
    return jsonify({"available": True, "meta": meta, "metrics": metrics})


@app.route("/api/experiments")
def experiments():
    import pandas as pd
    if not _exists(config.EXPERIMENTS_CSV):
        return jsonify({"available": False, "rows": []})
    df = pd.read_csv(config.EXPERIMENTS_CSV)
    sel = utils.load_json(config.MODEL_SELECTION_JSON) or {}
    return jsonify({"available": True,
                    "selected": sel.get("selected_model"),
                    "rows": df.to_dict("records")})


@app.route("/api/evaluation")
def evaluation():
    metrics = utils.load_json(config.FINAL_METRICS_JSON)
    errors = utils.load_json(config.ERROR_DIR / "error_analysis.json")
    return jsonify({
        "available": bool(metrics),
        "metrics": metrics,
        "errors": errors,
        "plots": {
            "confusion": "/results/evaluation/confusion_matrix.png",
            "confusion_norm": "/results/evaluation/confusion_matrix_normalized.png",
            "errors_grid": "/results/error_analysis/representative_errors.png",
        },
    })


@app.route("/results/<path:relpath>")
def results_file(relpath: str):
    # Safe static serving of generated artifacts (blocks path traversal).
    base = config.RESULTS_DIR.resolve()
    target = (base / relpath).resolve()
    if base not in target.parents and target != base:
        abort(403)
    if not target.exists():
        abort(404)
    return send_from_directory(str(target.parent), target.name)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    file = request.files["image"]
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported type '{ext}'."}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex[:12]
    saved = UPLOAD_DIR / f"{uid}{ext}"
    file.save(str(saved))

    try:
        from src import predict as predictor
        gradcam_out = config.EXPLAIN_DIR / f"pred_{uid}.png"
        result = predictor.predict_image(str(saved), with_gradcam=True,
                                          gradcam_out=str(gradcam_out))
        if gradcam_out.exists():
            result["gradcam_url"] = f"/results/explainability/{gradcam_out.name}"
        result["upload_url"] = f"/results/_uploads/{saved.name}"
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc),
                        "hint": "Model not trained yet or TensorFlow missing."}), 503


if __name__ == "__main__":
    config.ensure_dirs()
    print("Air Quality dashboard -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
