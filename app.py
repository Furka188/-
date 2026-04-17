from __future__ import annotations

import math
import uuid
import zipfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12MB per request
mp_face_mesh = mp.solutions.face_mesh
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class AnalysisResult:
    metrics: Dict[str, float]
    quality: Dict[str, float]
    landmarks: List[Tuple[int, float, float]]
    overlay_path: Path
    report_path: Path


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.dist(a, b))


def _to_px(landmark, width: int, height: int) -> Tuple[float, float]:
    return (landmark.x * width, landmark.y * height)


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _image_quality(image_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "sharpness_laplacian": round(sharpness, 3),
    }


def _extract_metrics(points: Dict[int, Tuple[float, float]]) -> Dict[str, float]:
    left_eye_center = ((points[33][0] + points[133][0]) / 2, (points[33][1] + points[133][1]) / 2)
    right_eye_center = ((points[362][0] + points[263][0]) / 2, (points[362][1] + points[263][1]) / 2)

    ipd = _distance(left_eye_center, right_eye_center)
    jaw = _distance(points[234], points[454])
    face_height = _distance(points[10], points[152])
    nose_bridge = _distance(points[6], points[1])

    left_eye_area = _polygon_area([points[33], points[160], points[158], points[133], points[153], points[144]])
    right_eye_area = _polygon_area([points[362], points[385], points[387], points[263], points[373], points[380]])

    symmetry_pairs = [(33, 263), (133, 362), (61, 291), (234, 454), (70, 300), (105, 334)]
    mid_x = (points[234][0] + points[454][0]) / 2
    symmetry_errors = []
    for left_idx, right_idx in symmetry_pairs:
        lx, _ = points[left_idx]
        rx, _ = points[right_idx]
        symmetry_errors.append(abs(abs(mid_x - lx) - abs(rx - mid_x)))
    symmetry_score = max(0.0, 1.0 - (float(np.mean(symmetry_errors)) / (jaw + 1e-6)))

    metrics = {
        "interocular_distance_px": ipd,
        "nose_width_px": _distance(points[129], points[358]),
        "nose_bridge_length_px": nose_bridge,
        "mouth_width_px": _distance(points[61], points[291]),
        "jaw_width_px": jaw,
        "face_height_px": face_height,
        "left_eye_width_px": _distance(points[33], points[133]),
        "right_eye_width_px": _distance(points[362], points[263]),
        "left_eye_area_px2": left_eye_area,
        "right_eye_area_px2": right_eye_area,
        "eye_area_ratio_lr": left_eye_area / (right_eye_area + 1e-6),
        "face_aspect_ratio": face_height / (jaw + 1e-6),
        "mouth_to_jaw_ratio": _distance(points[61], points[291]) / (jaw + 1e-6),
        "nose_to_face_height_ratio": nose_bridge / (face_height + 1e-6),
        "symmetry_score_0_1": symmetry_score,
    }

    if ipd > 0:
        metrics["jaw_over_ipd"] = jaw / ipd
        metrics["face_height_over_ipd"] = face_height / ipd
        metrics["nose_width_over_ipd"] = metrics["nose_width_px"] / ipd
        metrics["mouth_width_over_ipd"] = metrics["mouth_width_px"] / ipd

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    metrics["eye_line_angle_deg"] = math.degrees(math.atan2(dy, dx))

    return {k: round(float(v), 4) for k, v in metrics.items()}


def _top_distinguishing_points(
    p1: List[Tuple[int, float, float]], p2: List[Tuple[int, float, float]], top_k: int = 15
) -> List[Dict[str, float]]:
    p1_map = {idx: (x, y) for idx, x, y in p1}
    p2_map = {idx: (x, y) for idx, x, y in p2}
    changes = []
    for idx in sorted(set(p1_map) & set(p2_map)):
        changes.append({"landmark": idx, "delta_px": round(_distance(p1_map[idx], p2_map[idx]), 4)})
    changes.sort(key=lambda item: item["delta_px"], reverse=True)
    return changes[:top_k]


def _validate_and_save_upload(photo, field_name: str) -> Path:
    if not photo or photo.filename == "":
        raise ValueError(f"Attach a valid file for {field_name}")

    clean_name = secure_filename(photo.filename)
    ext = Path(clean_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format for {field_name}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}_{clean_name}"
    photo.save(str(target))
    return target


def _cleanup_generated_files(max_age_seconds: int = 60 * 60 * 24) -> None:
    now = int(time.time())
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        for file_path in folder.glob("*"):
            if not file_path.is_file():
                continue
            age = now - int(file_path.stat().st_mtime)
            if age > max_age_seconds:
                file_path.unlink(missing_ok=True)


def analyze_face(image_path: Path) -> AnalysisResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Failed to read uploaded image")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        raise ValueError("No face detected in image")

    face_landmarks = result.multi_face_landmarks[0].landmark
    h, w, _ = image.shape
    points: Dict[int, Tuple[float, float]] = {}
    all_points: List[Tuple[int, float, float]] = []

    for idx, lm in enumerate(face_landmarks):
        x, y = _to_px(lm, w, h)
        points[idx] = (x, y)
        all_points.append((idx, round(x, 4), round(y, 4)))

    required = [1, 6, 10, 33, 61, 129, 133, 152, 234, 263, 291, 358, 362, 454]
    if any(i not in points for i in required):
        raise ValueError("Required landmarks unavailable for this photo")

    metrics = _extract_metrics(points)
    quality = _image_quality(image)

    overlay = image.copy()
    for idx, (x, y) in points.items():
        if idx % 5 == 0:
            cv2.circle(overlay, (int(x), int(y)), 1, (255, 255, 255), -1)

    for a, b in [(33, 133), (362, 263), (61, 291), (234, 454), (10, 152), (1, 6), (33, 263), (133, 362)]:
        pa, pb = points[a], points[b]
        cv2.line(overlay, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 255, 255), 1)

    run_id = uuid.uuid4().hex
    overlay_path = OUTPUT_DIR / f"{run_id}_overlay.png"
    report_path = OUTPUT_DIR / f"{run_id}_analysis.txt"
    cv2.imwrite(str(overlay_path), overlay)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("QWEEN FACE — BIOMETRY DOSSIER\n")
        f.write("=" * 72 + "\n")
        f.write("SECTION A — Metric Topology\n")
        for key, value in metrics.items():
            f.write(f"- {key}: {value}\n")
        f.write("\nSECTION B — Image Quality Envelope\n")
        for key, value in quality.items():
            f.write(f"- {key}: {value}\n")
        f.write("\nSECTION C — Landmark Coordinates (index: x, y)\n")
        for idx, x, y in all_points:
            f.write(f"{idx}: {x}, {y}\n")

    return AnalysisResult(metrics=metrics, quality=quality, landmarks=all_points, overlay_path=overlay_path, report_path=report_path)


def _build_project_zip() -> Path:
    archive_path = OUTPUT_DIR / "qween-face.zip"
    excluded_roots = {".git", ".venv", "outputs", "uploads", "__pycache__"}

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in BASE_DIR.rglob("*"):
            rel = path.relative_to(BASE_DIR)
            if not rel.parts or rel.parts[0] in excluded_roots:
                continue
            if path.is_file() and path.suffix != ".pyc":
                zf.write(path, rel.as_posix())
    return archive_path


def compare_analyses(first: AnalysisResult, second: AnalysisResult) -> Dict[str, object]:
    metric_diff: Dict[str, Dict[str, float]] = {}
    all_keys = set(first.metrics.keys()) | set(second.metrics.keys())
    for key in sorted(all_keys):
        v1 = first.metrics.get(key, 0.0)
        v2 = second.metrics.get(key, 0.0)
        delta = round(abs(v1 - v2), 4)
        pct = round((delta / abs(v1)) * 100, 3) if v1 else 0.0
        metric_diff[key] = {"photo_1": round(v1, 4), "photo_2": round(v2, 4), "difference": delta, "difference_percent_from_photo_1": pct}

    return {
        "metric_differences": metric_diff,
        "top_distinguishing_landmarks": _top_distinguishing_points(first.landmarks, second.landmarks),
    }


@app.get("/")
def index():
    _cleanup_generated_files()
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    try:
        input_path = _validate_and_save_upload(request.files.get("photo"), "photo")
        analysis = analyze_face(input_path)
        return jsonify(
            {
                "metrics": analysis.metrics,
                "quality": analysis.quality,
                "overlay_download": f"/download/image/{analysis.overlay_path.name}",
                "report_download": f"/download/report/{analysis.report_path.name}",
                "landmarks_count": len(analysis.landmarks),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/compare")
def compare():
    try:
        input_path_1 = _validate_and_save_upload(request.files.get("photo_1"), "photo_1")
        input_path_2 = _validate_and_save_upload(request.files.get("photo_2"), "photo_2")
        first = analyze_face(input_path_1)
        second = analyze_face(input_path_2)
        comparison = compare_analyses(first, second)
        return jsonify(
            {
                "comparison": comparison,
                "first_overlay": f"/download/image/{first.overlay_path.name}",
                "second_overlay": f"/download/image/{second.overlay_path.name}",
                "first_report": f"/download/report/{first.report_path.name}",
                "second_report": f"/download/report/{second.report_path.name}",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/download/project-zip")
def download_project_zip():
    return send_file(_build_project_zip(), as_attachment=True)


@app.get("/download/image/<name>")
def download_image(name: str):
    target = OUTPUT_DIR / name
    if not target.exists():
        return jsonify({"error": "Image not found"}), 404
    return send_file(target, as_attachment=True)


@app.get("/download/report/<name>")
def download_report(name: str):
    target = OUTPUT_DIR / name
    if not target.exists():
        return jsonify({"error": "Report not found"}), 404
    return send_file(target, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
