"""StayTuned Avatar Generator - Flask app for creating branded profile images."""
import os
import threading

# Hide GPU from CUDA - prevents onnxruntime GPU discovery warning on Render (no GPU)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

try:
    import mediapipe as mp
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False

# Lazy-load rembg on first upload - keeps startup fast so Render detects the port quickly
REMBG_AVAILABLE = None  # Set on first use
_rembg_remove = None
_rembg_session = None  # Reuse session for faster processing


def _get_rembg():
    """Get rembg remove function with u2netp (smallest/fastest model, ~4MB)."""
    global REMBG_AVAILABLE, _rembg_remove, _rembg_session
    if _rembg_remove is None:
        try:
            from rembg import remove, new_session
            _rembg_remove = remove
            _rembg_session = new_session("u2netp")  # Smallest model, ~4MB, fastest
            REMBG_AVAILABLE = True
        except Exception:
            REMBG_AVAILABLE = False
    return _rembg_remove, _rembg_session

BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
STATIC_FOLDER = Path(__file__).resolve().parent / "static"
OUTPUT_SIZE = 1024
PROFILE_SCALE = 0.85  # Larger profile - fills frame like example, less empty space at bottom
EDGE_FEATHER_RADIUS = 6  # Softer, more natural blend (Slack profile style)
REMBG_MAX_SIZE = 384  # Process at 384px for speed - 4x faster than 768
# Vertical offset: slight downward shift so subject isn't floating, bottom feels right
PROFILE_Y_OFFSET = 20  # Pixels down from center
BG_COLOR = (25, 32, 72)  # StayTuned blue fallback when background.png is missing

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)


def ensure_assets():
    """Create placeholder background if it doesn't exist (logo is embedded in background)."""
    bg_path = STATIC_FOLDER / "background.png"
    STATIC_FOLDER.mkdir(exist_ok=True)

    if not bg_path.exists():
        # Fallback: create branded placeholder (StayTuned blue theme)
        img = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), BG_COLOR)
        img.save(bg_path)


# Initialize assets on startup
ensure_assets()


def _warmup_rembg():
    """Pre-load rembg in background so first upload is fast."""
    import time
    time.sleep(5)  # Let app bind to port first
    _get_rembg()


# Start rembg warmup in background thread (no delay for port binding)
threading.Thread(target=_warmup_rembg, daemon=True).start()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def center_crop_to_square(img):
    """Crop image to square from center."""
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    return img.crop((left, top, left + size, top + size))


def face_crop_to_square(img):
    """Crop image to square centered on detected face. Falls back to center crop if no face."""
    if not FACE_DETECTION_AVAILABLE:
        return center_crop_to_square(img)

    width, height = img.size
    img_np = np.array(img.convert("RGB"))

    try:
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(img_np)

        if not results.detections:
            return center_crop_to_square(img)

        # Use largest detected face
        best = max(
            results.detections,
            key=lambda d: (d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height),
        )
        bbox = best.location_data.relative_bounding_box

        # Convert to pixel coords (relative 0-1)
        cx = (bbox.xmin + bbox.width / 2) * width
        cy = (bbox.ymin + bbox.height / 2) * height

        # Crop square centered on face - tight framing for face-only shot (~1.5x face)
        face_h = bbox.height * height
        max_size = min(width, height)
        size = min(max_size, max(int(face_h * 1.5), 100))

        left = int(cx - size / 2)
        top = int(cy - size / 2)

        # Clamp to image bounds
        left = max(0, min(left, width - size))
        top = max(0, min(top, height - size))
        right = left + size
        bottom = top + size

        return img.crop((left, top, right, bottom))
    except Exception:
        return center_crop_to_square(img)


def fit_background_to_canvas(bg_img, target_size):
    """Resize background to cover target_size, center crop for balanced composition."""
    w, h = bg_img.size
    if w == target_size and h == target_size:
        return bg_img
    scale = max(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = bg_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    return resized.crop((left, top, left + target_size, top + target_size))


def apply_vignette(img, strength=0.4):
    """Soft vignette: darkens edges to add depth and focus to center."""
    if strength <= 0:
        return img
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cx, cy = w / 2, h / 2
    # Radial falloff from center (1 at center, 0 at corners)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    mask = 1 - strength * (dist / max_dist) ** 2
    mask = np.clip(mask, 0, 1)
    if arr.shape[-1] == 4:
        arr[:, :, :3] = arr[:, :, :3] * mask[..., np.newaxis]
    else:
        arr = arr * mask[..., np.newaxis]
    return Image.fromarray(arr.astype(np.uint8))


def enhance_background(img):
    """Subtle enhancements for a polished, premium look."""
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.05)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    return img


def process_avatar(source_path, output_path):
    """Process uploaded image: crop, resize, place on clean solid background."""
    profile = Image.open(source_path)
    profile = ImageOps.exif_transpose(profile)  # Fix orientation from phone/camera EXIF
    profile = profile.convert("RGBA")

    # Crop to square centered on face (or center if no face detected)
    profile = face_crop_to_square(profile)

    # Remove background at 384px for speed - then resize to final (lazy-loaded, u2netp)
    profile_size = int(OUTPUT_SIZE * PROFILE_SCALE)
    rembg_fn, rembg_session = _get_rembg()
    if rembg_fn and rembg_session:
        try:
            small = profile.resize(
                (REMBG_MAX_SIZE, REMBG_MAX_SIZE),
                Image.Resampling.LANCZOS,
            )
            small_nobg = rembg_fn(small, session=rembg_session)
            profile = small_nobg.convert("RGBA").resize(
                (profile_size, profile_size),
                Image.Resampling.LANCZOS,
            )
        except Exception:
            profile = profile.resize((profile_size, profile_size), Image.Resampling.LANCZOS)
    else:
        profile = profile.resize((profile_size, profile_size), Image.Resampling.LANCZOS)

    # Soft, natural edge blend (like the example - well-defined but not harsh)
    if profile.mode == "RGBA" and EDGE_FEATHER_RADIUS > 0:
        r, g, b, a = profile.split()
        a_feathered = a.filter(ImageFilter.GaussianBlur(radius=EDGE_FEATHER_RADIUS))
        profile = Image.merge("RGBA", (r, g, b, a_feathered))

    # Load StayTuned background, fit to canvas, and polish it
    bg_path = STATIC_FOLDER / "background.png"
    background = Image.open(bg_path).convert("RGBA")
    background = fit_background_to_canvas(background, OUTPUT_SIZE)
    background = enhance_background(background.convert("RGB")).convert("RGBA")
    background = apply_vignette(background, strength=0.35)

    # Paste profile - centered horizontally, slightly lower to fix bottom space
    composite = background.copy()
    x = (OUTPUT_SIZE - profile_size) // 2
    y = (OUTPUT_SIZE - profile_size) // 2 + PROFILE_Y_OFFSET
    if profile.mode == "RGBA":
        composite.paste(profile, (x, y), profile.split()[3])
    else:
        composite.paste(profile, (x, y))

    # Save as PNG (no quality param - PNG is lossless)
    composite_rgb = composite.convert("RGB")
    composite_rgb.save(output_path, "PNG")
    return output_path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return "ok", 200


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only JPG and PNG files are allowed"}), 400

    upload_path = None
    output_path = None
    file_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[1].lower()
    upload_path = UPLOAD_FOLDER / f"{file_id}_input.{ext}"
    output_path = UPLOAD_FOLDER / f"{file_id}_output.png"

    try:
        file.save(upload_path)

        # Validate image
        try:
            with Image.open(upload_path) as img:
                img.verify()
        except Exception:
            upload_path.unlink(missing_ok=True)
            return jsonify({"error": "Invalid image file"}), 400

        # Process
        process_avatar(upload_path, output_path)

        # Clean up input
        upload_path.unlink(missing_ok=True)

        return jsonify({
            "success": True,
            "download_url": f"/download/{file_id}",
            "preview_url": f"/download/{file_id}",
        })
    except Exception as e:
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        if output_path and output_path.exists():
            output_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


@app.route("/download/<file_id>")
def download(file_id):
    path = UPLOAD_FOLDER / f"{file_id}_output.png"
    if not path.exists():
        return "File not found or expired", 404
    return send_file(
        path,
        mimetype="image/png",
        as_attachment=True,
        download_name="staytuned-avatar.png",
    )


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 5MB."}), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)
