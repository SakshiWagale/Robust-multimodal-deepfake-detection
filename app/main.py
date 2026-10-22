import os
import shutil
import uuid
import hashlib
import subprocess
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.preprocessing import detect_media_type, check_ffmpeg_available
from app.video_detector import VideoDetector
from app.audio_detector import AudioDetector
from app.fusion import fuse_predictions, DEFAULT_VIDEO_WEIGHT, DEFAULT_AUDIO_WEIGHT, DEFAULT_THRESHOLD
from app.pdf_report import generate_forensic_pdf

# Initialize detectors
video_detector = VideoDetector(model_path="models/video_classifier.keras")
audio_detector = AudioDetector(model_path="models/audio_classifier.keras")

video_model_loaded = False
audio_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_model_loaded, audio_model_loaded
    print("\n" + "=" * 60)
    print("      MULTIMODAL DEEPFAKE DETECTION SYSTEM STARTUP       ")
    print("=" * 60)

    # Check FFmpeg
    if check_ffmpeg_available():
        print("[SYSTEM] FFmpeg detected in system PATH.")
    else:
        print("[WARNING] FFmpeg NOT detected in system PATH. Audio extraction from videos will fail.")

    # Check and Load Video Model
    try:
        video_detector.load_model()
        video_model_loaded = True
    except FileNotFoundError as e:
        print(f"[MODEL ERROR] {e}", flush=True)
        video_model_loaded = False

    # Check and Load Audio Model
    try:
        audio_detector.load_model()
        audio_model_loaded = True
    except FileNotFoundError as e:
        print(f"[MODEL ERROR] {e}", flush=True)
        audio_model_loaded = False

    video_prep = (
        "Embedded Rescaling Layer (Inputs scaled [0, 255] internally)"
        if (video_model_loaded and video_detector.has_rescaling)
        else "Explicit Normalization (Inputs divided by 255.0 to [0.0, 1.0])"
    )
    audio_prep = "128-band Log-Mel Spectrogram (16000 Hz, 3.0s normalized to [0.0, 1.0])"

    print("\n" + "=" * 60, flush=True)
    print("MODEL LOADED", flush=True)
    print(f"VIDEO MODEL: {video_detector.model_path if video_model_loaded else 'MISSING'}", flush=True)
    print(f"AUDIO MODEL: {audio_detector.model_path if audio_model_loaded else 'MISSING'}", flush=True)
    print(f"VIDEO INPUT SHAPE: {video_detector.input_shape if video_model_loaded else 'N/A'}", flush=True)
    print(f"AUDIO INPUT SHAPE: {audio_detector.input_shape if audio_model_loaded else 'N/A'}", flush=True)
    print("CLASS MAPPING: fake = 0, real = 1 (Output sigmoid > threshold => REAL)", flush=True)
    print(f"VIDEO PREPROCESSING: {video_prep}", flush=True)
    print(f"AUDIO PREPROCESSING: {audio_prep}", flush=True)
    print("=" * 60 + "\n", flush=True)

    yield
    print("[SYSTEM] Shutting down Multimodal Deepfake Detection server.")


app = FastAPI(title="Multimodal Deepfake Detection", lifespan=lifespan)

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
RESULTS_DIR = os.path.join(STATIC_DIR, "results")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "video_model_loaded": video_model_loaded,
            "audio_model_loaded": audio_model_loaded,
            "ffmpeg_available": check_ffmpeg_available(),
        }
    )


@app.get("/upload", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "video_model_loaded": video_model_loaded,
            "audio_model_loaded": audio_model_loaded,
            "default_video_weight": DEFAULT_VIDEO_WEIGHT,
            "default_audio_weight": DEFAULT_AUDIO_WEIGHT,
            "default_threshold": DEFAULT_THRESHOLD,
        }
    )


@app.post("/predict")
async def predict_api(
    request: Request,
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(DEFAULT_THRESHOLD),
    video_weight: Optional[float] = Form(DEFAULT_VIDEO_WEIGHT),
    audio_weight: Optional[float] = Form(DEFAULT_AUDIO_WEIGHT),
):
    """
    Main prediction endpoint.
    Accepts video, audio, or image upload via multipart/form-data.
    Executes actual ML inference using trained models.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    original_filename = file.filename
    ext = os.path.splitext(original_filename)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
    saved_file_path = os.path.join(UPLOADS_DIR, unique_filename)

    # Save uploaded file
    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Detect media type from extension and content
    media_type = detect_media_type(saved_file_path)
    if media_type == "unknown":
        if os.path.exists(saved_file_path):
            os.remove(saved_file_path)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: Video (MP4, AVI, MOV, MKV, WEBM), "
                   f"Audio (WAV, MP3, M4A, AAC, FLAC), Image (JPG, PNG, WEBP)."
        )

    video_score = None
    audio_score = None
    errors = []

    # 1. Processing for Video (or Image)
    if media_type in ("video", "image"):
        if not os.path.exists(video_detector.model_path):
            raise HTTPException(
                status_code=503,
                detail=f"Video model not found at '{video_detector.model_path}'. "
                       f"Inference aborted. Please train the video model using 'python training/train_video.py'."
            )
        try:
            video_result = video_detector.predict(saved_file_path)
            video_score = video_result["score"]
        except Exception as e:
            errors.append(f"Video inference error: {str(e)}")

        # For video, also attempt audio extraction and classification
        if media_type == "video":
            if os.path.exists(audio_detector.model_path):
                try:
                    audio_result = audio_detector.predict(saved_file_path)
                    if audio_result.get("status") == "completed":
                        audio_score = audio_result["score"]
                    else:
                        # Audio track not available or failed extraction - do not invent score
                        errors.append(audio_result.get("message", "Audio track unavailable."))
                except Exception as e:
                    errors.append(f"Audio processing error: {str(e)}")
            else:
                errors.append("Audio model not trained; skipped audio stream analysis.")

    # 2. Processing for Audio only
    elif media_type == "audio":
        if not os.path.exists(audio_detector.model_path):
            raise HTTPException(
                status_code=503,
                detail=f"Audio model not found at '{audio_detector.model_path}'. "
                       f"Inference aborted. Please train the audio model using 'python training/train_audio.py'."
            )
        try:
            audio_result = audio_detector.predict(saved_file_path)
            if audio_result.get("status") == "completed":
                audio_score = audio_result["score"]
            else:
                raise ValueError(audio_result.get("message", "Audio inference failed."))
        except Exception as e:
            errors.append(f"Audio inference error: {str(e)}")

    # Check if any modality succeeded
    if video_score is None and audio_score is None:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed. Reasons: {'; '.join(errors)}"
        )

    # Perform Multimodal Fusion
    try:
        fusion_result = fuse_predictions(
            video_score=video_score,
            audio_score=audio_score,
            video_weight=video_weight if video_weight is not None else DEFAULT_VIDEO_WEIGHT,
            audio_weight=audio_weight if audio_weight is not None else DEFAULT_AUDIO_WEIGHT,
            threshold=threshold if threshold is not None else DEFAULT_THRESHOLD
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion calculation error: {str(e)}")

    # Forensic digital evidence calculations
    file_sha256 = "N/A"
    file_size_str = "N/A"
    try:
        with open(saved_file_path, "rb") as f:
            file_bytes = f.read()
            file_sha256 = hashlib.sha256(file_bytes).hexdigest().upper()
            size_kb = len(file_bytes) / 1024.0
            if size_kb > 1024:
                file_size_str = f"{size_kb / 1024.0:.2f} MB"
            else:
                file_size_str = f"{size_kb:.1f} KB"
    except Exception:
        pass

    case_id = f"VRT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    timestamp_str = datetime.now().strftime("%d %b %Y, %H:%M:%S UTC")

    # Generate standalone amplified MP3 audio for clear web playback
    audio_url = None
    if media_type == "video" and check_ffmpeg_available():
        try:
            audio_stream_filename = f"audio_{os.path.splitext(unique_filename)[0]}.mp3"
            audio_stream_path = os.path.join(UPLOADS_DIR, audio_stream_filename)
            subprocess.run([
                "ffmpeg", "-y", "-i", saved_file_path, "-vn",
                "-af", "volume=2.5", "-ar", "44100", "-ac", "2", "-b:a", "192k",
                audio_stream_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if os.path.exists(audio_stream_path) and os.path.getsize(audio_stream_path) > 500:
                audio_url = f"/static/uploads/{audio_stream_filename}"
        except Exception:
            pass
    elif media_type == "audio":
        audio_url = f"/static/uploads/{unique_filename}"

    response_data = {
        "request": request,
        "filename": original_filename,
        "media_type": media_type,
        "file_size": file_size_str,
        "sha256_hash": file_sha256,
        "case_id": case_id,
        "timestamp": timestamp_str,
        "video_score": fusion_result["video_score"],
        "audio_score": fusion_result["audio_score"],
        "final_score": fusion_result["final_score"],
        "verdict": fusion_result["verdict"],
        "confidence": fusion_result["confidence"],
        "threshold": fusion_result["threshold"],
        "analysis_type": fusion_result["analysis_type"],
        "status": "completed",
        "file_url": f"/static/uploads/{unique_filename}",
        "audio_url": audio_url,
        "notes": errors if errors else None
    }

    # ── Save result to JSON file so PDF download uses exact correct data ──────
    try:
        result_json_path = os.path.join(RESULTS_DIR, f"{case_id}.json")
        json_payload = {k: v for k, v in response_data.items() if k != "request"}
        with open(result_json_path, "w", encoding="utf-8") as jf:
            json.dump(json_payload, jf, ensure_ascii=False, indent=2)
    except Exception as je:
        print(f"[WARN] Could not save result JSON: {je}")

    # If an API client specifically requests pure JSON, return JSONResponse
    accept_header = request.headers.get("accept", "")
    if accept_header == "application/json":
        json_data = {k: v for k, v in response_data.items() if k != "request"}
        return JSONResponse(content=json_data)

    # By default (all browsers, HTML form submissions), render the result.html dashboard
    return templates.TemplateResponse("result.html", response_data)


@app.get("/result", response_class=HTMLResponse)
async def get_result_page(request: Request):
    """Renders the result page dashboard."""
    return templates.TemplateResponse("result.html", {"request": request})


@app.get("/download_report_pdf")
async def download_report_pdf(
    case_id: str = "VRT-2026-REPORT",
    # fallback params (used only if JSON file not found)
    filename: str = "media_evidence",
    media_type: str = "video",
    file_size: str = "N/A",
    sha256_hash: str = "N/A",
    timestamp: str = "",
    video_score: Optional[float] = None,
    audio_score: Optional[float] = None,
    final_score: float = 0.5,
    verdict: str = "REAL",
    confidence: float = 50.0,
    threshold: float = 0.55
):
    """
    Generates and streams the official PDF Forensic Certificate.
    Reads exact result data from saved JSON file (using case_id).
    Falls back to URL query params if JSON not found.
    """
    # ── Try loading from saved JSON (most reliable) ───────────────────────────
    result_json_path = os.path.join(RESULTS_DIR, f"{case_id}.json")
    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, "r", encoding="utf-8") as jf:
                saved = json.load(jf)
            filename      = saved.get("filename", filename)
            media_type    = saved.get("media_type", media_type)
            file_size     = saved.get("file_size", file_size)
            sha256_hash   = saved.get("sha256_hash", sha256_hash)
            timestamp     = saved.get("timestamp", timestamp)
            video_score   = saved.get("video_score", video_score)
            audio_score   = saved.get("audio_score", audio_score)
            final_score   = saved.get("final_score", final_score)
            verdict       = saved.get("verdict", verdict)
            confidence    = saved.get("confidence", confidence)
            threshold     = saved.get("threshold", threshold)
        except Exception as e:
            print(f"[WARN] Could not load result JSON for {case_id}: {e}")
    else:
        print(f"[WARN] No saved result JSON found for case_id={case_id}. Using URL params.")

    pdf_buffer = generate_forensic_pdf(
        filename=filename,
        media_type=media_type,
        file_size=file_size,
        sha256_hash=sha256_hash,
        case_id=case_id,
        timestamp=timestamp,
        video_score=video_score,
        audio_score=audio_score,
        final_score=final_score,
        verdict=verdict,
        confidence=confidence,
        threshold=threshold
    )
    headers = {
        "Content-Disposition": f'attachment; filename="VERITAS_Forensic_Report_{case_id}.pdf"'
    }
    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers=headers)
