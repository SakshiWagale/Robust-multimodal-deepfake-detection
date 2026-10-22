import os
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
import librosa

# Supported formats
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard Audio Specifications
TARGET_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 3.0
TARGET_AUDIO_SAMPLES = int(TARGET_SAMPLE_RATE * AUDIO_DURATION_SECONDS)  # 48,000 samples
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024

# Standard Video Specifications
VIDEO_FRAME_SIZE = (224, 224)
DEFAULT_NUM_FRAMES = 16


def detect_media_type(file_path: str) -> str:
    """
    Detect whether the file is a video, audio, or image based on file extension.
    Returns: 'video', 'audio', 'image', or 'unknown'
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def has_rescaling_layer(model) -> bool:
    """
    Inspects a Keras model architecture to check if a Rescaling layer is present.
    Ensures inference preprocessing exactly matches training preprocessing.
    """
    if model is None:
        return False

    def _check_layers(layers_list):
        for layer in layers_list:
            class_name = layer.__class__.__name__
            if class_name == "Rescaling":
                return True
            # Check nested functional or sequential sub-models
            if hasattr(layer, "layers") and layer.layers:
                if _check_layers(layer.layers):
                    return True
        return False

    return _check_layers(model.layers)


def has_mobilenet_preprocessing(model) -> bool:
    """
    Detects if the model uses MobileNetV2 architecture (contains a MobileNetV2 sub-model).
    MobileNetV2 uses preprocess_input which maps [0,255] -> [-1, 1].
    In that case, we must NOT divide by 255 nor apply Rescaling.
    The raw [0,255] float32 frames are passed directly; the graph handles scaling internally.
    """
    if model is None:
        return False

    def _check(layers_list):
        for layer in layers_list:
            layer_name = getattr(layer, "name", "")
            class_name = layer.__class__.__name__
            if "mobilenetv2" in layer_name.lower() or "mobilenet" in class_name.lower():
                return True
            if hasattr(layer, "layers") and layer.layers:
                if _check(layer.layers):
                    return True
        return False

    return _check(model.layers)


def get_preprocessing_mode(model) -> str:
    """
    Returns preprocessing mode for inference:
      'rescaling'  — model has embedded Rescaling(1/255) layer; pass raw [0,255] float32
      'mobilenet'  — model uses MobileNetV2 preprocess_input embedded in graph; pass raw [0,255] float32
      'normalize'  — model has no scaling layer; divide by 255 manually
    """
    if has_rescaling_layer(model):
        return "rescaling"
    if has_mobilenet_preprocessing(model):
        return "mobilenet"
    return "normalize"


def extract_frames(media_path: str, num_frames: int = DEFAULT_NUM_FRAMES, target_size=VIDEO_FRAME_SIZE):
    """
    Extracts evenly spaced frames from a video or loads a single image.
    Converts BGR to RGB and resizes to target_size (width, height).
    Returns a numpy array of shape (N, height, width, 3) in uint8 [0, 255].
    """
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Media file not found: {media_path}")

    media_type = detect_media_type(media_path)

    if media_type == "image":
        img = cv2.imread(media_path)
        if img is None:
            raise ValueError(f"Failed to read image at {media_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size)
        return np.expand_dims(img_resized, axis=0)

    # Video processing
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {media_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames > 0:
        actual_sample_count = min(num_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, actual_sample_count, dtype=int)

        current_idx = 0
        target_idx_set = set(frame_indices)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in target_idx_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, target_size)
                frames.append(frame_resized)
            current_idx += 1
            if len(frames) >= actual_sample_count:
                break
    else:
        step = 5
        count = 0
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, target_size)
                frames.append(frame_resized)
            count += 1

    cap.release()

    if not frames:
        raise ValueError(f"No valid frames could be decoded from video: {media_path}")

    return np.array(frames, dtype=np.uint8)


def preprocess_frames(frames: np.ndarray, model=None) -> np.ndarray:
    """
    Preprocess extracted frames for model inference.

    Preprocessing modes (auto-detected from model architecture):
      'rescaling'  — model has Rescaling(1/255) embedded: pass raw [0,255] as float32
      'mobilenet'  — model uses MobileNetV2 preprocess_input embedded in graph:
                     pass raw [0,255] as float32 (graph handles [-1,1] mapping internally)
      'normalize'  — no scaling in model: divide by 255.0 to get [0.0, 1.0]
    """
    mode = get_preprocessing_mode(model)
    frames_float = frames.astype(np.float32)

    if mode == "normalize":
        frames_float = frames_float / 255.0
        print("[PREPROCESSING] Mode=normalize: dividing by 255.0 -> [0.0, 1.0]")
    elif mode == "rescaling":
        print("[PREPROCESSING] Mode=rescaling: passing raw [0,255] (Rescaling layer in model)")
    elif mode == "mobilenet":
        print("[PREPROCESSING] Mode=mobilenet: passing raw [0,255] (MobileNetV2 preprocess_input in graph)")

    return frames_float



def check_ffmpeg_available() -> bool:
    """Check if ffmpeg executable is present in PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio_from_video(video_path: str, output_wav_path: str = None) -> str:
    """
    Extracts audio track from a video file using FFmpeg.
    Saves as 16kHz mono 16-bit PCM WAV.
    Returns path to WAV file if audio exists, or None if video has no audio.
    """
    if not check_ffmpeg_available():
        raise RuntimeError("FFmpeg is not installed or not in system PATH.")

    if output_wav_path is None:
        fd, output_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        output_wav_path
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        if proc.returncode != 0:
            if os.path.exists(output_wav_path):
                os.remove(output_wav_path)
            return None

        # Check if the generated file actually has audio content
        if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) < 200:
            if os.path.exists(output_wav_path):
                os.remove(output_wav_path)
            return None

        return output_wav_path

    except Exception:
        if os.path.exists(output_wav_path):
            os.remove(output_wav_path)
        return None


def extract_audio_features(audio_path: str,
                           target_sr: int = TARGET_SAMPLE_RATE,
                           duration: float = AUDIO_DURATION_SECONDS,
                           n_mels: int = N_MELS,
                           hop_length: int = HOP_LENGTH,
                           n_fft: int = N_FFT) -> np.ndarray:
    """
    Loads an audio file and computes normalized log-mel spectrogram features.
    Guarantees fixed duration representation:
    - Padded with zeros if shorter than target duration.
    - Sliced (center segment) if longer than target duration.
    Returns shape: (1, n_mels, time_steps, 1) float32 array in [0.0, 1.0].
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio as mono at target sample rate
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    target_length = int(target_sr * duration)
    if len(y) < target_length:
        # Pad with zeros
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode="constant")
    elif len(y) > target_length:
        # Center crop
        start = (len(y) - target_length) // 2
        y = y[start : start + target_length]

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    # Convert to log-mel spectrogram (dB scale)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Min-Max Normalization to [0.0, 1.0]
    min_val = log_mel.min()
    max_val = log_mel.max()
    if max_val - min_val > 1e-6:
        norm_spec = (log_mel - min_val) / (max_val - min_val)
    else:
        norm_spec = np.zeros_like(log_mel)

    # Format to (1, n_mels, time_steps, 1)
    feature_tensor = np.expand_dims(norm_spec, axis=(0, -1)).astype(np.float32)
    return feature_tensor
