import sys
import os
import argparse

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from app.preprocessing import detect_media_type, check_ffmpeg_available
from app.video_detector import VideoDetector
from app.audio_detector import AudioDetector
from app.fusion import fuse_predictions, DEFAULT_VIDEO_WEIGHT, DEFAULT_AUDIO_WEIGHT, DEFAULT_THRESHOLD


def run_detection(media_path: str, threshold: float = DEFAULT_THRESHOLD):
    if not os.path.exists(media_path):
        print(f"\n[ERROR] File not found: {media_path}")
        return

    media_type = detect_media_type(media_path)
    file_size_kb = os.path.getsize(media_path) / 1024.0
    size_str = f"{file_size_kb / 1024.0:.2f} MB" if file_size_kb > 1024 else f"{file_size_kb:.1f} KB"

    print("\n" + "=" * 65)
    print("      VERITAS MULTIMODAL FORENSICS — DIGITAL EVIDENCE REPORT     ")
    print("=" * 65)
    print(f" Target File       : {os.path.basename(media_path)}")
    print(f" Media Modality    : {media_type.upper()}")
    print(f" File Size         : {size_str}")
    print(f" Decision Threshold: {threshold}")
    print("-" * 65)

    video_score = None
    audio_score = None

    if media_type in ("video", "image"):
        print("[*] Analyzing visual frame sequence (MobileNetV2)...")
        v_det = VideoDetector()
        v_det.load_model()
        v_res = v_det.predict(media_path)
        video_score = v_res.get("score")
        print(f"    -> Visual Authenticity Score : {video_score:.4f}")

    if media_type in ("video", "audio") and check_ffmpeg_available():
        print("[*] Analyzing acoustic frequency spectrogram (128-band Mel CNN)...")
        a_det = AudioDetector()
        a_det.load_model()
        a_res = a_det.predict(media_path)
        if a_res.get("status") == "completed":
            audio_score = a_res.get("score")
            print(f"    -> Acoustic Authenticity Score: {audio_score:.4f}")
        else:
            print("    -> No active audio track detected.")

    # Multimodal Late-Fusion
    res = fuse_predictions(
        video_score=video_score,
        audio_score=audio_score,
        video_weight=DEFAULT_VIDEO_WEIGHT,
        audio_weight=DEFAULT_AUDIO_WEIGHT,
        threshold=threshold
    )

    verdict = res["verdict"]
    confidence = res["confidence"]
    final_score = res["final_score"]

    print("=" * 65)
    if verdict == "REAL":
        badge = "[ VERDICT: REAL / AUTHENTIC ]"
        print(f" \033[92m{badge}\033[0m")
    else:
        badge = "[ VERDICT: FAKE / SYNTHETIC DEEPFAKE ]"
        print(f" \033[91m{badge}\033[0m")

    print(f" Confidence        : {confidence}%")
    print(f" Final Fused Score : {final_score}")
    print(f" Pipeline          : {res['analysis_type']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python detect.py <path_to_video_or_image_or_audio>")
        print('Example: python detect.py "amey.mp4"\n')
    else:
        file_path = sys.argv[1]
        run_detection(file_path)
