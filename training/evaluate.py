import os
import sys
import glob
import argparse
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from app.preprocessing import detect_media_type, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS
from app.video_detector import VideoDetector
from app.audio_detector import AudioDetector
from app.fusion import fuse_predictions

THRESHOLDS_TO_EVALUATE = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
CLASS_NAMES = ["fake", "real"]  # fake = 0, real = 1


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50):
    """
    Computes confusion matrix elements, accuracy, precision, recall, and F1 at a specific threshold.
    y_true: 0 (fake), 1 (real)
    y_prob: probability of being real
    """
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm layout:
    # [ [TN (fake as fake), FP (fake as real)],
    #   [FN (real as fake), TP (real as real)] ]
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": threshold,
        "fake_correct (TN)": int(tn),
        "fake_incorrect (FP)": int(fp),
        "real_correct (TP)": int(tp),
        "real_incorrect (FN)": int(fn),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist()
    }


def print_metrics_report(title: str, results: dict):
    print(f"\n{'='*70}")
    print(f" {title.upper()} EVALUATION REPORT (Threshold = {results['threshold']:.2f})")
    print(f"{'='*70}")
    print(f"  FAKE correctly classified (TN)  : {results['fake_correct (TN)']}")
    print(f"  FAKE incorrectly classified (FP): {results['fake_incorrect (FP)']}")
    print(f"  REAL correctly classified (TP)  : {results['real_correct (TP)']}")
    print(f"  REAL incorrectly classified (FN): {results['real_incorrect (FN)']}")
    print(f"  -------------------------------------------------------------")
    print(f"  Overall Accuracy                : {results['accuracy'] * 100:.2f}%")
    print(f"  Precision                       : {results['precision']:.4f}")
    print(f"  Recall                          : {results['recall']:.4f}")
    print(f"  F1 Score                        : {results['f1_score']:.4f}")
    print(f"  -------------------------------------------------------------")
    print("  Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"    {results['confusion_matrix']}")
    print(f"{'='*70}\n")


def print_threshold_sweep_report(title: str, y_true: np.ndarray, y_prob: np.ndarray):
    print(f"\n{'-'*85}")
    print(f" THRESHOLD SWEEP ANALYSIS: {title.upper()}")
    print(f" (Reported transparently without post-hoc selection bias)")
    print(f"{'-'*85}")
    header = f"{'Threshold':^10} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} | {'TN/FP/FN/TP':^18}"
    print(header)
    print(f"{'-'*85}")

    for t in THRESHOLDS_TO_EVALUATE:
        m = calculate_metrics(y_true, y_prob, threshold=t)
        counts = f"{m['fake_correct (TN)']}/{m['fake_incorrect (FP)']}/{m['real_incorrect (FN)']}/{m['real_correct (TP)']}"
        print(f"{t:^10.2f} | {m['accuracy']*100:^9.2f}% | {m['precision']:^10.4f} | {m['recall']:^10.4f} | {m['f1_score']:^10.4f} | {counts:^18}")
    print(f"{'-'*85}\n")


def evaluate_video_test_set(test_dir: str = "dataset/video/test", model_path: str = "models/video_classifier.keras"):
    detector = VideoDetector(model_path=model_path)
    detector.load_model()

    y_true = []
    y_prob = []

    print(f"\n[VIDEO EVALUATION] Scanning test dataset at: {test_dir}")
    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue

        files = []
        for ext in list(VIDEO_EXTENSIONS) + list(IMAGE_EXTENSIONS):
            files.extend(glob.glob(os.path.join(cls_dir, f"*{ext}")))
            files.extend(glob.glob(os.path.join(cls_dir, f"*{ext.upper()}")))

        for fpath in files:
            try:
                res = detector.predict(fpath)
                y_true.append(label_idx)
                y_prob.append(res["score"])
            except Exception as e:
                print(f"  [Error] Skipping {fpath}: {e}")

    if not y_true:
        print("[ERROR] No valid test samples found for video evaluation.")
        return

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    standard_report = calculate_metrics(y_true, y_prob, threshold=0.50)
    print_metrics_report("Video Classifier (Test Set)", standard_report)
    print_threshold_sweep_report("Video Classifier", y_true, y_prob)


def evaluate_audio_test_set(test_dir: str = "dataset/audio/test", model_path: str = "models/audio_classifier.keras"):
    detector = AudioDetector(model_path=model_path)
    detector.load_model()

    y_true = []
    y_prob = []

    print(f"\n[AUDIO EVALUATION] Scanning test dataset at: {test_dir}")
    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue

        files = []
        for ext in list(AUDIO_EXTENSIONS):
            files.extend(glob.glob(os.path.join(cls_dir, f"*{ext}")))
            files.extend(glob.glob(os.path.join(cls_dir, f"*{ext.upper()}")))

        for fpath in files:
            try:
                res = detector.predict(fpath)
                if res.get("status") == "completed":
                    y_true.append(label_idx)
                    y_prob.append(res["score"])
            except Exception as e:
                print(f"  [Error] Skipping {fpath}: {e}")

    if not y_true:
        print("[ERROR] No valid test samples found for audio evaluation.")
        return

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    standard_report = calculate_metrics(y_true, y_prob, threshold=0.50)
    print_metrics_report("Audio Classifier (Test Set)", standard_report)
    print_threshold_sweep_report("Audio Classifier", y_true, y_prob)


def verify_single_file(file_path: str, expected: str = None, threshold: float = 0.50):
    """
    Validates a single known fake or real file as required:
    Prints:
        FILE:
        EXPECTED:
        MODEL SCORE:
        MODEL VERDICT:
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Verification file not found: {file_path}")

    media_type = detect_media_type(file_path)
    video_score = None
    audio_score = None

    if media_type in ("video", "image"):
        v_det = VideoDetector(model_path="models/video_classifier.keras")
        v_det.load_model()
        v_res = v_det.predict(file_path)
        video_score = v_res["score"]

        if media_type == "video" and os.path.exists("models/audio_classifier.keras"):
            a_det = AudioDetector(model_path="models/audio_classifier.keras")
            a_det.load_model()
            a_res = a_det.predict(file_path)
            if a_res.get("status") == "completed":
                audio_score = a_res["score"]

    elif media_type == "audio":
        a_det = AudioDetector(model_path="models/audio_classifier.keras")
        a_det.load_model()
        a_res = a_det.predict(file_path)
        audio_score = a_res["score"]

    fusion_res = fuse_predictions(video_score=video_score, audio_score=audio_score, threshold=threshold)

    print("\n" + "=" * 50)
    print(f"FILE: {file_path}")
    print(f"EXPECTED: {expected.upper() if expected else 'NOT SPECIFIED'}")
    print(f"MODEL SCORE: {fusion_res['final_score']}")
    print(f"MODEL VERDICT: {fusion_res['verdict']}")
    print(f"CONFIDENCE: {fusion_res['confidence']}%")
    print(f"THRESHOLD: {threshold}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Models")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation on test datasets")
    parser.add_argument("--modality", type=str, default="both", choices=["video", "audio", "both"], help="Modality to evaluate")
    parser.add_argument("--verify", action="store_true", help="Verify single file")
    parser.add_argument("--file", type=str, default=None, help="Path to file for single verification")
    parser.add_argument("--expected", type=str, default=None, choices=["fake", "real", "FAKE", "REAL"], help="Expected class")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification decision threshold")
    args = parser.parse_args()

    if args.verify:
        if not args.file:
            print("[ERROR] Please provide --file path when using --verify.")
        else:
            verify_single_file(args.file, expected=args.expected, threshold=args.threshold)
    elif args.batch:
        if args.modality in ("video", "both"):
            evaluate_video_test_set()
        if args.modality in ("audio", "both"):
            evaluate_audio_test_set()
    else:
        # Default behavior: run batch evaluation
        evaluate_video_test_set()
        evaluate_audio_test_set()
