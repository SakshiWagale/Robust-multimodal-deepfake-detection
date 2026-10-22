import os
import cv2
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

SPLITS = {
    "train": 8,  # samples per class
    "valid": 4,
    "test": 4
}


def create_sample_video(output_path: str, is_fake: bool, num_frames: int = 30, fps: int = 15):
    """
    Synthesizes a sample video clip with visual motion patterns.
    Real videos: coherent smooth circular trajectory and natural gradient textures.
    Fake videos: discontinuous jitter, boundary blending artifacts, and frequency noise.
    """
    width, height = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    center_x, center_y = width // 2, height // 2
    radius = 40

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if not is_fake:
            # Real: Smooth sinusoidal trajectory and smooth skin/lighting tone
            angle = 2 * np.pi * (i / num_frames)
            cx = int(center_x + 30 * np.cos(angle))
            cy = int(center_y + 20 * np.sin(angle))
            # Smooth gradient background
            for y in range(height):
                frame[y, :, :] = (int(30 + 40 * (y / height)), int(40 + 30 * (y / height)), 60)
            # Smooth coherent circle
            cv2.circle(frame, (cx, cy), radius, (200, 180, 150), -1)
            cv2.circle(frame, (cx, cy), radius, (160, 140, 110), 2)
        else:
            # Fake: Unnatural erratic spatial shift, high-frequency noise, blending boundary mismatch
            cx = int(center_x + 30 * np.cos(i) + np.random.randint(-8, 9))
            cy = int(center_y + 20 * np.sin(i * 1.5) + np.random.randint(-8, 9))
            # High-frequency blocky background
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            # Inconsistent artifacted circle with blurred/jittered borders
            cv2.circle(frame, (cx, cy), radius, (220, 120, 110), -1)
            cv2.circle(frame, (cx + 3, cy - 3), radius + 4, (100, 255, 255), 1)

        out.write(frame)

    out.release()


def create_sample_audio(output_path: str, is_fake: bool, duration: float = 3.0, sr: int = 16000):
    """
    Synthesizes a sample audio WAV signal.
    Real audio: Natural harmonic vocal formant tones with continuous phase.
    Fake audio: Phase discontinuities, robotic frequency warps, synthetic spectral buzz.
    """
    total_samples = int(duration * sr)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    if not is_fake:
        # Real: Natural voice formant simulation (fundamental f0 ~ 150Hz + harmonics)
        f0 = 150.0
        harmonics = [1.0, 0.6, 0.4, 0.25, 0.15]
        signal = np.zeros(total_samples)
        for idx, h_gain in enumerate(harmonics):
            freq = f0 * (idx + 1)
            # Slight natural vibrato
            vibrato = 5.0 * np.sin(2 * np.pi * 5.0 * t)
            signal += h_gain * np.sin(2 * np.pi * (freq + vibrato) * t)

        # Natural soft ambient acoustic envelope
        envelope = np.sin(np.pi * t / duration) ** 0.5
        signal = signal * envelope
        # Low natural noise floor
        signal += np.random.normal(0, 0.01, total_samples)
    else:
        # Fake: Robotic frequency modulation, sudden phase jumps, high-freq vocoder buzz
        signal = np.sin(2 * np.pi * 320.0 * t + np.sin(2 * np.pi * 40.0 * t))
        # Sudden artificial phase skips
        step_len = sr // 4
        for s in range(0, total_samples, step_len):
            signal[s:s + step_len] *= (-1) ** (s // step_len)
        # Add high-frequency robotic buzz
        signal += 0.3 * np.sign(np.sin(2 * np.pi * 1200.0 * t))
        signal += np.random.normal(0, 0.05, total_samples)

    # Normalize float to 16-bit PCM range
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.9

    sf.write(output_path, signal, sr, subtype="PCM_16")


def generate_all_datasets():
    print("=" * 60)
    print(" GENERATING SYNTHETIC SAMPLE DATASET FOR PIPELINE VALIDATION ")
    print("=" * 60)

    categories = ["fake", "real"]

    for split, count in SPLITS.items():
        print(f"\n--- Creating split: {split.upper()} ({count} samples per class) ---")
        for cat in categories:
            is_fake = (cat == "fake")

            # Video directory
            video_dir = os.path.join(DATASET_DIR, "video", split, cat)
            os.makedirs(video_dir, exist_ok=True)

            # Audio directory
            audio_dir = os.path.join(DATASET_DIR, "audio", split, cat)
            os.makedirs(audio_dir, exist_ok=True)

            for i in range(1, count + 1):
                vid_file = os.path.join(video_dir, f"sample_{cat}_{i:03d}.mp4")
                aud_file = os.path.join(audio_dir, f"sample_{cat}_{i:03d}.wav")

                if not os.path.exists(vid_file):
                    create_sample_video(vid_file, is_fake=is_fake)
                if not os.path.exists(aud_file):
                    create_sample_audio(aud_file, is_fake=is_fake)

            print(f"  [OK] {split}/{cat}: {count} videos & {count} audios generated.")

    print("\n[COMPLETE] Sample dataset is ready in dataset/ directory.")
    print("You can now train real models with:")
    print("  python training/train_video.py")
    print("  python training/train_audio.py")


if __name__ == "__main__":
    generate_all_datasets()
