import os
import numpy as np
import tensorflow as tf
from app.preprocessing import (
    extract_audio_features,
    extract_audio_from_video,
    detect_media_type,
    TARGET_SAMPLE_RATE,
    AUDIO_DURATION_SECONDS,
    N_MELS
)

CLASS_MAPPING = {
    0: "fake",
    1: "real"
}


class AudioDetector:
    def __init__(self, model_path: str = "models/audio_classifier.keras"):
        self.model_path = model_path
        self.model = None
        self.input_shape = None
        self.class_mapping = CLASS_MAPPING

    def load_model(self):
        """
        Loads the trained Keras audio classifier model.
        Prints architecture and mapping info upon loading.
        Raises FileNotFoundError if model has not been trained yet.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Audio model not found at '{self.model_path}'. "
                f"Please train the model first by running: python training/train_audio.py"
            )

        print(f"[AUDIO MODEL] Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        self.input_shape = self.model.input_shape

        preprocessing_desc = (
            f"Log-Mel Spectrogram ({N_MELS} mel bins, {TARGET_SAMPLE_RATE} Hz, "
            f"{AUDIO_DURATION_SECONDS}s normalized to [0.0, 1.0])"
        )

        print("--------------------------------------------------")
        print("MODEL LOADED: AUDIO MODEL")
        print(f"AUDIO MODEL PATH: {self.model_path}")
        print(f"AUDIO INPUT SHAPE: {self.input_shape}")
        print(f"CLASS MAPPING: fake = 0, real = 1 (Output sigmoid > threshold => REAL)")
        print(f"AUDIO PREPROCESSING: {preprocessing_desc}")
        print("--------------------------------------------------")

    def predict(self, audio_or_video_path: str) -> dict:
        """
        Performs inference on audio or extracts audio from a video file.
        Score represents probability of REAL:
            0.0 -> Definitely FAKE
            1.0 -> Definitely REAL
        Returns dict with score and status.
        """
        if self.model is None:
            self.load_model()

        media_type = detect_media_type(audio_or_video_path)
        temp_extracted_wav = None

        if media_type == "video":
            # Extract audio from video via FFmpeg
            temp_extracted_wav = extract_audio_from_video(audio_or_video_path)
            if temp_extracted_wav is None:
                return {
                    "score": None,
                    "status": "audio_track_unavailable",
                    "message": "Video has no audio track or audio extraction failed."
                }
            eval_path = temp_extracted_wav
        else:
            eval_path = audio_or_video_path

        try:
            # Extract log-mel spectrogram
            features = extract_audio_features(eval_path)

            # Predict probability (Sigmoid output: REAL prob)
            prediction = self.model.predict(features, verbose=0)
            score = float(prediction[0][0])

            return {
                "score": round(score, 4),
                "status": "completed",
                "message": "Audio inference successful."
            }
        finally:
            # Clean up temporary extracted WAV file if created
            if temp_extracted_wav and os.path.exists(temp_extracted_wav):
                try:
                    os.remove(temp_extracted_wav)
                except OSError:
                    pass
