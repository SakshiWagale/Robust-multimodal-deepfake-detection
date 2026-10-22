import os
import numpy as np
import tensorflow as tf
from app.preprocessing import extract_frames, preprocess_frames, has_rescaling_layer, VIDEO_FRAME_SIZE

CLASS_MAPPING = {
    0: "fake",
    1: "real"
}


class VideoDetector:
    def __init__(self, model_path: str = "models/video_classifier.keras"):
        self.model_path = model_path
        self.model = None
        self.has_rescaling = False
        self.input_shape = None
        self.class_mapping = CLASS_MAPPING

    def load_model(self):
        """
        Loads the trained Keras video classifier model.
        Prints architecture and mapping info upon loading.
        Raises FileNotFoundError if model has not been trained yet.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Video model not found at '{self.model_path}'. "
                f"Please train the model first by running: python training/train_video.py"
            )

        print(f"[VIDEO MODEL] Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        self.has_rescaling = has_rescaling_layer(self.model)
        self.input_shape = self.model.input_shape

        preprocessing_desc = (
            "Embedded Rescaling Layer (Inputs scaled [0, 255] internally)"
            if self.has_rescaling
            else "Explicit Normalization (Inputs divided by 255.0 to [0.0, 1.0])"
        )

        print("--------------------------------------------------")
        print("MODEL LOADED: VIDEO MODEL")
        print(f"VIDEO MODEL PATH: {self.model_path}")
        print(f"VIDEO INPUT SHAPE: {self.input_shape}")
        print(f"CLASS MAPPING: fake = 0, real = 1 (Output sigmoid > threshold => REAL)")
        print(f"VIDEO PREPROCESSING: {preprocessing_desc}")
        print("--------------------------------------------------")

    def predict(self, media_path: str, num_frames: int = 16, aggregation: str = "mean") -> dict:
        """
        Extracts frames from video/image, runs inference on each frame,
        and aggregates frame probabilities into a single video-level score.
        Score represents probability of REAL:
            0.0 -> Definitely FAKE
            1.0 -> Definitely REAL
        """
        if self.model is None:
            self.load_model()

        # Extract frames using OpenCV
        frames = extract_frames(media_path, num_frames=num_frames, target_size=VIDEO_FRAME_SIZE)

        # Preprocess frames matching the exact model requirements
        preprocessed = preprocess_frames(frames, model=self.model)

        # Predict frame probabilities batch-wise
        # Output shape is (N, 1) with sigmoid activation
        raw_predictions = self.model.predict(preprocessed, verbose=0)
        frame_scores = raw_predictions.flatten().astype(float).tolist()

        if aggregation == "median":
            aggregated_score = float(np.median(frame_scores))
        else:
            aggregated_score = float(np.mean(frame_scores))

        return {
            "score": round(aggregated_score, 4),
            "frame_count": len(frame_scores),
            "frame_scores": [round(s, 4) for s in frame_scores],
            "aggregation": aggregation,
            "status": "completed"
        }
