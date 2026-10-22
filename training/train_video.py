"""
train_video.py — Video Deepfake Classifier Training
====================================================
Architecture : MobileNetV2 (ImageNet pre-trained) + custom classification head
Class mapping : 0 = fake  |  1 = real
Output        : models/video_classifier.keras

TRAINING STRATEGY
-----------------
Phase 1 — Feature extraction (backbone frozen, head trains for 10 epochs)
Phase 2 — Fine-tuning (top 40 backbone layers unfrozen, very low LR for N epochs)

Why MobileNetV2?
  - Pre-trained on ImageNet (1.2 million real images of faces, objects, etc.)
  - Already understands texture, edges, facial features
  - Much better starting point than a CNN trained from scratch on synthetic data
  - Fine-tuning adapts it specifically to deepfake artifacts

PREPROCESSING NOTE
------------------
MobileNetV2 uses its own preprocessing (tf.keras.applications.mobilenet_v2.preprocess_input)
which maps [0,255] -> [-1, 1].  A Rescaling(1/255) layer is NOT used here.
The preprocessing.py inference code auto-detects this via has_rescaling_layer().
"""

import os
import sys
import glob
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLASS_NAMES = ["fake", "real"]
CLASS_MAPPING = {0: "fake", 1: "real"}
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
DEFAULT_PHASE1_EPOCHS = 10
DEFAULT_PHASE2_EPOCHS = 15


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_mobilenet_classifier(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)) -> tf.keras.Model:
    """
    MobileNetV2 transfer-learning model for binary deepfake classification.
    
    Class mapping:
        0 = FAKE
        1 = REAL
    Sigmoid output: score near 1.0 → REAL, near 0.0 → FAKE
    
    Preprocessing: MobileNetV2 uses preprocess_input (not Rescaling layer).
    """
    print("\n[MODEL] Building MobileNetV2 transfer learning classifier...")
    print(f"[MODEL] Class mapping: {CLASS_MAPPING}")

    inputs = layers.Input(shape=input_shape, name="frame_input")

    # MobileNetV2 preprocessing: maps [0,255] -> [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # MobileNetV2 backbone — ImageNet weights, no top classification layer
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False  # Frozen during Phase 1
    print(f"[MODEL] MobileNetV2 backbone loaded. Layers: {len(backbone.layers)}")
    print("[MODEL] Phase 1: Backbone FROZEN — only classification head trains")

    x = backbone(x, training=False)

    # Classification head (clean, robust architecture)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="head_dense1")(x)
    x = layers.Dropout(0.3, name="head_dropout1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenet_deepfake_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    print(f"[MODEL] Total parameters: {model.count_params():,}")
    print(f"[MODEL] Trainable parameters (Phase 1): {sum([np.prod(v.shape) for v in model.trainable_variables]):,}")
    return model


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int = 30, learning_rate: float = 2e-5):
    """
    Phase 2: Unfreeze the top `num_layers` of MobileNetV2 backbone for fine-tuning.
    Very low learning rate to avoid destroying pre-trained features.
    """
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        print("[WARNING] Could not find backbone sub-model. Skipping unfreeze.")
        return model

    backbone.trainable = True
    # Freeze all except the last `num_layers`
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False

    trainable_count = sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"\n[FINE-TUNE] Unfrozen top {num_layers} backbone layers")
    print(f"[FINE-TUNE] Trainable parameters now: {trainable_count:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_frame_cache(data_dir: str, cache_dir: str, frames_per_video: int = 8):
    """
    Extracts frames from video files into cache_dir for tf.data streaming.
    If image files are present, resizes and copies them.
    """
    os.makedirs(cache_dir, exist_ok=True)
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    for cls in CLASS_NAMES:
        src_cls_dir = os.path.join(data_dir, cls)
        dst_cls_dir = os.path.join(cache_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        if not os.path.exists(src_cls_dir):
            print(f"[WARNING] Class directory not found: {src_cls_dir}")
            continue

        # Copy/resize image files
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(src_cls_dir, ext)))

        for img_path in images:
            filename = os.path.basename(img_path)
            dst_path = os.path.join(dst_cls_dir, filename)
            if not os.path.exists(dst_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    cv2.imwrite(dst_path, img_resized)

        # Extract frames from video files
        videos = []
        for ext in video_extensions:
            videos.extend(glob.glob(os.path.join(src_cls_dir, ext)))

        for vid_path in videos:
            base_vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print(f"[WARNING] Cannot open video: {vid_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                indices = np.linspace(0, total_frames - 1, min(frames_per_video, total_frames), dtype=int)
            else:
                indices = list(range(min(frames_per_video, 20)))

            curr_idx = 0
            extracted = 0
            idx_set = set(indices)

            while cap.isOpened() and extracted < len(indices):
                ret, frame = cap.read()
                if not ret:
                    break
                if curr_idx in idx_set:
                    out_name = f"{base_vid_name}_frame_{curr_idx:04d}.jpg"
                    out_path = os.path.join(dst_cls_dir, out_name)
                    if not os.path.exists(out_path):
                        frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                        cv2.imwrite(out_path, frame_resized)
                    extracted += 1
                curr_idx += 1
            cap.release()

        total_images = len(glob.glob(os.path.join(dst_cls_dir, "*.jpg")))
        print(f"[CACHE] {cls}: {total_images} images ready")


def make_dataset(directory: str, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    """Creates a tf.data image dataset from a directory."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=shuffle
    )
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Light augmentation pipeline applied only during training.
    Helps generalize to real-world deepfake variations.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.04),
    ], name="augmentation")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    train_dir: str = "dataset/video/train",
    valid_dir: str = "dataset/video/valid",
    model_save_path: str = "models/video_classifier.keras",
    phase1_epochs: int = DEFAULT_PHASE1_EPOCHS,
    phase2_epochs: int = DEFAULT_PHASE2_EPOCHS,
    batch_size: int = BATCH_SIZE,
    unfreeze_layers: int = 30
):
    """
    Two-phase transfer learning training:
      Phase 1: Train classification head only (backbone frozen)
      Phase 2: Fine-tune top backbone layers + head (very low LR)
    """
    print("\n" + "=" * 65)
    print("     TRAINING VIDEO DEEPFAKE CLASSIFIER — MobileNetV2      ")
    print("=" * 65)
    print(f"CLASS MAPPING   : {CLASS_MAPPING}")
    print(f"TRAIN DIR       : {train_dir}")
    print(f"VALID DIR       : {valid_dir}")
    print(f"MODEL SAVE PATH : {model_save_path}")
    print(f"PHASE 1 EPOCHS  : {phase1_epochs}")
    print(f"PHASE 2 EPOCHS  : {phase2_epochs}")
    print(f"BATCH SIZE      : {batch_size}")

    # Prepare frame cache
    cache_base = "dataset/video/.cache"
    train_cache = os.path.join(cache_base, "train")
    valid_cache = os.path.join(cache_base, "valid")

    print("\n[DATASET] Preparing dataset frames...")
    prepare_frame_cache(train_dir, train_cache)
    prepare_frame_cache(valid_dir, valid_cache)

    # Build datasets
    train_ds = make_dataset(train_cache, batch_size, shuffle=True)
    valid_ds = make_dataset(valid_cache, batch_size, shuffle=False)

    # Count samples per class
    for cls in CLASS_NAMES:
        count = len(glob.glob(os.path.join(train_cache, cls, "*.jpg")))
        print(f"[DATASET] Train '{cls}': {count} images")
    for cls in CLASS_NAMES:
        count = len(glob.glob(os.path.join(valid_cache, cls, "*.jpg")))
        print(f"[DATASET] Valid '{cls}': {count} images")

    # Apply data augmentation to training set
    augment = build_augmentation_layer()
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # -----------------------------------------------------------------------
    # PHASE 1: Train classification head only
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PHASE 1: Feature Extraction (backbone FROZEN)")
    print("-" * 65)

    model = build_mobilenet_classifier(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    checkpoint_phase1 = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    early_stop_phase1 = callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    history1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=phase1_epochs,
        callbacks=[checkpoint_phase1, early_stop_phase1]
    )

    # -----------------------------------------------------------------------
    # PHASE 2: Fine-tune top backbone layers
    # -----------------------------------------------------------------------
    if phase2_epochs > 0:
        print("\n" + "-" * 65)
        print("PHASE 2: Fine-tuning (top backbone layers UNFROZEN)")
        print("-" * 65)

        model = unfreeze_top_layers(model, num_layers=unfreeze_layers, learning_rate=2e-5)

        checkpoint_phase2 = callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )
        early_stop_phase2 = callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        history2 = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=phase2_epochs,
            callbacks=[checkpoint_phase2, early_stop_phase2]
        )
    else:
        history2 = None

    print(f"\n[COMPLETE] Best model saved to: {model_save_path}")
    print("[COMPLETE] Training finished successfully.")

    return history1, history2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Video Deepfake Classifier (MobileNetV2)")
    parser.add_argument("--train_dir",       type=str, default="dataset/video/train")
    parser.add_argument("--valid_dir",       type=str, default="dataset/video/valid")
    parser.add_argument("--output",          type=str, default="models/video_classifier.keras")
    parser.add_argument("--phase1_epochs",   type=int, default=DEFAULT_PHASE1_EPOCHS)
    parser.add_argument("--phase2_epochs",   type=int, default=DEFAULT_PHASE2_EPOCHS)
    parser.add_argument("--batch_size",      type=int, default=BATCH_SIZE)
    parser.add_argument("--unfreeze_layers", type=int, default=40)
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        model_save_path=args.output,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        unfreeze_layers=args.unfreeze_layers
    )
