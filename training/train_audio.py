import os
import sys
import glob
import argparse
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from app.preprocessing import extract_audio_features, AUDIO_EXTENSIONS

CLASS_MAPPING = {0: "fake", 1: "real"}
CLASS_NAMES = ["fake", "real"]
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16


def build_audio_classifier(input_shape=(128, 94, 1)) -> tf.keras.Model:
    """
    Constructs a 2D CNN audio deepfake classifier operating on Log-Mel Spectrograms.
    Binary classification output with Sigmoid activation:
        0 = fake
        1 = real
    """
    inputs = layers.Input(shape=input_shape, name="spectrogram_input")

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Global Pooling and Dense Classifier
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="audio_deepfake_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model


def load_dataset(data_dir: str):
    """
    Loads audio files from 'fake' (label=0) and 'real' (label=1) subdirectories,
    computes log-mel spectrograms, and returns X, y numpy arrays.
    """
    X_list = []
    y_list = []

    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            continue

        audio_files = []
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(glob.glob(os.path.join(cls_dir, f"*{ext}")))
            audio_files.extend(glob.glob(os.path.join(cls_dir, f"*{ext.upper()}")))

        print(f"[{cls.upper()}] Found {len(audio_files)} audio files in {cls_dir}")

        for filepath in audio_files:
            try:
                features = extract_audio_features(filepath)
                # features has shape (1, n_mels, time_steps, 1)
                X_list.append(features[0])
                y_list.append(label_idx)
            except Exception as e:
                print(f"  [Warning] Failed to extract features from {filepath}: {e}")

    if not X_list:
        raise ValueError(f"No valid audio samples found in {data_dir}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def train(train_dir: str = "dataset/audio/train",
          valid_dir: str = "dataset/audio/valid",
          model_save_path: str = "models/audio_classifier.keras",
          epochs: int = DEFAULT_EPOCHS,
          batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Executes training of audio deepfake classifier.
    """
    print("\n" + "=" * 60)
    print("           TRAINING AUDIO DEEPFAKE CLASSIFIER            ")
    print("=" * 60)
    print(f"CLASS MAPPING: {CLASS_MAPPING}")
    print(f"TRAIN DIR: {train_dir}")
    print(f"VALID DIR: {valid_dir}")
    print(f"MODEL DESTINATION: {model_save_path}")

    print("\n[DATASET] Extracting features for training set...")
    X_train, y_train = load_dataset(train_dir)
    print(f"Train samples: {len(X_train)}, Shape: {X_train.shape}")

    print("\n[DATASET] Extracting features for validation set...")
    X_val, y_val = load_dataset(valid_dir)
    print(f"Validation samples: {len(X_val)}, Shape: {X_val.shape}")

    input_shape = X_train.shape[1:]  # (128, 94, 1)
    print(f"Input Spectrogram Shape: {input_shape}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model = build_audio_classifier(input_shape=input_shape)
    model.summary()

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    print("\n[TRAINING] Starting model fit...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stopping, checkpoint]
    )

    model.save(model_save_path)
    print(f"\n[COMPLETE] Audio deepfake classifier saved successfully to: {model_save_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Classifier")
    parser.add_argument("--train_dir", type=str, default="dataset/audio/train", help="Path to training set")
    parser.add_argument("--valid_dir", type=str, default="dataset/audio/valid", help="Path to validation set")
    parser.add_argument("--output", type=str, default="models/audio_classifier.keras", help="Model save path")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        model_save_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
