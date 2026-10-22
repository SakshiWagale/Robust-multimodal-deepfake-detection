# Multimodal Deepfake Detection System

A complete, production-grade deep-learning system that analyzes uploaded **video**, **audio**, and **image** media to predict whether content is **REAL** or **FAKE** using trained neural networks, OpenCV spatio-temporal frame analysis, Librosa Log-Mel Spectrogram extraction, FFmpeg demuxing, and weighted decision fusion.

> **CRITICAL REALITY & GENERALIZATION NOTICE**  
> No AI model can detect every arbitrary deepfake with 100% accuracy. Deepfake generation methods (e.g., diffusion models, NeRFs, neural voice cloning, GANs) evolve rapidly. Model detection performance depends heavily on the diversity, resolution, and quality of the training datasets (such as FaceForensics++, DFDC, Celeb-DF, ASVspoof). Predictions reflect probabilistic pattern matching against trained forensic features and must be interpreted as statistical indicators rather than infallible proofs.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Technology Stack](#technology-stack)
4. [Dataset Structure](#dataset-structure)
5. [Windows PowerShell Setup & Virtual Environment](#windows-powershell-setup--virtual-environment)
6. [FFmpeg Setup on Windows](#ffmpeg-setup-on-windows)
7. [Dependencies Installation](#dependencies-installation)
8. [Dataset Preparation](#dataset-preparation)
9. [Model Training](#model-training)
10. [Model Verification & Evaluation](#model-verification--evaluation)
11. [Running the FastAPI Web Application](#running-the-fastapi-web-application)
12. [REST API Documentation (`/predict`)](#rest-api-documentation-predict)
13. [Frontend Dashboard Walkthrough](#frontend-dashboard-walkthrough)
14. [Troubleshooting & Common Windows Errors](#troubleshooting--common-windows-errors)

---

## 1. Project Overview

Modern deepfakes manipulate both visual streams (face-swapping, lip-syncing, reenactment) and auditory streams (voice cloning, text-to-speech synthesis). This project deploys an end-to-end multimodal pipeline:

1. **Visual Stream Detector**:
   - Extracts uniformly sampled frames across video duration using OpenCV.
   - Standardizes frames to $(224 \times 224 \times 3)$.
   - Evaluates frame probabilities using a 2D CNN with embedded `Rescaling(1./255)`.
   - Aggregates frame probabilities using statistical mean.
   - Explicit Class Mapping: `fake = 0`, `real = 1`.
2. **Acoustic Stream Detector**:
   - Demuxes the audio track from video containers using FFmpeg.
   - Normalizes audio to $16\text{ kHz}$ mono 16-bit PCM over a fixed $3.0\text{-second}$ window.
   - Extracts 128-band normalized Log-Mel Spectrograms using Librosa.
   - Evaluates spectrograms via a dedicated 2D CNN classifier.
   - Explicit Class Mapping: `fake = 0`, `real = 1`.
3. **Multimodal Decision Fusion**:
   - When both video and audio streams are available:
     $$\text{Final Probability} = \frac{w_{\text{video}} \cdot P_{\text{video}}(\text{real}) + w_{\text{audio}} \cdot P_{\text{audio}}(\text{real})}{w_{\text{video}} + w_{\text{audio}}}$$
   - Default Weights: $w_{\text{video}} = 0.6$, $w_{\text{audio}} = 0.4$.
   - Configurable Decision Threshold: $\tau = 0.50$.
   - If $\text{Final Probability} \ge \tau \implies \mathbf{REAL}$; otherwise $\mathbf{FAKE}$.
   - If audio extraction fails or is unavailable: seamlessly falls back to video-only inference without inventing artificial data.
   - If audio-only is uploaded: executes audio inference directly.

---

## 2. Folder Structure

```
multimodal_deepfake_detection/
│
├── dataset/
│   ├── video/
│   │   ├── train/
│   │   │   ├── fake/
│   │   │   └── real/
│   │   ├── valid/
│   │   │   ├── fake/
│   │   │   └── real/
│   │   └── test/
│   │       ├── fake/
│   │       └── real/
│   │
│   └── audio/
│       ├── train/
│       │   ├── fake/
│       │   └── real/
│       ├── valid/
│       │   ├── fake/
│       │   └── real/
│       └── test/
│           ├── fake/
│           └── real/
│
├── models/
│   ├── video_classifier.keras       # Trained video CNN model
│   └── audio_classifier.keras       # Trained audio CNN model
│
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI server & route handlers
│   ├── video_detector.py            # Video inference engine
│   ├── audio_detector.py            # Audio inference engine
│   ├── preprocessing.py             # OpenCV/Librosa/FFmpeg preprocessing
│   └── fusion.py                    # Configurable multimodal fusion engine
│
├── training/
│   ├── __init__.py
│   ├── train_video.py               # Video classifier training script
│   ├── train_audio.py               # Audio classifier training script
│   ├── evaluate.py                  # Evaluation & single-file verification script
│   └── generate_sample_dataset.py   # Synthetic dataset builder for pipeline testing
│
├── static/
│   ├── uploads/                     # Upload directory for analysis files
│   └── results/
│
├── templates/
│   ├── index.html                   # Overview & system status dashboard
│   ├── upload.html                  # Drag-and-drop media upload interface
│   └── result.html                  # Comprehensive forensic results view
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Technology Stack

- **Deep Learning Framework**: TensorFlow 2.x & Keras 3.x
- **Computer Vision**: OpenCV (`opencv-python`)
- **Acoustic Analysis**: Librosa & SoundFile
- **Metrics & Evaluation**: Scikit-learn & NumPy
- **Backend Service**: FastAPI & Uvicorn (ASGI)
- **Multimedia Processing**: FFmpeg
- **Frontend UI**: Vanilla HTML5, CSS3, JavaScript (responsive, dark mode, no external CDNs)

---

## 4. Dataset Structure

The project expects datasets in `dataset/` partitioned into `train`, `valid`, and `test` splits:

- **Video Folders** (`dataset/video/{train,valid,test}/{fake,real}`):
  - Accepts video clips (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) or frame image sequences (`.jpg`, `.png`).
- **Audio Folders** (`dataset/audio/{train,valid,test}/{fake,real}`):
  - Accepts audio files (`.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`).

Class folders **must** be named strictly `fake` (label 0) and `real` (label 1).

---

## 5. Windows PowerShell Setup & Virtual Environment

Open Windows PowerShell in the project directory:

```powershell
# Navigate to project root
cd "C:\Users\SAKSHI RAJESH WAGALE\.gemini\antigravity\scratch\multimodal_deepfake_detection"

# Create a virtual environment
python -m venv venv

# If PowerShell script execution is restricted, enable execution for CurrentUser:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

---

## 6. FFmpeg Setup on Windows

The application uses FFmpeg to demux audio streams from uploaded videos.

### Verify FFmpeg
Run in PowerShell:
```powershell
ffmpeg -version
```

### If FFmpeg is missing:
Install via **winget** (official Windows Package Manager):
```powershell
winget install Gyan.FFmpeg
```
Or via Chocolatey:
```powershell
choco install ffmpeg
```

#### Adding FFmpeg manually to PATH (if downloaded as a ZIP):
1. Extract FFmpeg to `C:\ffmpeg`.
2. Open PowerShell as Administrator and run:
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ffmpeg\bin", [EnvironmentVariableTarget]::Machine)
   ```
3. Restart your PowerShell window and verify with `ffmpeg -version`.

---

## 7. Dependencies Installation

Install all required packages inside your activated virtual environment:

```powershell
python -m pip install -r requirements.txt
```

---

## 8. Dataset Preparation

### Option A: Use Your Real Forensic Datasets
Place your real videos and deepfakes (e.g., from FaceForensics++, DFDC, ASVspoof) into:
- `dataset/video/train/fake/` and `dataset/video/train/real/`
- `dataset/video/valid/fake/` and `dataset/video/valid/real/`
- `dataset/video/test/fake/` and `dataset/video/test/real/`
- `dataset/audio/train/fake/` and `dataset/audio/train/real/`
- `dataset/audio/valid/fake/` and `dataset/audio/valid/real/`
- `dataset/audio/test/fake/` and `dataset/audio/test/real/`

### Option B: Generate Synthetic Baseline Dataset
To test and verify the entire machine learning pipeline end-to-end immediately:
```powershell
python training/generate_sample_dataset.py
```
This synthesizes controlled real and fake video clips and audio signals across all splits.

---

## 9. Model Training

Training and inference are strictly separated. The web application will **not** train models on startup. Models must be explicitly trained using the dedicated training scripts.

### Step 1: Train the Video Classifier
```powershell
python training/train_video.py --epochs 15 --batch_size 16
```
- Extracts frames from video/image datasets.
- Trains CNN architecture with `Rescaling(1./255)` layer.
- Saves the best model checkpoint to `models/video_classifier.keras`.

### Step 2: Train the Audio Classifier
```powershell
python training/train_audio.py --epochs 20 --batch_size 16
```
- Extracts 128-band Log-Mel Spectrograms using Librosa at 16 kHz.
- Trains 2D CNN classifier with EarlyStopping and ModelCheckpoint.
- Saves the best model checkpoint to `models/audio_classifier.keras`.

---

## 10. Model Verification & Evaluation

### Single File Verification (Known Fake and Known Real)
Verify models against individual ground-truth files before starting the web UI:

#### Test Known Real File:
```powershell
python training/evaluate.py --verify --file dataset/video/test/real/sample_real_001.mp4 --expected real
```

#### Test Known Fake File:
```powershell
python training/evaluate.py --verify --file dataset/video/test/fake/sample_fake_001.mp4 --expected fake
```

The script prints:
```
==================================================
FILE: dataset/video/test/fake/sample_fake_001.mp4
EXPECTED: FAKE
MODEL SCORE: 0.1420
MODEL VERDICT: FAKE
CONFIDENCE: 85.8%
THRESHOLD: 0.50
==================================================
```

### Batch Evaluation & Threshold Sweeping
Run complete statistical evaluation on the test split:
```powershell
python training/evaluate.py --batch --modality both
```
This generates:
- True Positives (REAL correct)
- True Negatives (FAKE correct)
- False Positives (FAKE classified as REAL)
- False Negatives (REAL classified as FAKE)
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- Multi-threshold sweep across `0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70` without post-hoc selection bias.

---

## 11. Running the FastAPI Web Application

Launch the FastAPI application with Uvicorn:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

### Startup Verification Log
On startup, the system prints model information:
```
============================================================
      MULTIMODAL DEEPFAKE DETECTION SYSTEM STARTUP       
============================================================
[SYSTEM] FFmpeg detected in system PATH.
[VIDEO MODEL] Loading model from models/video_classifier.keras...
--------------------------------------------------
MODEL LOADED: VIDEO MODEL
VIDEO MODEL PATH: models/video_classifier.keras
VIDEO INPUT SHAPE: (None, 224, 224, 3)
CLASS MAPPING: fake = 0, real = 1 (Output sigmoid > threshold => REAL)
VIDEO PREPROCESSING: Embedded Rescaling Layer (Inputs scaled [0, 255] internally)
--------------------------------------------------
[AUDIO MODEL] Loading model from models/audio_classifier.keras...
--------------------------------------------------
MODEL LOADED: AUDIO MODEL
AUDIO MODEL PATH: models/audio_classifier.keras
AUDIO INPUT SHAPE: (None, 128, 94, 1)
CLASS MAPPING: fake = 0, real = 1 (Output sigmoid > threshold => REAL)
AUDIO PREPROCESSING: Log-Mel Spectrogram (128 mel bins, 16000 Hz, 3.0s normalized to [0.0, 1.0])
--------------------------------------------------
[SYSTEM] Both Video and Audio deep learning models loaded successfully.
============================================================
```

If a `.keras` model file is missing, the server logs a clear notification and gracefully refuses inference requests with an informative HTTP 503 error, never returning mock predictions.

---

## 12. REST API Documentation (`/predict`)

### Endpoint: `POST /predict`
Accepts `multipart/form-data`:
- `file`: Media file (video, audio, or image).
- `threshold` *(optional, float, default `0.50`)*: Decision threshold.
- `video_weight` *(optional, float, default `0.6`)*: Fusion weight for video stream.
- `audio_weight` *(optional, float, default `0.4`)*: Fusion weight for audio stream.

### Example PowerShell API Call (`curl` / `Invoke-RestMethod`):
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method Post `
  -Form @{
    file = Get-Item "dataset/video/test/fake/sample_fake_001.mp4"
    threshold = "0.50"
    video_weight = "0.6"
    audio_weight = "0.4"
  } | ConvertTo-Json
```

### JSON Response:
```json
{
  "filename": "sample_fake_001.mp4",
  "media_type": "video",
  "video_score": 0.1245,
  "audio_score": 0.1980,
  "final_score": 0.1539,
  "verdict": "FAKE",
  "confidence": 84.61,
  "threshold": 0.50,
  "analysis_type": "multimodal (video + audio)",
  "status": "completed",
  "file_url": "/static/uploads/ab12cd34_sample_fake_001.mp4",
  "notes": null
}
```

---

## 13. Frontend Dashboard Walkthrough

Open your browser at:
```
http://localhost:8000
```

1. **Overview Page (`/`)**: Displays real-time model status (Loaded / Not Found), system checks (FFmpeg detection), and architecture overview.
2. **Upload Interface (`/upload`)**:
   - Drag-and-drop or file picker.
   - Supports Video (`.mp4`, `.avi`, `.mov`), Audio (`.wav`, `.mp3`), and Images (`.jpg`, `.png`).
   - Advanced settings accordion for custom fusion weights and thresholds.
   - Animated progress spinner during neural forward-pass execution.
3. **Forensic Result Page (`/result`)**:
   - High-contrast verdict hero badge (`REAL` in green or `FAKE` in red).
   - "Prediction generated by trained AI model" guarantee badge.
   - Confidence percentage meter.
   - Side-by-side video and audio score progress bars.
   - Media playback player for instant visual/audio review.

---

## 14. Troubleshooting & Common Windows Errors

### 1. PowerShell Script Execution Blocked (`running scripts is disabled on this system`)
**Symptom**: `.\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled`.  
**Fix**: Run in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. FFmpeg Not Recognized
**Symptom**: `FFmpeg is not installed or not in system PATH`.  
**Fix**: Install via `winget install Gyan.FFmpeg` and restart PowerShell.

### 3. Missing C++ Build Tools or DLL Errors
**Symptom**: `ImportError: DLL load failed while importing _pywrap_tensorflow_internal`.  
**Fix**: Install the Microsoft Visual C++ 2015–2022 Redistributable (x64) from Microsoft's official download page.

### 4. Video Files with No Audio Track
**Behavior**: If an uploaded video does not contain an audio track, the system gracefully reports `audio_score: null` and evaluates the video using video-only inference. It never invents an audio score.

### 5. Out of Memory (OOM) During Video Training
**Fix**: Reduce batch size in `train_video.py`:
```powershell
python training/train_video.py --batch_size 8
```
