# ── main.py ───────────────────────────────────────────────────────────────────
# Real-World Traditional ML Object Tracker
# Zero Neural Networks — Works on Your Custom Videos
#
# Full pipeline:
#   Upload Videos → (Optional) Annotate → Dataset Split
#   → Grid-Search Hyperparameters (Train) → Select Best Config (Val)
#   → Final Evaluation (Test) → Download Annotated Videos + Report
#
# Algorithms used (zero neural networks):
#   Background Subtraction  : MOG2 / GMG
#   Camera Stabilisation    : Lucas-Kanade Optical Flow
#   Blob Detection          : Morphological operations
#   State Estimation        : Kalman Filter (x,y,w,h,vx,vy)
#   Assignment              : Hungarian Algorithm
#   Re-Identification       : HSV Color Histogram
#   Neural Detection        : YOLOv8 (optional)
#   Tuning                  : Grid Search
#
# Run this script after uploading your videos to Google Drive.
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
import numpy as np
from google.colab import drive
from ultralytics import YOLO

# ── Suppress minor OpenCV / numpy warnings ────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Google Drive mount & project root ─────────────────────────────────────────
drive.mount('/content/drive')
DRIVE_BASE = "/content/drive/MyDrive/ML_Tracker"

# Override the default DRIVE_BASE in each support module
import support.data_pipeline  as dp
import support.experiment      as exp
import support.visualization   as viz

dp.DRIVE_BASE  = DRIVE_BASE
exp.DRIVE_BASE = DRIVE_BASE
viz.DRIVE_BASE = DRIVE_BASE

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 — Directory setup
# ─────────────────────────────────────────────────────────────────────────────
dp.setup_directories(DRIVE_BASE)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Load YOLO model
# ─────────────────────────────────────────────────────────────────────────────
print("⏳ Loading YOLOv8n model...")
yolo_model = YOLO("yolov8n.pt")   # downloads automatically on first run
print("✅ YOLOv8n loaded.")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Scan uploaded videos (place files in dataset/uploaded/ first)
# ─────────────────────────────────────────────────────────────────────────────
dp.scan_uploaded_videos(DRIVE_BASE)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Assign videos to train / val / test splits
#  Edit dp.VIDEO_SPLITS to change assignments.
# ─────────────────────────────────────────────────────────────────────────────
SPLIT_VIDEOS = dp.assign_splits(
    video_splits        = dp.VIDEO_SPLITS,
    video_moving_camera = dp.VIDEO_MOVING_CAMERA,
    drive_base          = DRIVE_BASE,
)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4c — Auto-generate YOLO ground-truth annotations
#  Edit dp.VIDEO_CLASS_FILTER to control which COCO classes are kept per video.
# ─────────────────────────────────────────────────────────────────────────────
dp.run_yolo_annotation_pipeline(
    split_videos        = SPLIT_VIDEOS,
    yolo_model          = yolo_model,
    video_class_filter  = dp.VIDEO_CLASS_FILTER,
    drive_base          = DRIVE_BASE,
)

