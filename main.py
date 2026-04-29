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

