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

# Optional: copy manually prepared annotation CSVs from uploaded/
# dp.copy_annotation_csvs(SPLIT_VIDEOS, DRIVE_BASE)

# Optional: interactive frame labeler
# dp.interactive_labeler("my_video.mp4", label_split="train", drive_base=DRIVE_BASE)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Print video inventory
# ─────────────────────────────────────────────────────────────────────────────
dp.print_video_inventory(SPLIT_VIDEOS, DRIVE_BASE)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Hyperparameter grid search (training)
# ─────────────────────────────────────────────────────────────────────────────
train_results, best_params, best_train_f1 = exp.run_grid_search(
    split_videos        = SPLIT_VIDEOS,
    yolo_model          = yolo_model,
    drive_base          = DRIVE_BASE,
    video_class_filter  = dp.VIDEO_CLASS_FILTER,
)

# To skip re-running and reload from Drive:
# train_results, best_params, best_train_f1 = exp.load_train_results(DRIVE_BASE)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — Validation: select best configuration
# ─────────────────────────────────────────────────────────────────────────────
val_results, BEST_PARAMS, BEST = exp.run_validation(
    train_results       = train_results,
    best_params         = best_params,
    best_train_f1       = best_train_f1,
    split_videos        = SPLIT_VIDEOS,
    yolo_model          = yolo_model,
    top_k               = 5,
    drive_base          = DRIVE_BASE,
    video_class_filter  = dp.VIDEO_CLASS_FILTER,
)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 — Test: final evaluation on held-out videos
# ─────────────────────────────────────────────────────────────────────────────
test_metrics, avg = exp.run_test(
    best_params         = BEST_PARAMS,
    split_videos        = SPLIT_VIDEOS,
    yolo_model          = yolo_model,
    drive_base          = DRIVE_BASE,
    video_class_filter  = dp.VIDEO_CLASS_FILTER,
)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 — Results dashboard
# ─────────────────────────────────────────────────────────────────────────────
combos = (
    exp.make_combos(exp.PARAM_GRID_MOG2)
    + exp.make_combos(exp.PARAM_GRID_GMG)
    + exp.make_combos(exp.PARAM_GRID_YOLO)
)

viz.plot_dashboard(
    train_results  = train_results,
    val_results    = val_results,
    test_metrics   = test_metrics,
    avg            = avg,
    best           = BEST,
    best_train_f1  = best_train_f1,
    drive_base     = DRIVE_BASE,
)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 10 — Summary report
# ─────────────────────────────────────────────────────────────────────────────
viz.print_summary(
    split_videos   = SPLIT_VIDEOS,
    combos         = combos,
    best_train_f1  = best_train_f1,
    best           = BEST,
    best_params    = BEST_PARAMS,
    avg            = avg,
    test_metrics   = test_metrics,
    random_seed    = RANDOM_SEED,
    drive_base     = DRIVE_BASE,
)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 11 — Confirm output locations on Drive
# ─────────────────────────────────────────────────────────────────────────────
viz.print_outputs(DRIVE_BASE)
