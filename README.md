## Overview

This tracker implements a complete machine learning pipeline with proper train/validation/test splits and automated hyperparameter tuning. The core tracking components use traditional computer vision methods rather than deep learning architectures.

**Pipeline**: Video Upload → Annotation → Dataset Split → Grid Search Training → Validation → Test Evaluation → Results Export

## Core Algorithms

- **Background Subtraction**: MOG2, GMG, or YOLO for foreground detection
- **Motion Compensation**: Lucas-Kanade optical flow for camera stabilization
- **State Estimation**: 6-state Kalman filter (position, velocity)
- **Data Association**: Hungarian algorithm for optimal assignment
- **Re-Identification**: HSV color histograms for track recovery

## Requirements

- Python 3.8 or higher
- Google Colab or Jupyter environment
- Google Drive for persistent storage

## Installation

```bash
pip install filterpy scipy opencv-python-headless ipywidgets ultralytics lap
```

## Usage

### Basic Workflow

1. Mount Google Drive and create directory structure (Cell 0)
2. Install dependencies (Cell 1)
3. Initialize libraries and core classes (Cells 2-3)
4. Upload videos to `dataset/uploaded/`
5. Assign videos to train/val/test splits (Cell 4b)
6. Generate ground truth annotations (Cell 4c or Cell 5)
7. Run grid search on training data (Cell 7)
8. Evaluate on validation set and select best configuration (Cell 8)
9. Test on held-out data (Cell 9)
10. View dashboard and download results (Cells 10-12)

### Directory Structure

```
ML_Tracker/
├── dataset/
│   ├── train/videos/
│   ├── train/annotations/
│   ├── val/videos/
│   ├── val/annotations/
│   ├── test/videos/
│   ├── test/annotations/
│   └── uploaded/
└── outputs/
    ├── tracked/
    └── reports/
```

## Configuration

The grid search explores the following hyperparameter space:

```python
{
    "method": ["mog2", "gmg", "yolo"],
    "max_disappeared": [10, 20, 30],
    "max_distance": [20, 35, 50],
    "hist_bins": [8, 12, 16],
    "camera_stabilization": [True, False]
}
```

Modify `DRIVE_BASE` in Cell 0 to change project location:

```python
DRIVE_BASE = "/content/drive/MyDrive/ML_Tracker"
```

## Evaluation Metrics

**Detection Metrics**:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: Harmonic mean of precision and recall

**Tracking Metrics**:
- MOTA: Multi-Object Tracking Accuracy
- MAE: Mean Absolute Error in pixels
- ID Switches: Track identity changes

## Output

The system generates:
- Annotated videos with bounding boxes and track IDs
- Performance dashboard (7-panel visualization)
- JSON experiment report with complete results

## Technical Implementation

### Core Components

| Component | Implementation | Function |
|-----------|---------------|----------|
| MotionCompensator | Lucas-Kanade Optical Flow | Camera stabilization |
| BackgroundSubtractor | MOG2/GMG/LSBP | Foreground detection |
| KalmanTracker | 6-state filter | Position and velocity smoothing |
| MultiObjectTracker | Hungarian + HSV | Track management and assignment |

### Processing Pipeline

```
Input Frame
    → Motion Compensation
    → Background Subtraction
    → Blob Detection
    → Kalman Prediction
    → Hungarian Assignment
    → HSV Re-Identification
    → Tracked Output
```


## Dependencies

- OpenCV: Computer vision operations
- FilterPy: Kalman filtering
- SciPy: Hungarian algorithm
- Ultralytics: YOLOv8 detection
- LAP: Linear assignment optimization
