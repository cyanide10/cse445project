# ── support/data_pipeline.py ──────────────────────────────────────────────────
# Dataset management:
#   - Directory setup
#   - Video scanning & split assignment
#   - Annotation helpers (load/save CSV)
#   - Interactive frame labeler (Cell 5b)
#   - Auto YOLO ground-truth generation (Cell 4c)
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import csv
import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ── Config (set DRIVE_BASE before importing this module) ──────────────────────
DRIVE_BASE = "/content/drive/MyDrive/ML_Tracker"

SUPPORTED_EXT = {'.mp4', '.avi', '.mov', '.mkv'}

# ── Moving-camera flags per video ────────────────────────────────────────────
VIDEO_MOVING_CAMERA = {
    "DVD logo.mp4"               : False,
    "Traffic IP Camera video.mp4": False,
    "Aim Lab1.mp4"               : False,
    "Aim Lab2.mp4"               : False,
    "Top View Pedestrian.mp4"    : False,
    "Store cam.mp4"              : False,
    "Golden Retriever.mp4"       : True,
    "football_juggling.mp4"      : True,
    "puppy.mp4"                  : True,
    "solo_dance_nsfw.mp4"        : False,
}

# ── Split assignment ──────────────────────────────────────────────────────────
VIDEO_SPLITS = {
    "DVD logo.mp4"               : "train",
    "Traffic IP Camera video.mp4": "train",
    "Aim Lab1.mp4"               : "train",
    "Top View Pedestrian.mp4"    : "train",
    "Golden Retriever.mp4"       : "train",
    "Store cam.mp4"              : "train",
    "Aim Lab2.mp4"               : "val",
    "football_juggling.mp4"      : "val",
    "puppy.mp4"                  : "test",
    "solo_dance_nsfw.mp4"        : "test",
}

# ── YOLO class filter per video for ground-truth generation ───────────────────
# COCO class IDs: 0=person, 2=car, 15=cat, 16=dog, etc.
# None = keep all detected classes
VIDEO_CLASS_FILTER = {
    "puppy.mp4"           : [16],   # dog only
    "solo_dance_nsfw.mp4" : [0],    # person only
}

YOLO_GT_CONF     = 0.4
YOLO_GT_MIN_AREA = 500


# ═══════════════════════════════════════════════════════════════════════════════
#  Directory Setup
# ═══════════════════════════════════════════════════════════════════════════════
def setup_directories(drive_base: str = DRIVE_BASE) -> None:
    """Create the full project directory tree under drive_base."""
    dirs = [
        f"{drive_base}/dataset/train/videos",
        f"{drive_base}/dataset/train/annotations",
        f"{drive_base}/dataset/val/videos",
        f"{drive_base}/dataset/val/annotations",
        f"{drive_base}/dataset/test/videos",
        f"{drive_base}/dataset/test/annotations",
        f"{drive_base}/dataset/uploaded",
        f"{drive_base}/outputs/tracked",
        f"{drive_base}/outputs/reports",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Directory tree created under: {drive_base}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Video Scanning & Split Assignment
# ═══════════════════════════════════════════════════════════════════════════════
def scan_uploaded_videos(drive_base: str = DRIVE_BASE) -> list:
    """List all supported video files in the uploaded/ folder."""
    upload_dir     = f"{drive_base}/dataset/uploaded"
    all_files      = os.listdir(upload_dir)
    uploaded_names = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ]
    if not uploaded_names:
        print(f"⚠️  No video files found in: {upload_dir}")
        print("   Supported formats: .mp4  .avi  .mov  .mkv")
    else:
        print(f"✅ Found {len(uploaded_names)} video(s):")
        for fname in uploaded_names:
            fpath   = os.path.join(upload_dir, fname)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"   ✓  {fname}  ({size_mb:.2f} MB)")
    return uploaded_names


def assign_splits(
    video_splits: dict = VIDEO_SPLITS,
    video_moving_camera: dict = VIDEO_MOVING_CAMERA,
    drive_base: str = DRIVE_BASE,
) -> dict:
    """Copy videos from uploaded/ into split folders. Returns SPLIT_VIDEOS dict."""
    split_videos = {"train": [], "val": [], "test": []}

    for fname, split in video_splits.items():
        if split not in split_videos:
            print(f"   ⚠️  Invalid split '{split}' for {fname}")
            continue
        src  = f"{drive_base}/dataset/uploaded/{fname}"
        dest = f"{drive_base}/dataset/{split}/videos/{fname}"
        if os.path.exists(src):
            shutil.copy(src, dest)
            split_videos[split].append(fname)
            cam_flag = "📷 moving" if video_moving_camera.get(fname, False) else "🎥 static"
            print(f"   [{split:5s}]  {fname}  ({cam_flag})")
        else:
            print(f"   ⚠️  File not found: {src}")

    print()
    for split, vids in split_videos.items():
        status = "✅" if vids else "⚠️ "
        print(f"   {status} {split:5s}: {len(vids)} video(s)  {vids}")

    moving_count = sum(1 for v in video_moving_camera.values() if v)
    print(f"\n   📷 Moving-camera videos : {moving_count}")
    print(f"   🎥 Static-camera videos : {len(video_moving_camera) - moving_count}")

    if not split_videos["train"]:
        print("\n⚠️  WARNING: No training videos — grid search will be skipped.")
    if not split_videos["val"]:
        print("⚠️  WARNING: No validation videos — best config will default to best training config.")
    if not split_videos["test"]:
        print("⚠️  WARNING: No test videos — final evaluation will be skipped.")

    return split_videos


def print_video_inventory(split_videos: dict, drive_base: str = DRIVE_BASE) -> None:
    """Print resolution / frame-count info for every assigned video."""
    print("📹 Video inventory across all splits:\n")
    found_any = False
    for split, vids in split_videos.items():
        for vname in vids:
            vpath = f"{drive_base}/dataset/{split}/videos/{vname}"
            if not os.path.exists(vpath):
                print(f"  ⚠️  [{split}]  {vname}  — file not found")
                continue
            cap = cv2.VideoCapture(vpath)
            W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            F   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            FPS = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"  [{split:5s}]  {vname}: {W}x{H}, {F} frames @ {FPS:.1f} fps")
            found_any = True

    if not found_any:
        print("  (No videos found — run scan_uploaded_videos() and assign_splits() first)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Annotation Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_annotations_csv(csv_path: str) -> dict:
    """Load a ground-truth CSV → {frame_id: [{"object_id", "bbox", "centroid"}, ...]}"""
    annotations = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
            obj = {
                "object_id": int(row["object_id"]),
                "bbox":      (x, y, w, h),
                "centroid":  (x + w // 2, y + h // 2),
            }
            annotations.setdefault(fid, []).append(obj)
    return annotations


def save_annotations_csv(annotations: dict, csv_path: str) -> None:
    """Save annotations dict to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "object_id", "x", "y", "w", "h"])
        for fid in sorted(annotations):
            for obj in annotations[fid]:
                x, y, w, h = obj["bbox"]
                writer.writerow([fid, obj["object_id"], x, y, w, h])


def clean_name(filename: str) -> str:
    """Strip extension and any trailing (N) suffix, lowercase for comparison."""
    base = os.path.splitext(filename)[0]
    return re.sub(r'\s*\(\d+\)\s*$', '', base).strip().lower()


def load_gt_for_video(
    split: str,
    video_name: str,
    drive_base: str = DRIVE_BASE,
) -> dict:
    """Load ground-truth annotations for a given split/video. Returns {} if not found."""
    base    = os.path.splitext(video_name)[0]
    csvpath = f"{drive_base}/dataset/{split}/annotations/{base}.csv"
    if not os.path.exists(csvpath):
        clean = re.sub(r'\s*\(\d+\)\s*$', '', base).strip()
        csvpath = f"{drive_base}/dataset/{split}/annotations/{clean}.csv"
        if not os.path.exists(csvpath):
            return {}
    return load_annotations_csv(csvpath)


def copy_annotation_csvs(split_videos: dict, drive_base: str = DRIVE_BASE) -> None:
    """
    Cell 5a: Copy manually prepared annotation CSVs from uploaded/ into the
    correct split annotation folder, matched by video filename.
    """
    ann_dir   = f"{drive_base}/dataset/uploaded"
    csv_files = [f for f in os.listdir(ann_dir) if f.lower().endswith('.csv')]

    if not csv_files:
        print("   (No CSV files found in uploaded/ — continuing without ground-truth annotations)")
        return

    for fname in csv_files:
        csv_clean = clean_name(fname)
        placed    = False
        for split in ['train', 'val', 'test']:
            for vname in split_videos[split]:
                vid_clean = clean_name(vname)
                if vid_clean == csv_clean:
                    clean_base = re.sub(r'\s*\(\d+\)\s*$', '',
                                        os.path.splitext(vname)[0]).strip()
                    dest = f"{drive_base}/dataset/{split}/annotations/{clean_base}.csv"
                    shutil.copy(os.path.join(ann_dir, fname), dest)
                    print(f"   ✅  [{split}]  {fname}  →  {clean_base}.csv")
                    placed = True
        if not placed:
            print(f"   ⚠️   Could not match '{fname}'")
            print(f"        CSV clean name : '{csv_clean}'")
            print(f"        Video names    : "
                  f"{[clean_name(v) for vlist in split_videos.values() for v in vlist]}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 5b — Interactive Frame Labeler
# ═══════════════════════════════════════════════════════════════════════════════
def interactive_labeler(
    label_video: str,
    label_split: str = "train",
    label_every_n: int = 5,
    drive_base: str = DRIVE_BASE,
) -> None:
    """
    Draw bounding boxes frame-by-frame with click-drag. Annotations are saved
    to the split's annotations/ folder with linear interpolation between
    labelled keyframes.

    Set label_video = '' to skip.
    """
    if not label_video:
        print("label_video is empty — skipping interactive labeler.")
        return

    vpath    = f"{drive_base}/dataset/{label_split}/videos/{label_video}"
    ann_path = (
        f"{drive_base}/dataset/{label_split}/annotations/"
        f"{os.path.splitext(label_video)[0]}.csv"
    )

    if not os.path.exists(vpath):
        print(f"❌ Video not found: {vpath}")
        return

    cap     = cv2.VideoCapture(vpath)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_anns: dict = {}

    frame_indices = list(range(0, total_f, label_every_n))
    print(f"Labeling '{label_video}': {len(frame_indices)} keyframes")
    print("Instructions: click-drag to draw bounding boxes, then close the figure to advance.")

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        coords = {"x0": None, "y0": None, "drawing": False, "rects": []}
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(rgb)
        ax.set_title(
            f"Frame {fi}/{total_f - 1}  —  Click-drag to draw bounding boxes\n"
            "Close this figure when done with this frame.",
            fontsize=11
        )

        def on_press(event):
            if event.inaxes != ax:
                return
            coords["x0"], coords["y0"] = event.xdata, event.ydata
            coords["drawing"] = True

        def on_release(event):
            if not coords["drawing"] or event.inaxes != ax:
                return
            x0, y0 = coords["x0"], coords["y0"]
            x1, y1 = event.xdata, event.ydata
            x, y = int(min(x0, x1)), int(min(y0, y1))
            w, h = int(abs(x1 - x0)), int(abs(y1 - y0))
            if w > 5 and h > 5:
                coords["rects"].append((x, y, w, h))
                rect = plt.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor="lime", facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(x, y - 5, f"obj{len(coords['rects'])}",
                        color="lime", fontsize=9, fontweight="bold")
                fig.canvas.draw()
            coords["drawing"] = False

        fig.canvas.mpl_connect("button_press_event",   on_press)
        fig.canvas.mpl_connect("button_release_event", on_release)
        plt.tight_layout()
        plt.show()

        all_anns[fi] = [
            {
                "object_id": bid,
                "bbox":      (x, y, w, h),
                "centroid":  (x + w // 2, y + h // 2),
            }
            for bid, (x, y, w, h) in enumerate(coords["rects"], start=1)
        ]

    cap.release()

    # ── Linear interpolation between labelled keyframes ───────────────────────
    sorted_keys = sorted(all_anns.keys())
    for i in range(len(sorted_keys) - 1):
        f0, f1   = sorted_keys[i], sorted_keys[i + 1]
        objs0, objs1 = all_anns[f0], all_anns[f1]
        n_common = min(len(objs0), len(objs1))
        for fi in range(f0 + 1, f1):
            alpha = (fi - f0) / (f1 - f0)
            interp = []
            for j in range(n_common):
                b0 = np.array(objs0[j]["bbox"])
                b1 = np.array(objs1[j]["bbox"])
                bi = (b0 * (1 - alpha) + b1 * alpha).astype(int)
                cx, cy = bi[0] + bi[2] // 2, bi[1] + bi[3] // 2
                interp.append({
                    "object_id": j + 1,
                    "bbox":      tuple(bi.tolist()),
                    "centroid":  (cx, cy),
                })
            all_anns[fi] = interp

    save_annotations_csv(all_anns, ann_path)
    print(f"\n✅ Saved {len(all_anns)} annotated frames → {ann_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 4c — Auto-Generate YOLO Ground-Truth Annotations
# ═══════════════════════════════════════════════════════════════════════════════
def generate_yolo_annotations(
    video_path: str,
    yolo_model,
    allowed_classes: list = None,
    conf: float = YOLO_GT_CONF,
    min_area: int = YOLO_GT_MIN_AREA,
) -> dict:
    """
    Run YOLO frame-by-frame on a video.
    Returns: {frame_id: [{"object_id", "bbox", "centroid"}, ...]}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"      ❌ Could not open: {video_path}")
        return {}

    annotations = {}
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = yolo_model(frame, verbose=False, conf=conf)[0]
        frame_dets = []
        obj_id     = 1

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if allowed_classes is not None and cls_id not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y, w, h      = x1, y1, x2 - x1, y2 - y1
            area             = w * h

            if area < min_area:
                continue

            frame_dets.append({
                "object_id": obj_id,
                "bbox"     : (x, y, w, h),
                "centroid" : (x + w // 2, y + h // 2),
            })
            obj_id += 1

        if frame_dets:
            annotations[frame_idx] = frame_dets

        frame_idx += 1

    cap.release()
    return annotations


def run_yolo_annotation_pipeline(
    split_videos: dict,
    yolo_model,
    video_class_filter: dict = VIDEO_CLASS_FILTER,
    drive_base: str = DRIVE_BASE,
) -> None:
    """
    Cell 4c: Run YOLO ground-truth generation on all splits and save CSVs.
    Must be called before the grid search (Cell 7).
    """
    import re as _re

    print("=" * 65)
    print(f"{'  CELL 4c — YOLO Ground-Truth Annotation':^65}")
    print("=" * 65)
    print(f"   YOLO conf      : {YOLO_GT_CONF}")
    print(f"   Min area       : {YOLO_GT_MIN_AREA} px²")
    print()

    total_saved = 0

    for split in ["train", "val", "test"]:
        vids = split_videos.get(split, [])
        if not vids:
            print(f"   [{split:5s}]  (no videos)")
            continue

        for vname in vids:
            vpath      = f"{drive_base}/dataset/{split}/videos/{vname}"
            base       = _re.sub(r'\s*\(\d+\)\s*$', '', os.path.splitext(vname)[0]).strip()
            csv_path   = f"{drive_base}/dataset/{split}/annotations/{base}.csv"
            classes    = video_class_filter.get(vname, None)
            class_str  = str(classes) if classes is not None else "all classes"

            print(f"   [{split:5s}]  {vname}  (filter: {class_str})")

            if not os.path.exists(vpath):
                print(f"            ⚠️  Video not found — skipping")
                continue

            anns = generate_yolo_annotations(vpath, yolo_model, allowed_classes=classes)

            if not anns:
                print(f"            ⚠️  No detections — CSV not saved")
                continue

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_id", "object_id", "x", "y", "w", "h"])
                for fid in sorted(anns):
                    for obj in anns[fid]:
                        x, y, w, h = obj["bbox"]
                        writer.writerow([fid, obj["object_id"], x, y, w, h])

            n_frames = len(anns)
            n_boxes  = sum(len(v) for v in anns.values())
            print(f"            ✅ {n_frames} annotated frames, {n_boxes} total boxes → {csv_path}")
            total_saved += 1

    print()
    print(f"✅ Cell 4c complete — {total_saved} annotation CSV(s) saved.")
    print("   → Proceed to Cell 6 → Cell 7 for grid search.")
