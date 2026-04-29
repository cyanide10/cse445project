# ── support/experiment.py ─────────────────────────────────────────────────────
# Full experiment pipeline:
#   run_tracker_on_video() — core tracking + evaluation engine  (Cell 6)
#   run_grid_search()      — hyperparameter grid search          (Cell 7)
#   run_validation()       — top-K config selection              (Cell 8)
#   run_test()             — held-out final evaluation           (Cell 9)
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import json
import math
import os
import time
import itertools
import numpy as np
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment

from support.tracker_classes import (
    MotionCompensator,
    BackgroundSubtractor,
    YOLODetector,
    KalmanTracker,
    MultiObjectTracker,
)
from support.data_pipeline import load_gt_for_video, VIDEO_MOVING_CAMERA

DRIVE_BASE = "/content/drive/MyDrive/ML_Tracker"

# ── Per-video warmup overrides ────────────────────────────────────────────────
VIDEO_WARMUP = {
    "DVD logo.mp4":                20,
    "Traffic IP Camera video.mp4": 20,
    "Aim Lab1.mp4":                20,
    "Aim Lab2.mp4":                20,
    "Top View Pedestrian.mp4":     20,
    "Store cam.mp4":               20,
    "Golden Retriever.mp4":        20,
    "football_juggling.mp4":       20,
    "puppy.mp4":                   20,
    "solo_dance_nsfw.mp4":         20,
}

# ── Hyperparameter search grids ───────────────────────────────────────────────
PARAM_GRID_MOG2 = {
    "bg_method":       ["MOG2"],
    "var_threshold":   [30, 50, 70],
    "min_area":        [200, 400],
    "morph_kernel":    [5],
    "max_disappeared": [20, 35],
    "max_distance":    [120, 180],
    "reid_threshold":  [0.55],
    "kf_r":            [10.0],
    "kf_q":            [1.0],
    "match_threshold": [50],
}

PARAM_GRID_GMG = {
    "bg_method":       ["GMG"],
    "var_threshold":   [40],
    "min_area":        [200, 400],
    "morph_kernel":    [5],
    "max_disappeared": [20, 35],
    "max_distance":    [120, 180],
    "reid_threshold":  [0.55],
    "kf_r":            [10.0],
    "kf_q":            [1.0],
    "match_threshold": [50],
}

PARAM_GRID_YOLO = {
    "bg_method":       ["YOLO"],
    "var_threshold":   [40],
    "yolo_conf":       [0.3, 0.5],
    "min_area":        [200, 400],
    "morph_kernel":    [5],
    "max_disappeared": [20, 35],
    "max_distance":    [120, 180],
    "reid_threshold":  [0.55],
    "kf_r":            [10.0],
    "kf_q":            [1.0],
    "match_threshold": [50],
}

DEFAULT_PARAMS = {
    "bg_method": "MOG2", "var_threshold": 40, "min_area": 200,
    "morph_kernel": 5, "max_disappeared": 20, "max_distance": 120,
    "reid_threshold": 0.55, "kf_r": 10.0, "kf_q": 1.0, "match_threshold": 50,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 6 — Core Tracking + Evaluation Engine
# ═══════════════════════════════════════════════════════════════════════════════
def run_tracker_on_video(
    video_path: str,
    params: dict,
    yolo_model=None,
    output_path: str   = None,
    ground_truth: dict = None,
    warmup: int        = None,
    video_moving_camera: dict = VIDEO_MOVING_CAMERA,
    video_class_filter: dict  = None,
) -> tuple:
    """
    Run the full tracking pipeline on a single video.

    Returns: (metrics_dict, tracking_log_list)
      metrics keys: precision, recall, f1, mae_px, mota, id_switches, tp, fp, fn, frames, detector
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open: {video_path}")
        return None, []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    vname     = os.path.basename(video_path)
    bg_method = params.get("bg_method", "MOG2")
    use_yolo  = bg_method == "YOLO"

    # ── Warmup ───────────────────────────────────────────────────────────────
    if warmup is None:
        if use_yolo:
            warmup = 0
        elif bg_method == "GMG":
            warmup = 65
        else:
            warmup = VIDEO_WARMUP.get(vname, 20)

    # ── Moving camera ─────────────────────────────────────────────────────────
    if vname in video_moving_camera:
        moving_camera = video_moving_camera[vname]
    else:
        moving_camera = params.get("moving_camera", False)

    compensator = MotionCompensator(enabled=moving_camera)

    # ── Detector ──────────────────────────────────────────────────────────────
    if use_yolo:
        _video_classes = None
        if video_class_filter is not None:
            _video_classes = video_class_filter.get(vname, params.get("allowed_classes", None))
        else:
            _video_classes = params.get("allowed_classes", None)

        detector = YOLODetector(
            model           = yolo_model,
            conf_threshold  = params.get("yolo_conf", 0.4),
            min_area        = params.get("min_area", 500),
            allowed_classes = _video_classes,
        )
    else:
        detector = BackgroundSubtractor(
            method        = bg_method,
            var_threshold = params.get("var_threshold", 40),
            min_area      = params.get("min_area", 500),
            morph_kernel  = params.get("morph_kernel", 5),
        )

    # ── Tracker ───────────────────────────────────────────────────────────────
    mot = MultiObjectTracker(
        max_disappeared = params.get("max_disappeared", 25),
        max_distance    = params.get("max_distance", 120),
        reid_threshold  = params.get("reid_threshold", 0.55),
        kf_r            = params.get("kf_r", 10.0),
        kf_q            = params.get("kf_q", 1.0),
    )
    KalmanTracker._id_counter = 0

    # ── Metrics bookkeeping ───────────────────────────────────────────────────
    total_tp    = 0
    total_fp    = 0
    total_fn    = 0
    all_errors  = []
    id_switches = 0
    prev_assign = {}
    tracking_log = []
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stable, _ = compensator.compensate(frame.copy())

        if use_yolo:
            detections = detector.detect(frame)
        else:
            mask = detector.get_mask(stable)
            if frame_idx >= warmup:
                detections = detector.detect(mask)
            else:
                detections = []

        trackers = mot.update(detections, frame)

        pred_bboxes = []
        for kt in trackers.values():
            x, y, w, h, vx, vy, speed = kt.get_state()
            pred_bboxes.append({
                "bbox":     (x, y, w, h),
                "centroid": (x + w // 2, y + h // 2),
                "id":       kt.id,
            })

        # ── Evaluate against GT ───────────────────────────────────────────────
        gt_objs = (ground_truth or {}).get(frame_idx, [])
        if gt_objs:
            gt_ctrs   = [o["centroid"] for o in gt_objs]
            pred_ctrs = [p["centroid"] for p in pred_bboxes]
            match_thresh = params.get("match_threshold", 50)

            if pred_ctrs and gt_ctrs:
                D_eval       = dist.cdist(np.array(pred_ctrs), np.array(gt_ctrs))
                row_i, col_i = linear_sum_assignment(D_eval)
                matched_p, matched_g = set(), set()

                for r, c in zip(row_i, col_i):
                    if D_eval[r, c] <= match_thresh:
                        total_tp += 1
                        all_errors.append(float(D_eval[r, c]))
                        matched_p.add(r)
                        matched_g.add(c)
                        gt_id   = gt_objs[c].get("object_id", c)
                        pred_id = pred_bboxes[r]["id"]
                        if gt_id in prev_assign and prev_assign[gt_id] != pred_id:
                            id_switches += 1
                        prev_assign[gt_id] = pred_id

                total_fp += len(pred_ctrs) - len(matched_p)
                total_fn += len(gt_ctrs)   - len(matched_g)
            else:
                total_fp += len(pred_ctrs)
                total_fn += len(gt_ctrs)

        # ── Write annotated output video ──────────────────────────────────────
        if writer:
            disp = frame.copy()
            for kt in trackers.values():
                x, y, w, h, vx, vy, speed = kt.get_state()
                color = kt.color
                cv2.rectangle(disp, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    disp, f"ID:{kt.id}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                )
            writer.write(disp)

        tracking_log.append({"frame": frame_idx, "n_tracks": len(trackers)})
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    # ── Final metrics ─────────────────────────────────────────────────────────
    prec = total_tp / (total_tp + total_fp + 1e-9)
    rec  = total_tp / (total_tp + total_fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    mae  = float(np.mean(all_errors)) if all_errors else float("nan")
    gt_n = total_tp + total_fn
    mota = max(0.0, 1.0 - (total_fn + total_fp + id_switches) / max(gt_n, 1))

    metrics = {
        "precision":   round(prec, 4),
        "recall":      round(rec,  4),
        "f1":          round(f1,   4),
        "mae_px":      round(mae,  2) if not math.isnan(mae) else None,
        "mota":        round(mota, 4),
        "id_switches": id_switches,
        "tp":          total_tp,
        "fp":          total_fp,
        "fn":          total_fn,
        "frames":      frame_idx,
        "detector":    bg_method,
    }

    return metrics, tracking_log


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 7 — Hyperparameter Grid Search
# ═══════════════════════════════════════════════════════════════════════════════
def make_combos(grid: dict) -> list:
    keys = list(grid.keys())
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


def run_grid_search(
    split_videos: dict,
    yolo_model=None,
    drive_base: str = DRIVE_BASE,
    video_class_filter: dict = None,
) -> tuple:
    """
    Run full hyperparameter grid search on training videos.
    Returns: (train_results, best_params, best_train_f1)
    """
    combos = (
        make_combos(PARAM_GRID_MOG2)
        + make_combos(PARAM_GRID_GMG)
        + make_combos(PARAM_GRID_YOLO)
    )

    print(f"🔍 Grid search space: {len(combos):,} parameter combinations")
    print(f"   MOG2 combos : {len(make_combos(PARAM_GRID_MOG2))}")
    print(f"   GMG  combos : {len(make_combos(PARAM_GRID_GMG))}  (var_threshold fixed)")
    print(f"   YOLO combos : {len(make_combos(PARAM_GRID_YOLO))}  (yolo_conf searched instead)")
    print(f"📹 Training on : {len(split_videos.get('train', []))} video(s)")

    train_results = []
    best_train_f1 = -1.0
    best_params   = None

    if not split_videos.get("train"):
        print("⚠️  Skipping grid search — no training videos available.")
        return train_results, DEFAULT_PARAMS.copy(), float("nan")

    est_sec = len(combos) * len(split_videos["train"]) * 3
    print(f"⏱️  Estimated runtime: ~{est_sec // 60}–{est_sec // 30} minutes")
    print()
    print("=" * 65)
    print(f"{'  TRAINING — Hyperparameter Grid Search':^65}")
    print("=" * 65)

    t_start   = time.time()
    log_every = max(1, len(combos) // 10)

    for ci, params in enumerate(combos):
        print(
            f"🔄 Combo [{ci+1:4d}/{len(combos)}] | "
            f"bg={params['bg_method']:4s} | "
            f"var_thresh={params['var_threshold']:2} | "
            f"min_area={params['min_area']:4} | "
            f"max_dist={params['max_distance']:3} | "
            f"max_disap={params['max_disappeared']}",
            flush=True
        )

        video_f1s = []
        for vname in split_videos["train"]:
            vpath      = f"{drive_base}/dataset/train/videos/{vname}"
            gt         = load_gt_for_video("train", vname, drive_base)
            metrics, _ = run_tracker_on_video(
                vpath, params,
                yolo_model=yolo_model,
                ground_truth=gt,
                video_class_filter=video_class_filter,
            )
            if metrics:
                video_f1s.append(metrics["f1"])

        avg_f1 = float(np.mean(video_f1s)) if video_f1s else 0.0
        train_results.append({"params": params, "train_f1": round(avg_f1, 4)})

        if avg_f1 > best_train_f1:
            best_train_f1 = avg_f1
            best_params   = params.copy()
            print(f"   ⭐ New best F1: {best_train_f1:.4f}", flush=True)

        if (ci + 1) % log_every == 0 or ci == len(combos) - 1:
            elapsed = time.time() - t_start
            eta     = elapsed / (ci + 1) * (len(combos) - ci - 1)
            print(
                f"   📊 [{ci+1:4d}/{len(combos)}]  "
                f"best F1={best_train_f1:.4f}  "
                f"elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s",
                flush=True
            )

    print()
    print("─" * 65)
    print(f"✅ Grid search complete.  Best training F1: {best_train_f1:.4f}")
    print("\nBest hyperparameters:")
    for k, v in best_params.items():
        print(f"   {k:22s}: {v}")

    # ── Save to Drive ─────────────────────────────────────────────────────────
    if train_results:
        save_path = f"{drive_base}/outputs/train_results.json"
        with open(save_path, "w") as f:
            json.dump({
                "train_results": train_results,
                "best_params":   best_params,
                "best_train_f1": best_train_f1,
            }, f)
        print(f"✅ Results saved → {save_path}")

    return train_results, best_params, best_train_f1


def load_train_results(drive_base: str = DRIVE_BASE) -> tuple:
    """Load saved grid-search results from Drive. Returns (train_results, best_params, best_train_f1)."""
    path = f"{drive_base}/outputs/train_results.json"
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        print(f"✅ Loaded {len(saved['train_results'])} results from Drive.")
        return saved["train_results"], saved["best_params"], saved["best_train_f1"]
    print("⚠️  No saved results found — run run_grid_search() first.")
    return [], DEFAULT_PARAMS.copy(), float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 8 — Validation: Select Best Configuration
# ═══════════════════════════════════════════════════════════════════════════════
def run_validation(
    train_results: list,
    best_params: dict,
    best_train_f1: float,
    split_videos: dict,
    yolo_model=None,
    top_k: int = 5,
    drive_base: str = DRIVE_BASE,
    video_class_filter: dict = None,
) -> tuple:
    """
    Evaluate the top-K training configs on validation videos.
    Returns: (val_results, BEST_PARAMS, BEST_dict)
    """
    top_k = min(top_k, len(train_results))

    if not train_results:
        print("⚠️  No training results — run run_grid_search() first.")
        best = {
            "rank": 1, "val_f1": float("nan"), "val_mae": float("nan"),
            "val_mota": float("nan"), "val_idswitch": float("nan"),
            "params": best_params,
        }
        return [best], best_params, best

    top_k_configs = sorted(train_results, key=lambda x: x["train_f1"], reverse=True)[:top_k]
    val_results   = []

    print("=" * 65)
    print(f"{'  VALIDATION — Evaluating Top Configurations':^65}")
    print("=" * 65)

    if not split_videos.get("val"):
        print("⚠️  No validation videos — using best training config directly.")
        best = {
            "rank": 1, "params": best_params,
            "train_f1": best_train_f1,
            "val_f1": float("nan"), "val_mae": float("nan"),
            "val_mota": float("nan"), "val_idswitch": float("nan"),
        }
        return [best], best_params, best

    for rank, entry in enumerate(top_k_configs):
        params     = entry["params"]
        vid_scores = []

        print(
            f"🔄 Config [{rank+1}/{len(top_k_configs)}] | "
            f"bg={params['bg_method']:4s} | "
            f"min_area={params['min_area']:4} | "
            f"max_dist={params['max_distance']:3} | "
            f"max_disap={params['max_disappeared']}",
            flush=True
        )

        for vname in split_videos["val"]:
            vpath      = f"{drive_base}/dataset/val/videos/{vname}"
            gt         = load_gt_for_video("val", vname, drive_base)
            metrics, _ = run_tracker_on_video(
                vpath, params,
                yolo_model=yolo_model,
                ground_truth=gt,
                video_class_filter=video_class_filter,
            )
            if metrics:
                vid_scores.append(metrics)

        if not vid_scores:
            continue

        avg_f1   = float(np.mean([s["f1"]             for s in vid_scores]))
        avg_mae  = float(np.nanmean([s["mae_px"] or 0  for s in vid_scores]))
        avg_mota = float(np.mean([s["mota"]            for s in vid_scores]))
        avg_idsw = float(np.mean([s["id_switches"]     for s in vid_scores]))

        val_results.append({
            "rank":         rank + 1,
            "params":       params,
            "train_f1":     entry["train_f1"],
            "val_f1":       round(avg_f1,   4),
            "val_mae":      round(avg_mae,   2),
            "val_mota":     round(avg_mota,  4),
            "val_idswitch": round(avg_idsw,  2),
        })

        print(
            f"   ✅ Config #{rank+1}  "
            f"train_F1={entry['train_f1']:.4f}  "
            f"val_F1={avg_f1:.4f}  "
            f"MAE={avg_mae:.1f}px  "
            f"MOTA={avg_mota:.4f}  "
            f"IDsw={avg_idsw:.1f}",
            flush=True
        )

    if not val_results:
        print("⚠️  No validation results — falling back to best training config.")
        best = {
            "rank": 1, "params": best_params,
            "val_f1": float("nan"), "val_mae": float("nan"),
            "val_mota": float("nan"), "val_idswitch": float("nan"),
        }
        return [best], best_params, best

    best        = max(val_results, key=lambda x: (x["val_f1"], -x["val_mae"]))
    best_params = best["params"]

    print()
    print("─" * 65)
    print(f"✅ CHOSEN CONFIG  (val F1={best['val_f1']:.4f}, MAE={best['val_mae']:.1f}px)")
    print(f"   Detector: {best_params.get('bg_method', 'N/A')}")
    for k, v in best_params.items():
        print(f"   {k:22s}: {v}")

    return val_results, best_params, best


# ═══════════════════════════════════════════════════════════════════════════════
#  Cell 9 — Test: Final Evaluation on Held-Out Videos
# ═══════════════════════════════════════════════════════════════════════════════
def run_test(
    best_params: dict,
    split_videos: dict,
    yolo_model=None,
    drive_base: str = DRIVE_BASE,
    video_class_filter: dict = None,
) -> tuple:
    """
    Run the chosen config on test videos. Saves annotated output videos.
    Returns: (test_metrics_list, avg_dict)
    """
    print("=" * 65)
    print(f"{'  TEST — Final Evaluation (held-out videos)':^65}")
    print("=" * 65)

    test_metrics = []
    avg          = {}

    if not split_videos.get("test"):
        print("⚠️  No test videos found.")
        return test_metrics, avg

    for vname in split_videos["test"]:
        vpath  = f"{drive_base}/dataset/test/videos/{vname}"
        opath  = f"{drive_base}/outputs/tracked/tracked_{vname}"
        gt     = load_gt_for_video("test", vname, drive_base)

        print(f"\n▶  Processing: {vname}")
        metrics, log = run_tracker_on_video(
            vpath, best_params,
            yolo_model=yolo_model,
            output_path=opath,
            ground_truth=gt,
            video_class_filter=video_class_filter,
        )
        if not metrics:
            print(f"   ❌ Failed to process {vname}")
            continue

        metrics["video"] = vname
        test_metrics.append(metrics)

        print(f"   Frames processed : {metrics['frames']}")
        print(f"   Detector         : {metrics.get('detector', 'N/A')}")
        print(f"   Precision        : {metrics['precision']:.4f}")
        print(f"   Recall           : {metrics['recall']:.4f}")
        print(f"   F1 Score         : {metrics['f1']:.4f}")
        if metrics["mae_px"] is not None:
            print(f"   MAE (px)         : {metrics['mae_px']:.2f}")
        print(f"   MOTA             : {metrics['mota']:.4f}")
        print(f"   ID Switches      : {metrics['id_switches']}")
        print(f"   TP / FP / FN     : {metrics['tp']} / {metrics['fp']} / {metrics['fn']}")
        gt_note = "annotated video saved" if gt else "tracked video saved (no GT metrics)"
        print(f"   Output           : {opath}  [{gt_note}]")

    if test_metrics:
        scalar_keys = ["precision", "recall", "f1", "mota", "id_switches"]
        avg = {k: round(float(np.mean([m[k] for m in test_metrics])), 4)
               for k in scalar_keys}
        mae_vals    = [m["mae_px"] for m in test_metrics if m["mae_px"] is not None]
        avg["mae_px"] = round(float(np.mean(mae_vals)), 2) if mae_vals else None

        print()
        print("─" * 50)
        print(f"{'  AVERAGE TEST RESULTS':^50}")
        print("─" * 50)
        for k, v in avg.items():
            print(f"   {k:15s}: {v}")
        print("─" * 50)
    else:
        print("\n⚠️  No test results generated.")

    return test_metrics, avg
