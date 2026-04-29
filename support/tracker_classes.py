# ── support/tracker_classes.py ────────────────────────────────────────────────
# Core classical ML components:
#   MotionCompensator    — Lucas-Kanade camera stabilisation
#   BackgroundSubtractor — MOG2 / GMG + morphological cleanup
#   YOLODetector         — YOLOv8 inference, same output format as BGSub
#   KalmanTracker        — 6-state constant-velocity Kalman filter
#   MultiObjectTracker   — Hungarian assignment + HSV Re-ID
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from collections import OrderedDict, deque
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# ═══════════════════════════════════════════════════════════════════════════════
#  A. CAMERA MOTION COMPENSATOR
# ═══════════════════════════════════════════════════════════════════════════════
class MotionCompensator:
    def __init__(self, enabled: bool = True):
        self.enabled   = enabled
        self.prev_gray = None
        self.lk_params = dict(
            winSize  = (21, 21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        self.feat_params = dict(
            maxCorners   = 200,
            qualityLevel = 0.01,
            minDistance  = 10,
            blockSize    = 5,
        )

    def compensate(self, frame: np.ndarray):
        if not self.enabled:
            return frame, np.eye(2, 3, dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H    = np.eye(2, 3, dtype=np.float32)

        if self.prev_gray is not None:
            pts0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feat_params)
            if pts0 is not None and len(pts0) >= 6:
                pts1, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, pts0, None, **self.lk_params
                )
                good0 = pts0[status.ravel() == 1]
                good1 = pts1[status.ravel() == 1]
                if len(good0) >= 6:
                    H, _ = cv2.estimateAffinePartial2D(
                        good0, good1, method=cv2.RANSAC, ransacReprojThreshold=3
                    )
                    if H is None:
                        H = np.eye(2, 3, dtype=np.float32)
                    else:
                        frame = cv2.warpAffine(
                            frame, H, (frame.shape[1], frame.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE,
                        )

        self.prev_gray = gray
        return frame, H


# ═══════════════════════════════════════════════════════════════════════════════
#  B. BACKGROUND SUBTRACTOR  (MOG2 / GMG)
# ═══════════════════════════════════════════════════════════════════════════════
class BackgroundSubtractor:
    def __init__(
        self,
        method: str        = "MOG2",
        var_threshold: int = 40,
        min_area: int      = 500,
        morph_kernel: int  = 5,
    ):
        self.min_area = min_area
        self.kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )

        if method == "MOG2":
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=var_threshold, detectShadows=True
            )
        elif method == "GMG":
            self.bg = cv2.bgsegm.createBackgroundSubtractorGMG(
                initializationFrames=60
            )
        else:
            raise ValueError(
                f"Unknown background method: '{method}'. "
                "Choose from: 'MOG2', 'GMG'."
            )

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        mask = self.bg.apply(frame)
        mask[mask == 127] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        return mask

    def detect(self, mask: np.ndarray) -> list:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                "bbox":     (x, y, w, h),
                "centroid": (x + w // 2, y + h // 2),
                "area":     int(area),
            })
        return detections


# ═══════════════════════════════════════════════════════════════════════════════
#  C. YOLO DETECTOR
#     Wraps YOLOv8 inference into the same dict format as BackgroundSubtractor
#     so the rest of the pipeline (Kalman, Hungarian, Re-ID) is unchanged.
# ═══════════════════════════════════════════════════════════════════════════════
class YOLODetector:
    def __init__(
        self,
        model,
        conf_threshold: float = 0.4,
        min_area: int         = 500,
        allowed_classes: list = None,
    ):
        self.model           = model
        self.conf_threshold  = conf_threshold
        self.min_area        = min_area
        self.allowed_classes = allowed_classes  # None = all classes

    def detect(self, frame: np.ndarray) -> list:
        results    = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        detections = []
        for box in results.boxes:
            if self.allowed_classes is not None:
                cls_id = int(box.cls[0])
                if cls_id not in self.allowed_classes:
                    continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y, w, h      = x1, y1, x2 - x1, y2 - y1
            area             = w * h

            if area < self.min_area:
                continue

            detections.append({
                "bbox":     (x, y, w, h),
                "centroid": (x + w // 2, y + h // 2),
                "area":     area,
            })
        return detections


# ═══════════════════════════════════════════════════════════════════════════════
#  D. KALMAN TRACKER (per object)
# ═══════════════════════════════════════════════════════════════════════════════
class KalmanTracker:
    _id_counter: int = 0

    def __init__(self, bbox: tuple, kf_r: float = 10.0, kf_q: float = 1.0):
        KalmanTracker._id_counter += 1
        self.id         = KalmanTracker._id_counter
        self.age        = 0
        self.hits       = 0
        self.no_det     = 0
        self.trajectory = deque(maxlen=80)
        self.color      = None
        self.hist       = None

        kf = KalmanFilter(dim_x=6, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=float)
        kf.P     *= 500
        kf.R     *= kf_r
        kf.Q[4:, 4:] *= kf_q

        x, y, w, h = bbox
        kf.x = np.array([[x], [y], [w], [h], [0.0], [0.0]])
        self.kf = kf

    def predict(self) -> tuple:
        self.kf.predict()
        self.age += 1
        px, py = int(self.kf.x[0, 0]), int(self.kf.x[1, 0])
        self.trajectory.append((px, py))
        return px, py

    def update(self, bbox: tuple) -> None:
        x, y, w, h = bbox
        self.kf.update(np.array([[x], [y], [w], [h]]))
        self.hits  += 1
        self.no_det = 0

    def get_state(self) -> tuple:
        x, y, w, h = [int(self.kf.x[i, 0]) for i in range(4)]
        vx = float(self.kf.x[4, 0])
        vy = float(self.kf.x[5, 0])
        speed = float(np.hypot(vx, vy))
        return x, y, w, h, vx, vy, speed


# ═══════════════════════════════════════════════════════════════════════════════
#  E. COLOR HISTOGRAM RE-IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
def compute_hist(frame: np.ndarray, bbox: tuple):
    x, y, w, h = bbox
    x, y = max(0, x), max(0, y)
    roi  = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return None
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def hist_similarity(h1, h2) -> float:
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(
        h1.reshape(-1, 1).astype(np.float32),
        h2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL,
    ))


# ═══════════════════════════════════════════════════════════════════════════════
#  F. MULTI-OBJECT TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
PALETTE = [
    (255, 80,  80),  (80,  255, 80),  (80,  80,  255), (255, 220, 0),
    (255, 80,  255), (0,   220, 255), (255, 140, 0),   (140, 0,   255),
    (0,   255, 160), (200, 200, 200), (255, 160, 80),  (80,  255, 200),
]


class MultiObjectTracker:
    def __init__(
        self,
        max_disappeared: int   = 25,
        max_distance: float    = 120.0,
        reid_threshold: float  = 0.60,
        kf_r: float            = 10.0,
        kf_q: float            = 1.0,
    ):
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance
        self.reid_threshold  = reid_threshold
        self.kf_r            = kf_r
        self.kf_q            = kf_q
        self.trackers        = OrderedDict()
        self.dormant         = []
        self.id_counter      = 0

    def _register(self, det: dict, frame: np.ndarray) -> KalmanTracker:
        kt        = KalmanTracker(det["bbox"], self.kf_r, self.kf_q)
        kt.color  = PALETTE[self.id_counter % len(PALETTE)]
        kt.hist   = compute_hist(frame, det["bbox"])
        self.trackers[kt.id] = kt
        self.id_counter     += 1
        return kt

    def update(self, detections: list, frame: np.ndarray) -> OrderedDict:
        for kt in list(self.trackers.values()):
            kt.predict()

        if not detections:
            for oid, kt in list(self.trackers.items()):
                kt.no_det += 1
                if kt.no_det > self.max_disappeared:
                    self.dormant.append(kt)
                    del self.trackers[oid]
            return self.trackers

        if not self.trackers:
            for det in detections:
                self._register(det, frame)
            return self.trackers

        track_ids  = list(self.trackers.keys())
        track_kts  = [self.trackers[i] for i in track_ids]
        det_ctrs   = [d["centroid"] for d in detections]
        pred_ctrs  = [(int(kt.kf.x[0, 0]), int(kt.kf.x[1, 0])) for kt in track_kts]

        spatial    = dist.cdist(np.array(pred_ctrs), np.array(det_ctrs))
        appearance = np.ones_like(spatial)
        for ti, kt in enumerate(track_kts):
            if kt.hist is None:
                continue
            for di, det in enumerate(detections):
                h2 = compute_hist(frame, det["bbox"])
                appearance[ti, di] = 1.0 - hist_similarity(kt.hist, h2)

        sp_norm = spatial / (self.max_distance + 1e-6)
        cost    = 0.70 * sp_norm + 0.30 * appearance

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_t, matched_d = set(), set()

        for r, c in zip(row_ind, col_ind):
            if spatial[r, c] > self.max_distance:
                continue
            oid = track_ids[r]
            self.trackers[oid].update(detections[c]["bbox"])
            self.trackers[oid].hist = compute_hist(frame, detections[c]["bbox"])
            matched_t.add(r)
            matched_d.add(c)

        for r, oid in enumerate(track_ids):
            if r not in matched_t:
                self.trackers[oid].no_det += 1
                if self.trackers[oid].no_det > self.max_disappeared:
                    self.dormant.append(self.trackers[oid])
                    del self.trackers[oid]

        for c, det in enumerate(detections):
            if c in matched_d:
                continue
            h_new   = compute_hist(frame, det["bbox"])
            best_sim, best_kt = -1.0, None
            for dkt in self.dormant:
                sim = hist_similarity(dkt.hist, h_new)
                if sim > best_sim:
                    best_sim, best_kt = sim, dkt

            if best_sim >= self.reid_threshold and best_kt is not None:
                best_kt.update(det["bbox"])
                best_kt.no_det = 0
                best_kt.hist   = h_new
                self.trackers[best_kt.id] = best_kt
                self.dormant.remove(best_kt)
            else:
                self._register(det, frame)

        self.dormant = self.dormant[-30:]
        return self.trackers
