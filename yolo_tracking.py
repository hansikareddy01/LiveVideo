"""
yolo_tracking.py  -  Stage 2: YOLOv8n-pose + ByteTrack

Takes frames from the frame grabber, finds people, gives each a stable
track ID, and computes simple body features for the rule engine.
It does NOT decide what an event is (that is rule_engine.py).

Run a quick self-test on one clip:
    python yolo_tracking.py --video clip1.mp4 --camera cam1
"""
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import cv2
import numpy as np

# ---- tunables (tune on some clips, test on the others) ----
KP_CONF = 0.3          # ignore keypoints below this confidence
DARK_BRIGHTNESS = 60   # mean gray value below this = "dark scene"
CONF_NORMAL = 0.25
CONF_DARK = 0.15
IMGSZ = 640

# COCO keypoint indices
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12

# ByteTrack tuned for ~4 fps input (defaults assume ~30 fps).
# match_thresh is the max allowed (1 - IoU) for a match, so RAISING it
# lets a person who moved further between frames still keep their ID.
BYTETRACK_4FPS = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 15
match_thresh: 0.9
fuse_score: True
"""


@dataclass
class PersonDet:
    track_id: int
    box: List[float]                 # x1, y1, x2, y2
    conf: float
    keypoints: List[List[float]]     # 17 x [x, y, conf]
    torso_angle: Optional[float]     # deg from vertical; ~90 = lying
    aspect: float                    # box width / height; >1 = wide & low
    kp_speed: float                  # keypoint movement / box height


@dataclass
class FrameResult:
    camera_id: str
    wall_time: str
    stream_time: float
    person_count: int
    persons: List[PersonDet]
    brightness: float
    conf_used: float
    infer_ms: float


def _write_tracker_cfg(path="bytetrack_4fps.yaml"):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(BYTETRACK_4FPS)
    return path


def _torso_angle(kp: np.ndarray) -> Optional[float]:
    """Angle of shoulder-mid -> hip-mid line vs vertical (0 = upright, 90 = lying)."""
    sho = [kp[i] for i in (L_SHO, R_SHO) if kp[i][2] >= KP_CONF]
    hip = [kp[i] for i in (L_HIP, R_HIP) if kp[i][2] >= KP_CONF]
    if not sho or not hip:
        return None
    s = np.mean([p[:2] for p in sho], axis=0)
    h = np.mean([p[:2] for p in hip], axis=0)
    dx, dy = abs(h[0] - s[0]), abs(h[1] - s[1])
    return float(np.degrees(np.arctan2(dx, dy)))


class YoloTracker:
    """One instance per camera (keeps its own model + tracker state)."""

    def __init__(self, camera_id, model_path="yolov8n-pose.pt",
                 threads=4, use_clahe=True):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(threads)   # fix once so timings are comparable
        cv2.setNumThreads(1)

        self.camera_id = camera_id
        self.model = YOLO(model_path)
        self.tracker_cfg = _write_tracker_cfg()
        self.use_clahe = use_clahe
        self.prev_kps = {}               # track_id -> keypoints (17x3)
        self.infer_ms: List[float] = []
        self.person_counts: List[int] = []
        dummy = np.zeros((IMGSZ, IMGSZ, 3), np.uint8)
        self.model.track(dummy, persist=True, tracker=self.tracker_cfg,
                         imgsz=IMGSZ, conf=CONF_NORMAL, classes=[0], verbose=False)

    # ---- call between videos ----
    def reset(self):
        """Reset tracker state (IDs restart). Timing stats are kept."""
        self.model.predictor = None
        self.prev_kps = {}

    def reset_stats(self):
        self.infer_ms, self.person_counts = [], []

    # ---- main entry ----
    def process(self, frame, wall_time, stream_time) -> FrameResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        dark = brightness < DARK_BRIGHTNESS
        conf = CONF_DARK if dark else CONF_NORMAL

        inp = frame
        if dark and self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            inp = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        t0 = time.perf_counter()
        res = self.model.track(
            inp, persist=True, tracker=self.tracker_cfg,
            imgsz=IMGSZ, conf=conf, classes=[0], verbose=False,
        )[0]
        ms = (time.perf_counter() - t0) * 1000.0

        persons = self._extract(res)
        self.infer_ms.append(ms)
        self.person_counts.append(len(persons))

        return FrameResult(
            camera_id=self.camera_id, wall_time=wall_time,
            stream_time=round(stream_time, 2),
            person_count=len(persons), persons=persons,
            brightness=round(brightness, 1), conf_used=conf,
            infer_ms=round(ms, 1),
        )

    def _extract(self, res) -> List[PersonDet]:
        out: List[PersonDet] = []
        if res.boxes is None or len(res.boxes) == 0:
            self.prev_kps = {}
            return out

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = (res.boxes.id.cpu().numpy().astype(int)
               if res.boxes.id is not None else np.full(len(boxes), -1))
        kps = (res.keypoints.data.cpu().numpy()
               if res.keypoints is not None else np.zeros((len(boxes), 17, 3)))

        new_prev = {}
        for box, c, tid, kp in zip(boxes, confs, ids, kps):
            w, h = box[2] - box[0], box[3] - box[1]
            aspect = float(w / h) if h > 0 else 0.0

            speed = 0.0
            if tid in self.prev_kps and h > 0:
                prev = self.prev_kps[tid]
                ok = (kp[:, 2] >= KP_CONF) & (prev[:, 2] >= KP_CONF)
                if ok.any():
                    d = np.linalg.norm(kp[ok, :2] - prev[ok, :2], axis=1)
                    speed = float(d.mean() / h)
            if tid != -1:
                new_prev[tid] = kp

            out.append(PersonDet(
                track_id=int(tid),
                box=[round(float(v), 1) for v in box],
                conf=round(float(c), 2),
                keypoints=[[round(float(v), 1) for v in p] for p in kp],
                torso_angle=(round(a, 1) if (a := _torso_angle(kp)) is not None else None),
                aspect=round(aspect, 2),
                kp_speed=round(speed, 3),
            ))
        self.prev_kps = new_prev
        return out

    # ---- hand-over numbers ----
    def stats(self) -> dict:
        if not self.infer_ms:
            return {}
        counts = np.array(self.person_counts)
        return {
            "frames": len(self.infer_ms),
            "infer_ms_p50": round(float(np.percentile(self.infer_ms, 50)), 1),
            "infer_ms_p95": round(float(np.percentile(self.infer_ms, 95)), 1),
            "avg_person_count": round(float(counts.mean()), 2),
            "pct_frames_zero_detections": round(float((counts == 0).mean() * 100), 1),
        }


# ---------------- self-test on one clip ----------------
def _summarize(r: FrameResult) -> str:
    bits = []
    for p in r.persons:
        ang = "n/a" if p.torso_angle is None else f"{p.torso_angle:.0f}"
        bits.append(f"id{p.track_id}(ang={ang},asp={p.aspect},spd={p.kp_speed})")
    return (f"t={r.stream_time:6.2f}s  people={r.person_count}  "
            f"bright={r.brightness:5.1f}  conf={r.conf_used}  "
            f"{r.infer_ms:6.1f} ms  " + " ".join(bits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--json", default=None, help="optional path to dump results as JSON")
    args = ap.parse_args()

    from datetime import datetime
    tracker = YoloTracker(args.camera)
    tracker.reset()

    cap = cv2.VideoCapture(args.video)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / args.fps))
    start = datetime.now()
    results, idx = [], 0

    while True:
        if not cap.grab():            # skip decoding frames we drop
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                st = idx / native
                wall = (start.timestamp() + st)
                wall_iso = datetime.fromtimestamp(wall).isoformat(timespec="seconds")
                r = tracker.process(frame, wall_iso, st)
                results.append(r)
                print(_summarize(r))
        idx += 1
    cap.release()

    print("\n--- STAGE 2 SUMMARY ---")
    for k, v in tracker.stats().items():
        print(f"{k}: {v}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f)


if __name__ == "__main__":
    main()
