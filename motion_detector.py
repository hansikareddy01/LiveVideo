"""
motion_detector.py  -  cheap motion signals (about 1-3 ms per frame)

Covers what YOLO misses: dark scenes and people who have fallen.

Two things are measured on every kept frame:
  1. FRAME DIFFERENCE vs the previous kept frame -> motion_ratio.
     A person lying still gives ~0, so this is what detects "went still".
  2. MOG2 BACKGROUND SUBTRACTION with a very slow learning rate -> blob_ratio.
     A fallen person stays a foreground blob for a long time, which tells us
     "a big object is still there" (vs. a person who simply walked away).

Main signals for the rule engine / gate:
  spike             sudden jump in motion
  burst_then_still  big motion, then stillness >= stop_seconds, and a big
                    blob is still in the scene  -> "something large stopped"
  global_change     almost the whole frame changed (light switch, camera bump);
                    ignored so it does not trigger events

Self-test (no video needed):   python motion_detector.py --selftest
Run on a clip:                 python motion_detector.py --video clip1.mp4
"""
import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class MotionResult:
    stream_time: float
    motion_ratio: float              # fraction of pixels changed vs previous kept frame
    blob_ratio: float                # largest foreground blob area / frame area
    blob_box: Optional[List[int]]    # x1, y1, x2, y2 in ORIGINAL frame pixels
    spike: bool
    burst_then_still: bool
    still_for_s: float
    global_change: bool
    warming_up: bool
    ms: float


class MotionDetector:
    """One instance per camera. Call reset() between videos."""

    def __init__(self,
                 width=320,            # work at this width (speed); boxes are scaled back
                 diff_thresh=25,       # pixel change (0-255) that counts as motion
                 bg_history=120,
                 bg_var=25,
                 bg_lr=0.002,          # slow, so a still person stays a blob
                 warmup_frames=8,      # ignore events while the background is learned
                 spike_ratio=0.02,     # >=2% of pixels changing = big activity
                 spike_factor=3.0,     # ...and >=3x the recent baseline = a "spike"
                 still_ratio=0.004,    # <0.4% of pixels changing = still
                 min_blob_ratio=0.01,  # a "big" blob is >=1% of the frame
                 global_ratio=0.6,     # >=60% of pixels changed = lighting/camera event
                 stop_seconds=2.0,     # stillness needed after a burst
                 burst_expire_s=10.0):
        self.width = width
        self.diff_thresh = diff_thresh
        self.bg_history = bg_history
        self.bg_var = bg_var
        self.bg_lr = bg_lr
        self.warmup_frames = warmup_frames
        self.spike_ratio = spike_ratio
        self.spike_factor = spike_factor
        self.still_ratio = still_ratio
        self.min_blob_ratio = min_blob_ratio
        self.global_ratio = global_ratio
        self.stop_seconds = stop_seconds
        self.burst_expire_s = burst_expire_s

        self.k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.ms: List[float] = []
        self.n_spikes = 0
        self.reset()

    def reset(self):
        """Forget the background and all state (stats are kept)."""
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=self.bg_history, varThreshold=self.bg_var, detectShadows=True)
        self.prev = None
        self.n = 0
        self.base = 0.0
        self.burst_t = None
        self.still_since = None
        self.latched = False

    def reset_stats(self):
        self.ms, self.n_spikes = [], 0

    def process(self, frame, stream_time: float) -> MotionResult:
        t0 = time.perf_counter()
        h0, w0 = frame.shape[:2]
        scale = min(1.0, self.width / float(w0))
        small = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        warming = self.n < self.warmup_frames

        # 1) frame difference -> motion_ratio
        if self.prev is None:
            ratio = 0.0
        else:
            d = cv2.absdiff(gray, self.prev)
            _, dm = cv2.threshold(d, self.diff_thresh, 255, cv2.THRESH_BINARY)
            dm = cv2.morphologyEx(dm, cv2.MORPH_OPEN, self.k_small)
            ratio = cv2.countNonZero(dm) / float(dm.size)
        self.prev = gray

        # 2) slow background subtraction -> largest blob
        fg = self.bg.apply(gray, learningRate=-1 if warming else self.bg_lr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)   # drop shadows (127)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_small)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_big)
        cnts = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        blob_ratio, box = 0.0, None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            blob_ratio = cv2.contourArea(c) / float(fg.size)
            x, y, w, h = cv2.boundingRect(c)
            box = [int(v / scale) for v in (x, y, x + w, y + h)]

        # 3) event logic
        t = stream_time
        global_change = ratio >= self.global_ratio
        spike = bts = False
        still_for = 0.0

        if not warming and not global_change:
            spike = (ratio >= self.spike_ratio and
                     ratio >= self.spike_factor * max(self.base, 0.002))
            if ratio >= self.spike_ratio:                 # big activity = a "burst"
                self.burst_t, self.still_since, self.latched = t, None, False
            elif self.burst_t is not None and not self.latched:
                if ratio < self.still_ratio:
                    if self.still_since is None:
                        self.still_since = t
                    still_for = t - self.still_since
                    if still_for >= self.stop_seconds:
                        if blob_ratio >= self.min_blob_ratio:
                            bts, self.latched = True, True    # big blob stayed -> fire once
                        else:                                 # blob gone: person left the scene
                            self.burst_t, self.still_since = None, None
                else:
                    self.still_since = None                   # moderate motion, not still yet
                if (self.burst_t is not None and self.still_since is None
                        and t - self.burst_t > self.burst_expire_s):
                    self.burst_t = None
            self.base = 0.9 * self.base + 0.1 * ratio

        self.n += 1
        ms = (time.perf_counter() - t0) * 1000.0
        self.ms.append(ms)
        self.n_spikes += int(spike)
        return MotionResult(
            stream_time=round(t, 2), motion_ratio=round(ratio, 4),
            blob_ratio=round(blob_ratio, 4), blob_box=box, spike=spike,
            burst_then_still=bts, still_for_s=round(still_for, 2),
            global_change=global_change, warming_up=warming, ms=round(ms, 2))

    def stats(self) -> dict:
        if not self.ms:
            return {}
        return {
            "frames": len(self.ms),
            "motion_ms_p50": round(float(np.percentile(self.ms, 50)), 2),
            "motion_ms_p95": round(float(np.percentile(self.ms, 95)), 2),
            "pct_frames_spike": round(100.0 * self.n_spikes / len(self.ms), 1),
        }


# ---------------- self-test / demo ----------------
def _fmt(r: MotionResult) -> str:
    flags = [n for n, v in (("SPIKE", r.spike), ("BURST_THEN_STILL", r.burst_then_still),
                            ("GLOBAL", r.global_change), ("warmup", r.warming_up)) if v]
    return (f"t={r.stream_time:6.2f}s  motion={r.motion_ratio:.4f}  blob={r.blob_ratio:.4f}  "
            f"{r.ms:5.1f} ms  {' '.join(flags)}")


def _synthetic(person_leaves: bool, seed=0):
    """4 fps. 3 s empty scene, person walks in (2 s), then stands/lies still
    (4 s) or walks out of the frame."""
    rng = np.random.default_rng(seed)
    frames = []

    def noisy(rect=None):
        img = np.full((480, 640, 3), 100, np.uint8)
        if rect:
            x, y, w, h = rect
            img[y:y + h, x:x + w] = (20, 20, 20)
        noise = rng.normal(0, 2, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    for _ in range(12):                                   # empty scene
        frames.append(noisy())
    x = 40
    for _ in range(8):                                    # walking in, 40 px per frame
        frames.append(noisy((x, 200, 60, 120)))
        x += 40
    if person_leaves:
        for _ in range(8):                                # keeps walking out of frame
            frames.append(noisy((x, 200, 60, 120)) if x < 640 else noisy())
            x += 60
        for _ in range(16):
            frames.append(noisy())
    else:
        for _ in range(16):                               # stops and stays
            frames.append(noisy((x - 40, 200, 60, 120)))
    return frames


def selftest():
    ok = True
    for name, leaves in (("person stops and stays", False), ("person walks away", True)):
        det = MotionDetector()
        det.reset()
        results = [det.process(f, i / 4.0) for i, f in enumerate(_synthetic(leaves))]
        fired = [r for r in results if r.burst_then_still]
        expected = not leaves
        passed = bool(fired) == expected
        ok &= passed
        when = f" at t={fired[0].stream_time}s" if fired else ""
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: burst_then_still="
              f"{bool(fired)}{when} (expected {expected})")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not args.video:
        ap.error("give --video or --selftest")

    det = MotionDetector()
    det.reset()
    cap = cv2.VideoCapture(args.video)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / args.fps))
    idx = 0
    while cap.grab():                     # grab() skips decoding frames we drop
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                print(_fmt(det.process(frame, idx / native)))
        idx += 1
    cap.release()
    print("\n--- MOTION SUMMARY ---")
    for k, v in det.stats().items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
