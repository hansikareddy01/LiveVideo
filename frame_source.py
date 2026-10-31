"""
frame_source.py  -  Stage 1: one interface for video file, webcam and RTSP

    src = FrameSource("clip1.mp4", camera_id="cam1")        # file
    src = FrameSource(0, camera_id="cam2")                  # webcam
    src = FrameSource("rtsp://user:pw@10.0.0.5/stream1", camera_id="cam3")
    for frame in src.frames():
        if frame.after_gap:          # live stream came back after an outage:
            ...reset tracker / rule / motion state here...
        yolo.process(frame.image, frame.wall_time, frame.stream_time)

QUICK     grab() moves past frames we drop WITHOUT decoding them; only the
          ~4 fps we keep are decoded with retrieve(). capture_ms per kept frame
          is recorded (time inside grab()/retrieve(); for a live camera it
          includes waiting for the camera, so it is a decode cost mainly for files).
ACCURATE  full-resolution frames, a 6 s pre-roll, and pick_sharpest() (variance
          of the Laplacian) to choose the storyboard frame.
          Memory: ~6 MB per 1080p frame x ~25 frames = ~150 MB per camera for 6 s
          at 4 fps. Lower preroll_s or the camera resolution if that is too much.
APPLICATION  Every Frame has camera_id, an ISO wall-clock time (what goes on an
          alert) and stream_time in seconds (what timing rules use). For a file
          wall_time = wall_start + stream_time. For a live stream stream_time runs
          on a monotonic clock that KEEPS RUNNING through outages, so gaps show up.

LIVE STREAMS (webcam / RTSP) are read by a background thread, so a stalled
camera can never freeze the pipeline:
    * watchdog: no frame for stall_s (default 5 s) = stream is dead, even if
      OpenCV raised no error (common with RTSP)
    * reconnect with backoff 1, 2, 4, 8, 16, 30 s (repeating 30 s) until it works
    * SourceEvent log: "offline", "retry", "retry_failed", "online", "gave_up"
    * first frame after an outage has after_gap=True and gap_s set
    * the hand-off to the pipeline never blocks: if the consumer is slow, the
      oldest un-read frame is dropped (counted in stats["reader_drops"])
RTSP is opened over TCP by default (OPENCV_FFMPEG_CAPTURE_OPTIONS).
"""
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import cv2
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


# ---------------------------------------------------------------- data
@dataclass
class Frame:
    image: np.ndarray            # full-resolution BGR
    camera_id: str
    wall_time: str               # ISO, e.g. 2026-10-16T14:03:22.250
    stream_time: float           # seconds on the stream clock (use for timing rules)
    frame_index: int             # index of the grabbed frame it came from
    sharpness: float             # variance of Laplacian (higher = sharper)
    capture_ms: float            # grab()+retrieve() time spent to produce it
    after_gap: bool = False      # first frame after a reconnect
    gap_s: float = 0.0           # length of that outage


@dataclass
class SourceEvent:
    kind: str                    # offline | retry | retry_failed | online | gave_up
    wall_time: str
    stream_time: float
    detail: str
    gap_s: float = 0.0


# ---------------------------------------------------------------- helpers
def sharpness(img: np.ndarray, width: int = 640) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if w > width:
        g = cv2.resize(g, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def pick_sharpest(frames: List[Frame], k: int = 1) -> List[Frame]:
    """The k sharpest frames (best first). Use on the pre-roll for the storyboard."""
    return sorted(frames, key=lambda f: f.sharpness, reverse=True)[:k]


def detect_kind(src) -> str:
    if isinstance(src, int) or (isinstance(src, str) and src.strip().isdigit()):
        return "webcam"
    if str(src).lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
        return "rtsp"
    return "file"


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def _pcts(values):
    if not values:
        return None, None
    return (round(float(np.percentile(values, 50)), 2),
            round(float(np.percentile(values, 95)), 2))


# ---------------------------------------------------------------- live reader thread
class _Reader(threading.Thread):
    """Reads one connection until it fails. A new one is made per reconnect."""

    def __init__(self, owner: "FrameSource", out_q: "queue.Queue", stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.owner, self.q, self.stop_evt = owner, out_q, stop_evt
        self.cap = None
        self.error: Optional[str] = None

    def release(self):
        cap = self.cap
        if cap is not None:
            try:
                cap.release()                 # also unblocks a grab() stuck on a dead stream
            except Exception:
                pass

    def run(self):
        o = self.owner
        try:
            cap = o._opener(o.source)
            self.cap = cap
            if self.stop_evt.is_set():        # watchdog gave up on us while opening
                self.release()
                return
            if not cap.isOpened():
                self.error = "open failed"
                return
            interval = 1.0 / o.fps if o.fps > 0 else 0.0
            t_start = time.monotonic()
            next_t, idx, acc = None, 0, 0.0
            while not self.stop_evt.is_set():
                t0 = time.perf_counter()
                ok = cap.grab()
                acc += (time.perf_counter() - t0) * 1000.0
                if not ok:
                    self.error = "grab failed (stream dropped or ended)"
                    return
                o._grabbed += 1
                mono = time.monotonic()
                rel = mono - t_start
                if next_t is None:
                    next_t = rel                       # keep the very first frame
                if rel + 1e-9 >= next_t:
                    t1 = time.perf_counter()
                    ok, img = cap.retrieve()           # decode only the frames we keep
                    acc += (time.perf_counter() - t1) * 1000.0
                    if ok and img is not None:
                        item = (img, time.time(), mono, idx, acc)
                        acc = 0.0
                        next_t += interval
                        if next_t < rel - interval:    # fell behind: resync
                            next_t = rel
                        try:
                            self.q.put_nowait(item)    # never block the reader
                        except queue.Full:
                            try:
                                self.q.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.q.put_nowait(item)
                            except queue.Full:
                                pass
                            o._reader_drops += 1
                idx += 1
        except Exception as e:                         # any capture error = connection lost
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.release()


# ---------------------------------------------------------------- the source
class FrameSource:
    def __init__(self, source, camera_id: str = "cam1", fps: float = 4.0,
                 preroll_s: float = 6.0, realtime: bool = False,
                 wall_start: Optional[str] = None, stall_s: float = 5.0,
                 backoff=(1, 2, 4, 8, 16, 30), max_retries: Optional[int] = None,
                 kind: Optional[str] = None, opener: Optional[Callable] = None,
                 on_event: Optional[Callable[[SourceEvent], None]] = None,
                 compute_sharpness: bool = True):
        """
        fps         sampling rate (frames kept per second of stream time)
        realtime    file only: pace delivery to the video clock (simulate live)
        wall_start  file only: ISO time of the first frame (default: now)
        stall_s     live only: no frame for this long = dead stream
        backoff     live only: seconds to wait between reconnect attempts
        max_retries live only: None = retry forever
        opener      callable(source) -> cv2.VideoCapture-like (for tests)
        """
        self.source = int(source) if (isinstance(source, str) and source.strip().isdigit()) else source
        self.camera_id = camera_id
        self.fps = fps
        self.preroll_s = preroll_s
        self.realtime = realtime
        self.wall_start = wall_start
        self.stall_s = stall_s
        self.backoff = tuple(backoff) or (1,)
        self.max_retries = max_retries
        self.kind = kind or detect_kind(self.source)
        self._opener = opener or (lambda s: cv2.VideoCapture(s))
        self.on_event = on_event
        self.compute_sharpness = compute_sharpness

        self.events: List[SourceEvent] = []
        self.capture_ms: List[float] = []
        self._preroll: deque = deque()
        self._grabbed = 0
        self._kept = 0
        self._reader_drops = 0
        self.reconnects = 0
        self.offline_events = 0
        self.abandoned = 0
        self._video_s = 0.0
        self._t_start: Optional[float] = None
        self._t_resume: Optional[float] = None
        self._mono0 = time.monotonic()
        self._closed = False
        self._reader: Optional[_Reader] = None

    # ---- public ----
    def frames(self):
        """Generator of Frame. Ends at end-of-file, close(), or gave_up."""
        if self.kind == "file":
            yield from self._iter_file()
        else:
            yield from self._iter_live()

    def preroll(self, seconds: Optional[float] = None) -> List[Frame]:
        """Frames from the last `seconds` (default preroll_s) of stream time, oldest first."""
        if not self._preroll:
            return []
        sec = self.preroll_s if seconds is None else seconds
        end = self._preroll[-1].stream_time
        return [f for f in self._preroll if end - f.stream_time <= sec]

    def close(self):
        self._closed = True
        if self._reader is not None:
            self._reader.stop_evt.set()
            self._reader.release()

    def stats(self) -> dict:
        p50, p95 = _pcts(self.capture_ms)
        wall = ((self._t_resume or time.perf_counter()) - self._t_start) if self._t_start else 0.0
        return {
            "kind": self.kind,
            "frames_grabbed": self._grabbed,
            "frames_kept": self._kept,
            "effective_fps": round(self._kept / self._video_s, 2) if self._video_s > 0 else None,
            "capture_ms_p50": p50, "capture_ms_p95": p95,
            "video_s": round(self._video_s, 2),
            "wall_s": round(wall, 2),
            "real_time_factor": round(self._video_s / wall, 2) if wall > 0 else None,
            "offline_events": self.offline_events,
            "reconnects": self.reconnects,
            "reader_drops": self._reader_drops,
            "abandoned_threads": self.abandoned,
        }

    # ---- internals ----
    def _event(self, kind: str, detail: str, gap_s: float = 0.0):
        ev = SourceEvent(kind=kind, wall_time=_iso(datetime.now()),
                         stream_time=round(time.monotonic() - self._mono0, 2),
                         detail=detail, gap_s=round(gap_s, 2))
        self.events.append(ev)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass

    def _finish(self, f: Frame) -> Frame:
        self._kept += 1
        self.capture_ms.append(f.capture_ms)
        self._video_s = max(self._video_s, f.stream_time)
        self._preroll.append(f)
        while self._preroll and f.stream_time - self._preroll[0].stream_time > self.preroll_s:
            self._preroll.popleft()
        return f

    def _sharp(self, img) -> float:
        return sharpness(img) if self.compute_sharpness else 0.0

    # ---- file ----
    def _iter_file(self):
        cap = self._opener(self.source)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.source}")
        native = cap.get(cv2.CAP_PROP_FPS)
        if not native or native != native or native < 1.0:
            native = 30.0
        wall0 = datetime.fromisoformat(self.wall_start) if self.wall_start else datetime.now()
        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        next_t, idx, acc = 0.0, 0, 0.0
        self._t_start = time.perf_counter()
        try:
            while not self._closed:
                self._t_resume = time.perf_counter()
                t0 = time.perf_counter()
                ok = cap.grab()                       # skip-decode: cheap
                acc += (time.perf_counter() - t0) * 1000.0
                if not ok:
                    break
                self._grabbed += 1
                t = idx / native
                if t + 1e-9 >= next_t:
                    t1 = time.perf_counter()
                    ok, img = cap.retrieve()          # decode only the kept frames
                    acc += (time.perf_counter() - t1) * 1000.0
                    if ok and img is not None:
                        if self.realtime:
                            delay = (self._t_start + t) - time.perf_counter()
                            if delay > 0:
                                time.sleep(delay)
                        f = Frame(image=img, camera_id=self.camera_id,
                                  wall_time=_iso(wall0 + timedelta(seconds=t)),
                                  stream_time=round(t, 3), frame_index=idx,
                                  sharpness=self._sharp(img), capture_ms=acc)
                        acc = 0.0
                        next_t += interval
                        if next_t < t - interval:
                            next_t = t
                        yield self._finish(f)
                idx += 1
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self._t_resume = time.perf_counter()

    # ---- live (webcam / RTSP) ----
    def _iter_live(self):
        self._t_start = time.perf_counter()
        self._mono0 = time.monotonic()
        attempt = 0
        outage_start: Optional[float] = None
        last_frame_mono: Optional[float] = None
        try:
            while not self._closed:
                q: "queue.Queue" = queue.Queue(maxsize=2)
                stop = threading.Event()
                rd = _Reader(self, q, stop)
                self._reader = rd
                rd.start()
                last = time.monotonic()
                reason = "reader ended"
                while not self._closed:
                    self._t_resume = time.perf_counter()
                    try:
                        img, ts, mono, idx, acc = q.get(timeout=0.1)
                    except queue.Empty:
                        if not rd.is_alive() and q.empty():
                            reason = rd.error or "reader ended"
                            break
                        if time.monotonic() - last > self.stall_s:
                            reason = f"stalled: no frame for {self.stall_s:.1f} s"
                            break
                        continue
                    gap, after = 0.0, False
                    if outage_start is not None:
                        gap = mono - outage_start
                        after = self._kept > 0
                        self.reconnects += 1
                        self._event("online", f"stream back after {gap:.1f} s", gap)
                        outage_start, attempt = None, 0
                    last = time.monotonic()
                    last_frame_mono = mono
                    f = Frame(image=img, camera_id=self.camera_id,
                              wall_time=_iso(datetime.fromtimestamp(ts)),
                              stream_time=round(mono - self._mono0, 3), frame_index=idx,
                              sharpness=self._sharp(img), capture_ms=acc,
                              after_gap=after, gap_s=round(gap, 2))
                    yield self._finish(f)

                # connection lost (or closed): clean up the reader
                stop.set()
                rd.release()
                rd.join(timeout=1.0)
                if rd.is_alive():
                    self.abandoned += 1               # stuck in open()/grab(); it is a daemon
                if self._closed:
                    break
                if outage_start is None:
                    outage_start = last_frame_mono if last_frame_mono is not None else self._mono0
                    self.offline_events += 1
                    self._event("offline", reason)
                else:
                    self._event("retry_failed", reason)
                attempt += 1
                if self.max_retries is not None and attempt > self.max_retries:
                    self._event("gave_up", f"gave up after {attempt - 1} retries")
                    return
                delay = self.backoff[min(attempt - 1, len(self.backoff) - 1)]
                self._event("retry", f"attempt {attempt} in {delay:.1f} s")
                end = time.monotonic() + delay
                while not self._closed and time.monotonic() < end:
                    time.sleep(0.02)
        finally:
            if self._reader is not None:
                self._reader.stop_evt.set()
                self._reader.release()
            self._t_resume = time.perf_counter()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print frames from a source (smoke test)")
    ap.add_argument(
        "source", nargs="?", default="videos/v026_converted 3.avi",
        help="Video path, webcam index, or RTSP URL (default: videos/v026_converted 3.avi)"
    )
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--seconds", type=float, default=15.0, help="stop after this much stream time")
    a = ap.parse_args()
    if detect_kind(a.source) == "file" and not os.path.isfile(a.source):
        ap.error(f"Video not found: {a.source!r}. Check the path or pass a valid source.")
    s = FrameSource(a.source, a.camera, fps=a.fps, on_event=lambda e: print("EVENT", e))
    for fr in s.frames():
        print(f"{fr.camera_id} {fr.wall_time} t={fr.stream_time:7.2f}s "
              f"sharp={fr.sharpness:7.1f} cap={fr.capture_ms:5.1f} ms "
              f"{'AFTER_GAP ' + str(fr.gap_s) if fr.after_gap else ''}")
        if fr.stream_time >= a.seconds:
            s.close()
            break
    print(s.stats())
