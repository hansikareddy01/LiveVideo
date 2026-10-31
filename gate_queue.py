"""
gate_queue.py  -  Stage 3b + 4: decide what is sent, and never block capture

GATE (per kept frame). A frame becomes a Candidate if any of these is true:
  1. a rule fired (RuleHit from rule_engine.py, incl. its motion fallback)
  2. motion spike AND (0 people detected OR dark scene)   -> category unknown
  3. heartbeat: no candidate sent for heartbeat_s (12 s)
Otherwise the frame is skipped. A per-category cooldown stops one event from
being sent over and over. If several rules fire on the same frame they are
merged into ONE candidate (best confidence is the main one, the rest go in
`also`), so the same JPEG is not sent twice.

QUEUE (maxsize=5). put() never blocks the capture loop. When full, the lowest
priority item is dropped first, and every drop is counted:
    tier 0  heartbeat                       (dropped first)
    tier 1  motion spike, category unknown
    tier 2  rule event                      (dropped only as a last resort)
Within a tier the OLDEST is dropped, so the queue stays fresh. An incoming
heartbeat never pushes out a real event (it is dropped itself).

Hand-over numbers:  Gate.stats() -> % frames skipped
                    CandidateQueue.stats() -> dropped, queue wait p50/p95
                    gate_recall() -> recall + time to detect from labels.csv
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------- data
@dataclass
class Candidate:
    camera_id: str
    wall_time: str
    stream_time: float
    trigger: str                       # "yolo_rule" | "motion" | "heartbeat"
    category_hint: Optional[str]       # one of the shared categories, or None
    confidence: float
    why: str
    track_ids: List[int] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    also: List[dict] = field(default_factory=list)   # other rules on the same frame
    frame: object = None               # full-res BGR image (encoded later by handoff.py)
    seq: int = 0                       # set by the queue
    enqueued_at: float = 0.0           # set by the queue (perf_counter)

    @property
    def tier(self) -> int:
        if self.trigger == "heartbeat":
            return 0
        if self.category_hint is None:
            return 1
        return 2

    def to_json_dict(self) -> dict:
        """Everything except the image and queue internals (for handoff.py)."""
        return {
            "camera_id": self.camera_id, "wall_time": self.wall_time,
            "stream_time": self.stream_time, "trigger": self.trigger,
            "category_hint": self.category_hint, "confidence": self.confidence,
            "why": self.why, "track_ids": self.track_ids,
            "evidence": self.evidence, "also": self.also,
        }


@dataclass
class GateConfig:
    heartbeat_s: float = 12.0
    dark_brightness: float = 60.0
    default_cooldown_s: float = 5.0
    cooldown_s: Dict[str, float] = field(default_factory=lambda: {
        "person_down": 10.0, "fighting": 8.0, "tailgating": 5.0,
        "crowd_incident": 15.0, "suspicious_activity": 30.0,
        "motion_unknown": 5.0,
    })


# ---------------------------------------------------------------- gate
class Gate:
    """One instance per camera. Call reset() between videos."""

    def __init__(self, camera_id: str, cfg: Optional[GateConfig] = None):
        self.camera_id = camera_id
        self.cfg = cfg or GateConfig()
        self.reset()
        self.reset_stats()

    def reset(self):
        self.last_send_t: Optional[float] = None
        self.last_cat_send: Dict[str, float] = {}

    def reset_stats(self):
        self.n_seen = 0
        self.n_sent = 0
        self.n_cooldown = 0
        self.by_trigger: Dict[str, int] = {}

    def _cooling(self, key: str, t: float) -> bool:
        cd = self.cfg.cooldown_s.get(key, self.cfg.default_cooldown_s)
        return (t - self.last_cat_send.get(key, -1e9)) < cd

    def step(self, fr, mr, hits, frame=None) -> Optional[Candidate]:
        """fr: FrameResult, mr: MotionResult or None, hits: list of RuleHit."""
        t = fr.stream_time
        c = self.cfg
        self.n_seen += 1
        if self.last_send_t is None:
            self.last_send_t = t              # heartbeat timer starts at the first frame

        base_ev = {
            "person_count": fr.person_count,
            "track_ids": [p.track_id for p in fr.persons],
            "boxes": [p.box for p in fr.persons],
            "brightness": fr.brightness,
            "motion_ratio": mr.motion_ratio if mr is not None else None,
        }
        cand: Optional[Candidate] = None

        # 1) rule hits (after per-category cooldown)
        ok = []
        for h in sorted(hits, key=lambda h: -h.confidence):
            if self._cooling(h.category, t):
                self.n_cooldown += 1
            else:
                ok.append(h)
        if ok:
            main = ok[0]
            for h in ok:
                self.last_cat_send[h.category] = t
            cand = Candidate(
                camera_id=fr.camera_id, wall_time=fr.wall_time, stream_time=t,
                trigger=main.trigger, category_hint=main.category,
                confidence=main.confidence, why=main.why, track_ids=main.track_ids,
                evidence={**base_ev, "rule": main.evidence},
                also=[{"category": h.category, "why": h.why, "confidence": h.confidence}
                      for h in ok[1:]], frame=frame)

        # 2) motion spike with nobody detected, or in the dark
        elif (mr is not None and mr.spike and not mr.global_change
              and not mr.warming_up
              and (fr.person_count == 0 or fr.brightness < c.dark_brightness)):
            if self._cooling("motion_unknown", t):
                self.n_cooldown += 1
            else:
                self.last_cat_send["motion_unknown"] = t
                reason = ("no person detected" if fr.person_count == 0
                          else "dark scene")
                cand = Candidate(
                    camera_id=fr.camera_id, wall_time=fr.wall_time, stream_time=t,
                    trigger="motion", category_hint=None, confidence=0.4,
                    why=f"motion spike ({mr.motion_ratio * 100:.1f}% of pixels), {reason}",
                    evidence=base_ev, frame=frame)

        # 3) heartbeat
        if cand is None and (t - self.last_send_t) >= c.heartbeat_s:
            cand = Candidate(
                camera_id=fr.camera_id, wall_time=fr.wall_time, stream_time=t,
                trigger="heartbeat", category_hint=None, confidence=0.0,
                why=f"heartbeat: nothing sent for {t - self.last_send_t:.0f} s",
                evidence=base_ev, frame=frame)

        if cand is not None:
            self.last_send_t = t
            self.n_sent += 1
            self.by_trigger[cand.trigger] = self.by_trigger.get(cand.trigger, 0) + 1
        return cand

    def stats(self) -> dict:
        skipped = self.n_seen - self.n_sent
        return {
            "frames_seen": self.n_seen,
            "candidates_sent": self.n_sent,
            "pct_frames_skipped": round(100.0 * skipped / self.n_seen, 1) if self.n_seen else 0.0,
            "sent_by_trigger": dict(self.by_trigger),
            "suppressed_by_cooldown": self.n_cooldown,
        }


# ---------------------------------------------------------------- queue
class CandidateQueue:
    """Bounded, thread-safe, put() never blocks. See module docstring."""

    def __init__(self, maxsize: int = 5):
        self.maxsize = maxsize
        self._items: List[Candidate] = []
        self._cv = threading.Condition(threading.Lock())
        self._seq = 0
        self.reset_stats()

    def reset_stats(self):
        self.n_put = 0
        self.n_got = 0
        self.dropped = {"heartbeat": 0, "motion": 0, "rule": 0}
        self.wait_ms: List[float] = []

    def __len__(self):
        with self._cv:
            return len(self._items)

    def put(self, cand: Candidate) -> bool:
        """Returns True if cand was queued, False if cand itself was dropped."""
        with self._cv:
            self._seq += 1
            cand.seq = self._seq
            cand.enqueued_at = time.perf_counter()
            self.n_put += 1
            if len(self._items) < self.maxsize:
                self._items.append(cand)
                self._cv.notify()
                return True
            # full: drop the lowest tier, oldest first. The newcomer is only
            # dropped if it is strictly lower priority than everything queued.
            victim = min(self._items + [cand], key=lambda c: (c.tier, c.seq))
            self.dropped[("heartbeat", "motion", "rule")[victim.tier]] += 1
            if victim is cand:
                return False
            self._items.remove(victim)
            self._items.append(cand)
            self._cv.notify()
            return True

    def get(self, timeout: Optional[float] = 0.0) -> Optional[Candidate]:
        """timeout=0 -> don't wait; timeout=None -> wait forever."""
        with self._cv:
            if not self._items and (timeout is None or timeout > 0):
                self._cv.wait_for(lambda: self._items, timeout)
            if not self._items:
                return None
            c = self._items.pop(0)
            self.wait_ms.append((time.perf_counter() - c.enqueued_at) * 1000.0)
            self.n_got += 1
            return c

    def stats(self) -> dict:
        d = self.dropped
        out = {"put": self.n_put, "got": self.n_got,
               "dropped_total": sum(d.values()),
               "dropped_heartbeat": d["heartbeat"],
               "dropped_motion": d["motion"],
               "dropped_rule_events": d["rule"]}      # should stay 0
        if self.wait_ms:
            out["queue_wait_ms_p50"] = round(float(np.percentile(self.wait_ms, 50)), 2)
            out["queue_wait_ms_p95"] = round(float(np.percentile(self.wait_ms, 95)), 2)
        return out


# ---------------------------------------------------------------- metric
def gate_recall(events, candidates, slack_before_s=1.0, slack_after_s=3.0,
                triggers=None) -> dict:
    """Of the real events, how many passed the gate?

    events      list of (start_s, end_s) from labels.csv
    candidates  Candidate objects (or anything with .stream_time / .trigger)
    triggers    optional set, e.g. {"yolo_rule"} for the YOLO-only number, or
                {"yolo_rule", "motion"} for YOLO+motion. Heartbeats are only
                counted if you include "heartbeat".
    An event is caught if a candidate lands in [start - slack_before, end + slack_after].
    time_to_detect = first such candidate time - event start (can be < 0 if the
    rule fired inside the slack before the labelled start).
    """
    cands = [c for c in candidates if triggers is None or c.trigger in triggers]
    caught, ttd = 0, []
    for (s, e) in events:
        hit = sorted(c.stream_time for c in cands
                     if s - slack_before_s <= c.stream_time <= e + slack_after_s)
        if hit:
            caught += 1
            ttd.append(round(hit[0] - s, 2))
    n = len(events)
    return {"events": n, "caught": caught,
            "recall": round(caught / n, 3) if n else None,
            "time_to_detect_s": ttd}


if __name__ == "__main__":
    print("gate_queue is a library. Run:  python -m unittest test_gate_queue -v")
