"""
rule_engine.py  -  Stage 3a: turn per-frame evidence into candidate events

Input  (every kept frame): FrameResult from yolo_tracking.py
                           MotionResult from motion_detector.py (optional)
Output : list of RuleHit, each with a category, a confidence, and a short
         "why" that cites the evidence, e.g.
         "2 people crossed the door line 1.8 s apart (tracks 3, 5)"

Rules look over TIME, not one frame:
  person_down          lying posture for ~2 s (+ a motion fallback)
  fighting             two close people with fast limb motion for ~1+ s
  tailgating           two tracks cross a door line in the same direction
                       within a few seconds
  crowd_incident       many people sustained, or a sudden surge in the count
  suspicious_activity  loitering: one track stays in a small area for a long time

Each condition fires ONCE per episode (latched), then re-arms. gate_queue.py
adds the per-category cooldown, the motion-spike rule and the heartbeat.
Confidence values are heuristics (0-1), not calibrated probabilities. Thresholds
are starting points: tune them on some clips, test on the others.

Self-test:  python rule_engine.py --selftest
"""
import argparse
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

# The shared categories. The brief lists 5; "normal" is assumed to be the sixth
# (confirm with your team). The VLM on Member 2's side uses the same names.
CATEGORIES = ["person_down", "fighting", "tailgating",
              "crowd_incident", "suspicious_activity", "normal"]


@dataclass
class RuleConfig:
    # person_down
    lying_angle: float = 60.0        # torso angle (deg from vertical) counted as lying
    lying_aspect: float = 1.2        # fallback when keypoints are bad: box w/h
    lying_s: float = 2.0             # must stay lying this long
    lying_gap_tol_s: float = 0.6     # ignore short flicker (keypoint noise)
    fall_transition_s: float = 2.0   # upright within this long before = "fell"
    upright_angle: float = 40.0      # below this (and narrow box) = clearly upright
    dark_brightness: float = 60.0
    # fighting  (kp_speed is per KEPT frame, so it depends on the sampling fps)
    fight_speed: float = 0.25        # mean keypoint speed of the pair
    fight_close: float = 0.8         # centre distance / box height
    fight_window_s: float = 2.5
    fight_min_hits: int = 4
    fight_min_frac: float = 0.6
    fight_min_span_s: float = 1.0
    fight_refire_s: float = 10.0
    # tailgating: line in NORMALISED coords ((x1,y1),(x2,y2)); needs frame_wh
    door_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    door_direction: str = "any"      # "in", "out" or "any"
    tailgate_gap_s: float = 3.0
    # crowd_incident
    crowd_count: int = 8
    crowd_s: float = 2.0
    crowd_surge: int = 4             # count rose by this much within surge_window_s
    crowd_surge_window_s: float = 5.0
    crowd_refire_s: float = 15.0
    # suspicious_activity (loitering)
    loiter_s: float = 60.0
    loiter_radius: float = 1.5       # max wander, in box heights
    # housekeeping
    prune_s: float = 5.0
    motion_dedupe_s: float = 5.0     # skip motion hint if a pose person_down just fired


@dataclass
class RuleHit:
    camera_id: str
    wall_time: str
    stream_time: float
    category: str
    confidence: float
    why: str
    trigger: str                      # "yolo_rule" | "motion"
    track_ids: List[int] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


class RuleEngine:
    """One instance per camera. Call reset() between videos."""

    def __init__(self, camera_id: str, cfg: Optional[RuleConfig] = None,
                 frame_wh: Optional[Tuple[int, int]] = None):
        self.camera_id = camera_id
        self.cfg = cfg or RuleConfig()
        self.frame_wh = frame_wh          # (width, height) in pixels, for door_line
        self.reset()

    def reset(self):
        self.lying: Dict[int, dict] = {}
        self.pair_hist: Dict[tuple, deque] = {}
        self.pair_fired: Dict[tuple, float] = {}
        self.side: Dict[int, int] = {}
        self.crossings: deque = deque()          # (t, tid, dir)
        self.tail_fired: Dict[tuple, float] = {}
        self.count_hist: deque = deque()
        self.crowd_since: Optional[float] = None
        self.crowd_last: float = -1e9
        self.tracks: Dict[int, dict] = {}
        self.last_hit: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def update(self, fr, mr=None) -> List[RuleHit]:
        t = fr.stream_time
        hits: List[RuleHit] = []
        hits += self._person_down(fr, t)
        for h in hits:
            self.last_hit[h.category] = t
        hits += self._motion_person_down(fr, mr, t)
        hits += self._fighting(fr, t)
        hits += self._tailgating(fr, t)
        hits += self._crowd(fr, t)
        hits += self._loitering(fr, t)
        for h in hits:
            self.last_hit[h.category] = t
        self._prune(t)
        return hits

    # ------------------------------------------------------------------
    def _hit(self, fr, category, conf, why, trigger, ids, evidence) -> RuleHit:
        return RuleHit(camera_id=fr.camera_id, wall_time=fr.wall_time,
                       stream_time=fr.stream_time, category=category,
                       confidence=round(min(conf, 0.95), 2), why=why,
                       trigger=trigger, track_ids=list(ids), evidence=evidence)

    # ---- person_down --------------------------------------------------
    def _is_lying(self, p) -> bool:
        if p.torso_angle is not None:
            return p.torso_angle >= self.cfg.lying_angle
        return p.aspect >= self.cfg.lying_aspect      # fallback

    def _person_down(self, fr, t) -> List[RuleHit]:
        c, out = self.cfg, []
        for p in fr.persons:
            if p.track_id < 0:
                continue
            s = self.lying.setdefault(p.track_id, {
                "since": None, "last_lying": None, "upright_before": None,
                "fired": False, "last_seen": t})
            s["last_seen"] = t
            if self._is_lying(p):
                if s["since"] is None:
                    s["since"] = t
                s["last_lying"] = t
            else:
                if s["since"] is None:
                    s["upright_before"] = t
                elif t - s["last_lying"] > c.lying_gap_tol_s:
                    s["since"], s["fired"] = None, False
                    s["upright_before"] = t
            if s["since"] is not None and not s["fired"] and t - s["since"] >= c.lying_s:
                s["fired"] = True
                dur = t - s["since"]
                fell = (s["upright_before"] is not None and
                        s["since"] - s["upright_before"] <= c.fall_transition_s)
                both = (p.torso_angle is not None and p.torso_angle >= c.lying_angle
                        and p.aspect >= c.lying_aspect)
                conf = 0.6 + (0.2 if fell else 0.0) + (0.1 if both else 0.0)
                ang = "n/a" if p.torso_angle is None else f"{p.torso_angle:.0f}"
                why = (f"track {p.track_id} lying for {dur:.1f} s "
                       f"(torso {ang} deg, aspect {p.aspect:.2f})"
                       + (", went down from upright" if fell else ""))
                out.append(self._hit(fr, "person_down", conf, why, "yolo_rule",
                                     [p.track_id],
                                     {"torso_angle": p.torso_angle, "aspect": p.aspect,
                                      "lying_s": round(dur, 1), "fell": fell,
                                      "box": p.box}))
        return out

    def _clearly_upright(self, p) -> bool:
        return (p.torso_angle is not None and p.torso_angle < self.cfg.upright_angle
                and p.aspect < 1.0)

    def _motion_person_down(self, fr, mr, t) -> List[RuleHit]:
        """Fallback: a big object stopped, and YOLO cannot say it is upright."""
        c = self.cfg
        if mr is None or not mr.burst_then_still:
            return []
        if t - self.last_hit.get("person_down", -1e9) < c.motion_dedupe_s:
            return []
        if fr.persons and all(self._clearly_upright(p) for p in fr.persons):
            return []        # YOLO sees people standing: someone just stopped walking
        dark = fr.brightness < c.dark_brightness
        why = (f"large motion, then still for {mr.still_for_s:.1f} s; "
               f"{fr.person_count} people detected" + (", dark scene" if dark else ""))
        conf = 0.5 + (0.05 if (fr.person_count == 0 or dark) else 0.0)
        return [self._hit(fr, "person_down", conf, why, "motion", [],
                          {"blob_box": mr.blob_box, "blob_ratio": mr.blob_ratio,
                           "still_for_s": mr.still_for_s, "brightness": fr.brightness})]

    # ---- fighting -----------------------------------------------------
    def _fighting(self, fr, t) -> List[RuleHit]:
        c, out = self.cfg, []
        people = [p for p in fr.persons if p.track_id >= 0]
        seen = set()
        for a, b in combinations(people, 2):
            key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
            seen.add(key)
            ca = ((a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2)
            cb = ((b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2)
            hh = max(a.box[3] - a.box[1], b.box[3] - b.box[1], 1.0)
            dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 / hh
            speed = (a.kp_speed + b.kp_speed) / 2.0
            flag = dist <= c.fight_close and speed >= c.fight_speed
            dq = self.pair_hist.setdefault(key, deque())
            dq.append((t, flag, speed))
            while dq and t - dq[0][0] > c.fight_window_s:
                dq.popleft()
            n_hit = sum(1 for _, f, _ in dq if f)
            span = dq[-1][0] - dq[0][0]
            if (n_hit >= c.fight_min_hits and n_hit / len(dq) >= c.fight_min_frac
                    and span >= c.fight_min_span_s
                    and t - self.pair_fired.get(key, -1e9) >= c.fight_refire_s):
                self.pair_fired[key] = t
                avg = sum(s for _, f, s in dq if f) / n_hit
                conf = 0.55 + min(0.35, max(0.0, avg - c.fight_speed))
                why = (f"tracks {key[0]} and {key[1]} close together with fast "
                       f"limb motion for {span:.1f} s (speed {avg:.2f})")
                out.append(self._hit(fr, "fighting", conf, why, "yolo_rule", list(key),
                                     {"avg_kp_speed": round(avg, 3),
                                      "span_s": round(span, 1), "hits": n_hit}))
        for key in list(self.pair_hist):
            if key not in seen and t - self.pair_hist[key][-1][0] > c.fight_window_s:
                del self.pair_hist[key]
        return out

    # ---- tailgating ---------------------------------------------------
    def _tailgating(self, fr, t) -> List[RuleHit]:
        c, out = self.cfg, []
        if c.door_line is None or self.frame_wh is None:
            return out
        W, H = self.frame_wh
        (ax, ay), (bx, by) = [(p[0] * W, p[1] * H) for p in c.door_line]
        dx, dy = bx - ax, by - ay
        len2 = dx * dx + dy * dy or 1.0
        new = []
        for p in fr.persons:
            if p.track_id < 0:
                continue
            fx, fy = (p.box[0] + p.box[2]) / 2.0, p.box[3]      # feet point
            cross = dx * (fy - ay) - dy * (fx - ax)
            s = 1 if cross > 0 else (-1 if cross < 0 else 0)
            u = ((fx - ax) * dx + (fy - ay) * dy) / len2         # position along the line
            prev = self.side.get(p.track_id)
            if prev and s and s != prev and -0.1 <= u <= 1.1:
                direction = "in" if s > 0 else "out"
                if c.door_direction in ("any", direction):
                    new.append((t, p.track_id, direction))
            if s:
                self.side[p.track_id] = s
        for (tn, tid, d) in new:
            for (to, oid, od) in self.crossings:
                key = (min(tid, oid), max(tid, oid))
                gap = tn - to
                if (oid != tid and od == d and 0 <= gap <= c.tailgate_gap_s
                        and key not in self.tail_fired):
                    self.tail_fired[key] = t
                    why = (f"2 people crossed the door line {gap:.1f} s apart "
                           f"(tracks {oid}, {tid}, direction {d})")
                    out.append(self._hit(fr, "tailgating", 0.75, why, "yolo_rule",
                                         [oid, tid], {"gap_s": round(gap, 2),
                                                      "direction": d}))
            self.crossings.append((tn, tid, d))
        while self.crossings and t - self.crossings[0][0] > 15.0:
            self.crossings.popleft()
        for k in [k for k, v in self.tail_fired.items() if t - v > 30.0]:
            del self.tail_fired[k]
        return out

    # ---- crowd_incident -----------------------------------------------
    def _crowd(self, fr, t) -> List[RuleHit]:
        c = self.cfg
        n = fr.person_count
        self.count_hist.append((t, n))
        while self.count_hist and t - self.count_hist[0][0] > c.crowd_surge_window_s:
            self.count_hist.popleft()
        if n >= c.crowd_count:
            if self.crowd_since is None:
                self.crowd_since = t
        else:
            self.crowd_since = None
        sustained = self.crowd_since is not None and t - self.crowd_since >= c.crowd_s
        low = min(x for _, x in self.count_hist)
        surge = (n - low) >= c.crowd_surge and n >= c.crowd_surge + 1
        if (sustained or surge) and t - self.crowd_last >= c.crowd_refire_s:
            self.crowd_last = t
            if sustained:
                why = f"{n} people in view for {t - self.crowd_since:.1f} s (limit {c.crowd_count})"
                conf = 0.6 + min(0.3, 0.05 * (n - c.crowd_count))
            else:
                why = f"people count rose from {low} to {n} within {c.crowd_surge_window_s:.0f} s"
                conf = 0.55
            return [self._hit(fr, "crowd_incident", conf, why, "yolo_rule", [],
                              {"person_count": n, "min_recent": low})]
        return []

    # ---- suspicious_activity (loitering) -------------------------------
    def _loitering(self, fr, t) -> List[RuleHit]:
        c, out = self.cfg, []
        for p in fr.persons:
            if p.track_id < 0:
                continue
            tr = self.tracks.setdefault(p.track_id, {"first": t, "pts": deque(),
                                                     "fired": False, "last_seen": t})
            tr["last_seen"] = t
            cx, cy = (p.box[0] + p.box[2]) / 2.0, (p.box[1] + p.box[3]) / 2.0
            tr["pts"].append((t, cx, cy, max(p.box[3] - p.box[1], 1.0)))
            while tr["pts"] and t - tr["pts"][0][0] > c.loiter_s:
                tr["pts"].popleft()
            lying = self.lying.get(p.track_id, {}).get("since") is not None
            if tr["fired"] or lying or t - tr["first"] < c.loiter_s:
                continue
            xs = [q[1] for q in tr["pts"]]
            ys = [q[2] for q in tr["pts"]]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            radius = max(((x - mx) ** 2 + (y - my) ** 2) ** 0.5 for x, y in zip(xs, ys))
            mean_h = sum(q[3] for q in tr["pts"]) / len(tr["pts"])
            if radius <= c.loiter_radius * mean_h:
                tr["fired"] = True
                why = (f"track {p.track_id} stayed within a small area for "
                       f"{t - tr['first']:.0f} s")
                out.append(self._hit(fr, "suspicious_activity", 0.4, why, "yolo_rule",
                                     [p.track_id],
                                     {"duration_s": round(t - tr["first"], 1),
                                      "radius_box_heights": round(radius / mean_h, 2)}))
        return out

    # ------------------------------------------------------------------
    def _prune(self, t):
        for d in (self.lying, self.tracks):
            for k in [k for k, v in d.items() if t - v["last_seen"] > self.cfg.prune_s]:
                del d[k]
        for k in [k for k in self.side if k not in self.lying and k not in self.tracks]:
            del self.side[k]


# ---------------- self-test: synthetic scenarios ----------------
def _person(tid, box, angle=8.0, aspect=0.4, speed=0.02):
    from yolo_tracking import PersonDet
    return PersonDet(track_id=tid, box=list(box), conf=0.9, keypoints=[],
                     torso_angle=angle, aspect=aspect, kp_speed=speed)


def _frame(t, persons, brightness=100.0):
    from yolo_tracking import FrameResult
    return FrameResult(camera_id="cam1", wall_time="2026-10-16T14:00:00",
                       stream_time=t, person_count=len(persons), persons=persons,
                       brightness=brightness, conf_used=0.25, infer_ms=95.0)


def _run(engine, frames, motions=None):
    hits = []
    for i, f in enumerate(frames):
        hits += engine.update(f, motions[i] if motions else None)
    return hits


def selftest() -> bool:
    ok = True

    def check(name, hits, want_cat, want_n=1):
        nonlocal ok
        got = [h for h in hits if h.category == want_cat]
        passed = len(got) == want_n and all(h.category == want_cat for h in hits)
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: "
              f"{[(h.category, h.stream_time, h.confidence, h.why) for h in hits] or 'no hits'}")

    # 1) fall: upright, then angle rises and stays high
    angles = [8, 9, 7, 10, 34, 71, 84, 86, 86, 85, 86, 86, 85, 86, 85, 86]
    fr = [_frame(i * 0.25, [_person(1, (100, 100, 160, 300), a, 0.4 if a < 60 else 1.8)])
          for i, a in enumerate(angles)]
    check("person falls", _run(RuleEngine("cam1"), fr), "person_down")

    # 2) normal walking: no hits
    fr = [_frame(i * 0.25, [_person(1, (100 + i * 5, 100, 160 + i * 5, 300), 6, 0.4, 0.05)])
          for i in range(40)]
    check("normal walking", _run(RuleEngine("cam1"), fr), "none", 0)

    # 3) fight: two close people with fast keypoints
    fr = [_frame(i * 0.25, [_person(1, (100, 100, 160, 300), 20, 0.5, 0.4),
                            _person(2, (150, 100, 210, 300), 25, 0.5, 0.45)])
          for i in range(10)]
    check("fight", _run(RuleEngine("cam1"), fr), "fighting")

    # 4) tailgating: vertical door line at x=0.5 of a 640x480 frame
    cfg = RuleConfig(door_line=((0.5, 0.0), (0.5, 1.0)))
    fr = []
    for i in range(24):
        t = i * 0.25
        x1 = 200 + i * 20                         # person 1 crosses x=320 around t=1.5
        x2 = 200 + max(0, i - 7) * 20             # person 2 crosses ~1.75 s later
        fr.append(_frame(t, [_person(1, (x1 - 30, 100, x1 + 30, 300)),
                             _person(2, (x2 - 30, 100, x2 + 30, 300))]))
    check("tailgating", _run(RuleEngine("cam1", cfg, (640, 480)), fr), "tailgating")

    # 5) crowd: 9 people for a few seconds
    people = [_person(i + 1, (50 * i, 100, 50 * i + 40, 250)) for i in range(9)]
    fr = [_frame(i * 0.25, people) for i in range(12)]
    check("crowd", _run(RuleEngine("cam1"), fr), "crowd_incident")

    # 6) loitering (loiter_s shortened to 10 s for the test)
    cfg = RuleConfig(loiter_s=10.0)
    fr = [_frame(i * 0.25, [_person(1, (100, 100, 160, 300))]) for i in range(60)]
    check("loitering", _run(RuleEngine("cam1", cfg), fr), "suspicious_activity")

    # 7) motion fallback: nobody detected, big object stopped
    from motion_detector import MotionResult
    mr_still = MotionResult(5.0, 0.0, 0.03, [10, 10, 80, 200], False, True, 2.0, False, False, 1.5)
    mr_none = MotionResult(5.0, 0.0, 0.03, None, False, False, 0.0, False, False, 1.5)
    hits = RuleEngine("cam1").update(_frame(5.0, [], 40.0), mr_still)
    check("motion fallback (0 people, dark)", hits, "person_down")

    # 8) motion burst-then-still but YOLO sees an upright person -> suppressed
    hits = RuleEngine("cam1").update(_frame(5.0, [_person(1, (100, 100, 160, 300), 5, 0.4)]), mr_still)
    check("motion suppressed when YOLO sees upright person", hits, "none", 0)
    _ = mr_none

    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    print("rule_engine is a library: call RuleEngine.update(frame_result, motion_result). "
          "Run --selftest to see it on synthetic scenarios.")
