"""
run_pipeline.py  -  ONE video through every stage, step by step, gate included

    python run_pipeline.py --video "v003_converted 3.mp4"
    python run_pipeline.py --video clip.mp4 --quiet            # only gate SENDs
    python run_pipeline.py --video clip.mp4 --realtime         # play at real speed (live demo)
    python run_pipeline.py --video clip.mp4 --send-url http://THEIR-IP:8000/candidate
    python run_pipeline.py --video 0                           # webcam
    python run_pipeline.py --video rtsp://...                  # RTSP, reconnects on drop

For each kept frame it prints the six stages:
    1 FRAME   frame_source   time, capture ms, sharpness
    2 YOLO    yolo_tracking  people, track IDs, torso angle, aspect, speed
    3 MOTION  motion_detector  motion ratio, blob, SPIKE / BURST_THEN_STILL
    4 RULES   rule_engine    any rule that fired, with its "why"
    5 GATE    gate_queue     SEND or SKIP, and the reason
    6 QUEUE   gate_queue     queued / DROPPED, queue size
Candidates are saved to --save-dir (default "candidates") as .jpg + .json, which
is exactly what Member 2 receives. With --send-url they are also POSTed.
"""
import argparse
import os
import re
import sys

from frame_source import FrameSource, pick_sharpest
from gate_queue import CandidateQueue, Gate
from handoff import HandoffSender
from motion_detector import MotionDetector
from rule_engine import RuleConfig, RuleEngine
import cv2
from yolo_tracking import FrameResult


class NullDetector:
    """--detector none: no people are ever detected (motion + gate only)."""
    name = "none"

    def __init__(self, camera_id):
        self.camera_id = camera_id

    def reset(self):
        pass

    def process(self, image, wall_time, stream_time):
        b = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
        return FrameResult(self.camera_id, wall_time, round(stream_time, 2), 0, [],
                           round(b, 1), 0.0, 0.0)

    def stats(self):
        return {}


def make_detector(kind, camera_id):
    if kind == "none":
        return NullDetector(camera_id)
    from yolo_tracking import YoloTracker        # needs ultralytics + torch
    d = YoloTracker(camera_id)
    d.name = "yolov8n-pose"
    return d


def people_str(fr):
    if not fr.persons:
        return "nobody detected"
    out = []
    for p in fr.persons:
        ang = "n/a" if p.torso_angle is None else f"{p.torso_angle:.0f}"
        out.append(f"id{p.track_id}(ang={ang},asp={p.aspect},spd={p.kp_speed})")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="file path, webcam index, or rtsp:// URL")
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--detector", default="yolo", choices=["yolo", "none"])
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--realtime", action="store_true", help="pace a file to its own clock")
    ap.add_argument("--quiet", action="store_true", help="print only frames the gate SENDS")
    ap.add_argument("--send-url")
    ap.add_argument("--save-dir", default="candidates")
    ap.add_argument("--door-line", help="x1,y1,x2,y2 normalised 0-1 (enables tailgating)")
    a = ap.parse_args()

    cfg = RuleConfig()
    if a.door_line:
        x1, y1, x2, y2 = [float(v) for v in a.door_line.split(",")]
        cfg.door_line = ((x1, y1), (x2, y2))

    src = FrameSource(a.video, a.camera, fps=a.fps, realtime=a.realtime,
                      on_event=lambda e: print(f"   !! SOURCE {e.kind.upper()}: {e.detail}"))
    det = make_detector(a.detector, a.camera)
    motion, gate = MotionDetector(), Gate(a.camera)
    queue = CandidateQueue(maxsize=5)
    base = os.path.splitext(os.path.basename(str(a.video)))[0]
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_") + "_"   # v001_converted_3_
    sender = HandoffSender(queue=queue, url=a.send_url, save_dir=a.save_dir, name_prefix=prefix)
    sender.start()
    det.reset()
    motion.reset()
    engine = None

    print(f"video={a.video}  detector={a.detector}  fps={a.fps}  "
          f"{'sending to ' + a.send_url if a.send_url else 'saving to ' + a.save_dir + '/'}\n")
    try:
        for f in src.frames():
            if engine is None:
                h, w = f.image.shape[:2]
                engine = RuleEngine(a.camera, cfg, frame_wh=(w, h))
            if f.after_gap:
                print(f"   !! stream returned after {f.gap_s}s: resetting tracker and rules")
                det.reset(); motion.reset(); gate.reset(); engine.reset()

            fr = det.process(f.image, f.wall_time, f.stream_time)                 # 2
            mr = motion.process(f.image, f.stream_time)                           # 3
            hits = engine.update(fr, mr)                                          # 4
            cool0 = gate.n_cooldown
            c = gate.step(fr, mr, hits, frame=f.image)                            # 5
            qline = ""
            if c is not None:                                                     # 6
                if c.trigger != "heartbeat":
                    pre = [x for x in src.preroll(6.0) if x is not f]
                    c.extra_frames = pick_sharpest(pre, 1) if pre else []
                ok = queue.put(c)
                qline = (f"queued ({len(queue)}/{queue.maxsize})" if ok
                         else "DROPPED (queue full, lowest priority)")

            if a.quiet and c is None:
                continue
            flags = [n for n, v in (("SPIKE", mr.spike), ("BURST_THEN_STILL", mr.burst_then_still),
                                    ("GLOBAL_CHANGE", mr.global_change), ("warmup", mr.warming_up)) if v]
            if c is not None:
                gate_line = (f"SEND  [{c.trigger}] {c.category_hint or '-'} "
                             f"conf={c.confidence}  {c.why}")
            elif gate.n_cooldown > cool0:
                gate_line = "SKIP  (same category sent recently: cooldown)"
            elif mr.spike:
                gate_line = "SKIP  (motion spike, but people visible in good light)"
            else:
                hb = max(0.0, gate.cfg.heartbeat_s - (f.stream_time - gate.last_send_t))
                gate_line = f"SKIP  (nothing to report; next heartbeat in {hb:.0f} s)"

            print(f"[t={f.stream_time:7.2f}s  {f.wall_time}]")
            print(f"  1 FRAME   capture {f.capture_ms:5.1f} ms   sharpness {f.sharpness:.0f}")
            print(f"  2 YOLO    {fr.person_count} people, {fr.infer_ms:.0f} ms, "
                  f"brightness {fr.brightness:.0f} | {people_str(fr)}")
            print(f"  3 MOTION  ratio {mr.motion_ratio:.4f}  blob {mr.blob_ratio:.4f}  {' '.join(flags)}")
            print("  4 RULES   " + ("; ".join(f"{h.category} ({h.confidence}): {h.why}" for h in hits)
                                    if hits else "none"))
            print(f"  5 GATE    {gate_line}")
            if qline:
                print(f"  6 QUEUE   {qline}")
            print()
    except KeyboardInterrupt:
        print("\nstopped")
        src.close()
    sender.stop()

    print("=" * 70)
    print("SUMMARY")
    for name, d in (("Stage 1 frame source", src.stats()), ("Stage 2 YOLO", det.stats()),
                    ("Stage 3 motion", motion.stats()), ("Stage 3 gate", gate.stats()),
                    ("Stage 4 queue", queue.stats()), ("Stage 4 handoff", sender.stats())):
        print(f"{name}: {d}")
    print(f"\nCandidates saved in: {a.save_dir}/" if a.save_dir else "")


if __name__ == "__main__":
    main()
