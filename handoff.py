"""
handoff.py  -  Stage 4: send candidates to Member 2's laptop

Takes Candidates off the CandidateQueue (gate_queue.py), encodes the frame with
cv2.imencode(".jpg"), and POSTs  JPEG bytes + JSON  to Member 2.

WIRE FORMAT (assumed; agree it with Member 2): HTTP POST, multipart/form-data
    meta     application/json   the Candidate JSON (see below)
    image    image/jpeg         the trigger frame
    image_1  image/jpeg         optional: sharpest pre-roll frame (storyboard)
meta = Candidate.to_json_dict() + {"schema": "event-candidate/1",
        "frames": [{"field": "image", "role": "trigger", "stream_time": ..}, ..],
        "jpeg_bytes": [...]}

Member 2's side, with FastAPI (pip install fastapi uvicorn python-multipart):

    from fastapi import FastAPI, File, Form, UploadFile
    import json
    app = FastAPI()
    @app.post("/candidate")
    async def candidate(meta: str = Form(...), image: UploadFile = File(None),
                        image_1: UploadFile = File(None)):
        m = json.loads(meta)                  # camera_id, wall_time, category_hint, why ...
        jpg = await image.read() if image else None
        return {"ok": True}

No laptop yet?  Run a receiver here:
    python handoff.py --receive --port 8000 --out received/
and test the link:
    python handoff.py --ping http://127.0.0.1:8000/candidate

The sender runs in its OWN thread, so a slow or offline laptop never blocks
capture. A failed send is retried (retries, with backoff); after that it is
counted as failed. Set save_dir to keep a copy of everything on disk too (also
the fallback when the laptop is not reachable yet).
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from email import message_from_bytes, policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------- encoding
def encode_jpeg(img: np.ndarray, quality: int = 85, max_width: Optional[int] = None) -> bytes:
    """cv2.imencode('.jpg') with optional downscale (max_width=None keeps full resolution)."""
    if max_width and img.shape[1] > max_width:
        h = int(img.shape[0] * max_width / img.shape[1])
        img = cv2.resize(img, (max_width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes()


def build_multipart(meta: dict, images: List[Tuple[str, bytes]]) -> Tuple[bytes, str]:
    boundary = "----cand" + uuid.uuid4().hex
    b = boundary.encode()
    parts = [b"--" + b + b"\r\n"
             b'Content-Disposition: form-data; name="meta"\r\n'
             b"Content-Type: application/json\r\n\r\n"
             + json.dumps(meta).encode() + b"\r\n"]
    for field, data in images:
        parts.append(b"--" + b + b"\r\n"
                     + f'Content-Disposition: form-data; name="{field}"; filename="{field}.jpg"\r\n'.encode()
                     + b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
    parts.append(b"--" + b + b"--\r\n")
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _pcts(vals):
    if not vals:
        return None, None
    return round(float(np.percentile(vals, 50)), 2), round(float(np.percentile(vals, 95)), 2)


# ---------------------------------------------------------------- sender
class HandoffSender:
    def __init__(self, queue=None, url: Optional[str] = None, save_dir: Optional[str] = None,
                 quality: int = 85, max_width: Optional[int] = None, timeout_s: float = 3.0,
                 retries: int = 2, retry_wait_s: float = 0.5, name_prefix: str = ""):
        """url=None and save_dir=None -> encode only (dry run, used for timing)."""
        self.queue, self.url, self.save_dir = queue, url, save_dir
        self.quality, self.max_width = quality, max_width
        self.timeout_s, self.retries, self.retry_wait_s = timeout_s, retries, retry_wait_s
        self.name_prefix = name_prefix            # e.g. "v001_converted_3_": keeps clips apart on disk
        self.encoded = self.delivered = self.failed = 0
        self.jpeg_ms: List[float] = []
        self.send_ms: List[float] = []
        self.bytes: List[int] = []
        self.last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # ---- one candidate ----
    def prepare(self, cand) -> Tuple[dict, List[Tuple[str, bytes]]]:
        t0 = time.perf_counter()
        images: List[Tuple[str, bytes]] = []
        frames_meta = []
        if cand.frame is not None:
            images.append(("image", encode_jpeg(cand.frame, self.quality, self.max_width)))
            frames_meta.append({"field": "image", "role": "trigger",
                                "stream_time": cand.stream_time})
        for i, ef in enumerate(getattr(cand, "extra_frames", []) or [], start=1):
            img = getattr(ef, "image", ef)
            images.append((f"image_{i}", encode_jpeg(img, self.quality, self.max_width)))
            frames_meta.append({"field": f"image_{i}", "role": "preroll_sharpest",
                                "stream_time": getattr(ef, "stream_time", None)})
        self.jpeg_ms.append((time.perf_counter() - t0) * 1000.0)
        meta = cand.to_json_dict()
        meta.update({"schema": "event-candidate/1", "frames": frames_meta,
                     "jpeg_bytes": [len(d) for _, d in images]})
        if self.name_prefix:
            meta["source"] = self.name_prefix.rstrip("_")
        self.encoded += 1
        self.bytes.append(sum(len(d) for _, d in images))
        return meta, images

    def _save(self, meta, images):
        stem = f"{self.name_prefix}{meta['camera_id']}_{meta['stream_time']:010.2f}_{meta['trigger']}"
        for field, data in images:
            suffix = "" if field == "image" else f"_{field}"
            with open(os.path.join(self.save_dir, f"{stem}{suffix}.jpg"), "wb") as f:
                f.write(data)
        with open(os.path.join(self.save_dir, f"{stem}.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _post(self, meta, images):
        body, ctype = build_multipart(meta, images)
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return 200 <= r.status < 300

    def send_one(self, cand) -> bool:
        """Encode, optionally save a copy, then POST with retries. True if delivered
        (or, in dry-run/save-only mode, if encoded)."""
        meta, images = self.prepare(cand)
        if self.save_dir:
            self._save(meta, images)
        if not self.url:
            return True
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                if self._post(meta, images):
                    self.send_ms.append((time.perf_counter() - t0) * 1000.0)
                    self.delivered += 1
                    return True
                self.last_error = "non-2xx response"
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.last_error = f"{type(e).__name__}: {e}"
            if attempt < self.retries:
                time.sleep(self.retry_wait_s * (2 ** attempt))
        self.failed += 1
        return False

    # ---- worker thread ----
    def start(self):
        if self.queue is None:
            raise ValueError("HandoffSender needs a queue to run as a worker")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            c = self.queue.get(timeout=0.2)
            if c is None:
                if self._stop.is_set():
                    break
                continue
            try:
                self.send_one(c)
            except Exception as e:                 # never let one bad candidate kill the worker
                self.failed += 1
                self.last_error = f"{type(e).__name__}: {e}"

    def stop(self, timeout: float = 10.0):
        """Finish what is already queued, then stop."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def stats(self) -> dict:
        j50, j95 = _pcts(self.jpeg_ms)
        s50, s95 = _pcts(self.send_ms)
        return {"encoded": self.encoded, "delivered": self.delivered, "failed": self.failed,
                "jpeg_ms_p50": j50, "jpeg_ms_p95": j95,
                "send_ms_p50": s50, "send_ms_p95": s95,
                "avg_payload_kb": round(float(np.mean(self.bytes)) / 1024, 1) if self.bytes else None,
                "last_error": self.last_error}


# ---------------------------------------------------------------- test receiver
def make_receiver(port: int = 8000, out_dir: Optional[str] = None, host: str = "0.0.0.0"):
    """A stand-in for Member 2's endpoint. server.received collects every POST."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            raw = (b"Content-Type: " + self.headers["Content-Type"].encode()
                   + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
            msg = message_from_bytes(raw, policy=policy.default)
            meta, imgs = None, {}
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                data = part.get_payload(decode=True)
                if name == "meta":
                    meta = json.loads(data.decode())
                else:
                    imgs[name] = data
            rec = {"meta": meta, "images": imgs}
            self.server.received.append(rec)
            if out_dir and meta:
                stem = f"{meta['camera_id']}_{meta['stream_time']:010.2f}_{meta['trigger']}"
                for name, data in imgs.items():
                    suffix = "" if name == "image" else f"_{name}"
                    with open(os.path.join(out_dir, f"{stem}{suffix}.jpg"), "wb") as f:
                        f.write(data)
                with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
                    json.dump(meta, f, indent=2)
            if meta:
                print(f"[receiver] {meta['camera_id']} t={meta['stream_time']} "
                      f"{meta['trigger']} {meta['category_hint']} | {meta['why']} "
                      f"| {len(imgs)} image(s)", flush=True)
            out = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):                 # quiet
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.received = []
    return server


def _ping(url: str) -> bool:
    from gate_queue import Candidate
    img = np.zeros((240, 320, 3), np.uint8)
    cv2.putText(img, "PING", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    c = Candidate(camera_id="ping", wall_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  stream_time=0.0, trigger="heartbeat", category_hint=None,
                  confidence=0.0, why="link test", frame=img)
    s = HandoffSender(url=url, retries=0)
    ok = s.send_one(c)
    print("OK, delivered" if ok else f"FAILED: {s.last_error}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--receive", action="store_true", help="run a test receiver")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="received")
    ap.add_argument("--ping", metavar="URL", help="send one test candidate")
    a = ap.parse_args()
    if a.ping:
        raise SystemExit(0 if _ping(a.ping) else 1)
    if a.receive:
        srv = make_receiver(a.port, a.out)
        print(f"receiver on http://0.0.0.0:{a.port}/  saving to {a.out}/  (Ctrl+C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    else:
        ap.print_help()
