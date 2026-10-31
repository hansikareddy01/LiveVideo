"""Run capture metrics using the matching frame_source.py.

From your project root:
    py run_metrics.py
    py run_metrics.py --video "videos/v001_converted 3.avi"
    py run_metrics.py --all

Writes JSON and CSV summaries into data/metrics/.
"""
import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import cv2
from frame_source import FrameSource, pick_sharpest

DEFAULT_VIDEO = "videos/v026_converted 3.avi"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def measure_video(video_path, camera_id, target_fps, max_seconds):
    source = FrameSource(str(video_path), camera_id=camera_id, fps=target_fps)
    frame_count = 0
    sharpest = []
    first_timestamp = None
    last_timestamp = None
    started = time.perf_counter()
    error = None

    try:
        for frame in source.frames():
            frame_count += 1
            if first_timestamp is None:
                first_timestamp = frame.wall_time
            last_timestamp = frame.wall_time
            # Keep only a small list of frames for the sharpness sample.
            sharpest.append(frame)
            if len(sharpest) > 50:
                sharpest = pick_sharpest(sharpest, k=10)
            if max_seconds > 0 and frame.stream_time >= max_seconds:
                source.close()
                break
    except (OSError, RuntimeError, cv2.error) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        source.close()

    elapsed = time.perf_counter() - started
    stats = source.stats()
    best = pick_sharpest(sharpest, k=1)
    result = {
        "video": str(video_path),
        "camera_id": camera_id,
        "status": "ok" if frame_count and not error else "error",
        "error": error,
        "target_fps": target_fps,
        "frames_kept": frame_count,
        "first_wall_time": first_timestamp,
        "last_wall_time": last_timestamp,
        "processing_wall_seconds": round(elapsed, 3),
        "average_processing_fps": round(frame_count / elapsed, 2) if elapsed else None,
        "capture_ms_p50": stats.get("capture_ms_p50"),
        "capture_ms_p95": stats.get("capture_ms_p95"),
        "source_stats": stats,
        "sharpest_frame_time_seconds": best[0].stream_time if best else None,
        "sharpest_frame_score": best[0].sharpness if best else None,
    }
    print(f"\nVideo: {video_path}")
    print(f"Status: {result['status']}")
    if error:
        print(f"Error: {error}")
    print(f"Frames kept: {frame_count}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Capture p50/p95: {result['capture_ms_p50']} / {result['capture_ms_p95']} ms")
    print(f"Source stats: {json.dumps(stats, ensure_ascii=False)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Measure frame-source capture performance")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help=f"Video path (default: {DEFAULT_VIDEO})")
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--seconds", type=float, default=0, help="Optional max stream seconds; 0 means full video")
    parser.add_argument("--all", action="store_true", help="Run all supported videos in the videos folder")
    args = parser.parse_args()

    videos_dir = Path("videos")
    if args.all:
        video_paths = sorted(p for p in videos_dir.glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
        if not video_paths:
            parser.error(f"No supported video files found in {videos_dir.resolve()}")
    else:
        video_paths = [Path(args.video)]
        if not video_paths[0].is_file():
            parser.error(f"Video not found: {video_paths[0]!s}. Check the path or use --video.")

    output_dir = Path("data") / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, path in enumerate(video_paths, start=1):
        camera_id = args.camera if not args.all else f"cam{index}"
        results.append(measure_video(path, camera_id, args.fps, args.seconds))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"metrics_{timestamp}.json"
    csv_path = output_dir / f"metrics_{timestamp}.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    columns = [
        "video", "camera_id", "status", "error", "target_fps", "frames_kept",
        "first_wall_time", "last_wall_time", "processing_wall_seconds",
        "average_processing_fps", "capture_ms_p50", "capture_ms_p95",
        "sharpest_frame_time_seconds", "sharpest_frame_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in columns})

    print(f"\nSaved JSON report: {json_path.resolve()}")
    print(f"Saved CSV report:  {csv_path.resolve()}")


if __name__ == "__main__":
    main()
