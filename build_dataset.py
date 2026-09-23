"""
Merge recorded inputs (inputs.jsonl) and ball detections (ball_tracks.jsonl)
from one or more sessions into a single CSV ready for training.

Each output row is one frame with:
  - session, frame, t          (bookkeeping)
  - ball_x, ball_y             (raw pixel position)
  - ball_rel_x, ball_rel_y     (position relative to screen center --
                                 a rough proxy for "relative to player",
                                 since the camera generally stays centered
                                 on your character)
  - ball_vel_x, ball_vel_y     (pixels PER SECOND since the previous
                                 frame that had a detected ball -- 0 for
                                 the first detected frame of a session.
                                 Real-time units instead of pixels/frame
                                 so the feature stays correct even if the
                                 loop that captured it wasn't running at
                                 a perfectly steady rate.)
  - ball_state                 ("idle" or "targeting")
  - held_<action>              one column per tracked key/button, 1 if
                                 held at that frame, else 0

Frames where the ball wasn't detected are dropped entirely -- there's
nothing useful to learn from a frame with no ball position, and we'd
rather have clean gaps than made-up data.

USAGE:
    python build_dataset.py recordings/session_A recordings/session_B ... -o dataset.csv

    (Run track_ball.py on each session first if you haven't already --
    this script expects ball_tracks.jsonl to already exist in each one.)
"""

import argparse
import csv
import json
from pathlib import Path

# Must match the actions tracked in record_gameplay.py
ACTIONS = ["w", "a", "s", "d", "mouse_left", "mouse_right"]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_frame_size(session_dir):
    """Peek at one frame image to get width/height for the relative-position feature."""
    from PIL import Image
    frames_dir = session_dir / "frames"
    first_frame = next(frames_dir.glob("*.jpg"), None)
    if first_frame is None:
        return None
    with Image.open(first_frame) as img:
        return img.size  # (width, height)


def build_rows(session_dir):
    session_dir = Path(session_dir)
    inputs_path = session_dir / "inputs.jsonl"
    tracks_path = session_dir / "ball_tracks.jsonl"

    if not inputs_path.exists() or not tracks_path.exists():
        print(f"Skipping {session_dir}: missing inputs.jsonl or ball_tracks.jsonl "
              f"(run record_gameplay.py and track_ball.py first).")
        return []

    inputs = load_jsonl(inputs_path)
    tracks = {t["frame"]: t for t in load_jsonl(tracks_path)}
    size = get_frame_size(session_dir)
    if size is None:
        print(f"Skipping {session_dir}: no frame images found.")
        return []
    width, height = size
    center_x, center_y = width / 2, height / 2

    # Frames aren't perfectly evenly spaced (capture jitter, dropped
    # frames), so a dt below this is almost certainly a timestamp glitch
    # rather than a real near-instantaneous re-detection -- treat it as
    # "no time passed" and fall back to zero velocity rather than dividing
    # by a near-zero number and producing a huge bogus spike.
    MIN_DT = 1e-3

    rows = []
    prev_x = prev_y = prev_t = None

    for inp in inputs:
        frame = inp["frame"]
        track = tracks.get(frame)
        if track is None or track.get("ball_x") is None:
            continue  # ball not detected this frame -- drop it

        x, y, t = track["ball_x"], track["ball_y"], inp["t"]
        dt = None if prev_t is None else t - prev_t
        if dt is None or dt < MIN_DT:
            vel_x = vel_y = 0
        else:
            vel_x = (x - prev_x) / dt
            vel_y = (y - prev_y) / dt
        prev_x, prev_y, prev_t = x, y, t

        rel_x, rel_y = x - center_x, y - center_y
        distance = (rel_x ** 2 + rel_y ** 2) ** 0.5
        # Positive = ball closing in on you, negative = moving away.
        # (dot of velocity with the direction from ball to screen center)
        closing_speed = 0.0 if distance < 1e-6 else -(rel_x * vel_x + rel_y * vel_y) / distance

        held = set(inp.get("held", []))
        row = {
            "session": session_dir.name,
            "frame": frame,
            "t": inp["t"],
            "ball_x": x,
            "ball_y": y,
            "ball_rel_x": rel_x,
            "ball_rel_y": rel_y,
            "ball_vel_x": vel_x,
            "ball_vel_y": vel_y,
            "ball_distance": distance,
            "ball_closing_speed": closing_speed,
            "ball_state": track["state"],
        }
        for action in ACTIONS:
            row[f"held_{action}"] = 1 if action in held else 0
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+", help="One or more recordings/session_XXXX folders")
    parser.add_argument("-o", "--output", default="dataset.csv")
    args = parser.parse_args()

    all_rows = []
    for session in args.sessions:
        rows = build_rows(session)
        print(f"{session}: {len(rows)} usable frames")
        all_rows.extend(rows)

    if not all_rows:
        print("No usable rows -- nothing written.")
        return

    fieldnames = list(all_rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} total rows to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()