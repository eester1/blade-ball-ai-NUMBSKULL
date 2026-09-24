"""
Merge recorded inputs (inputs.jsonl) and ball detections (ball_tracks.jsonl)
from one or more sessions into a single CSV ready for training.

Each output row is one frame with:
  - session, frame, t          (bookkeeping)
  - ball_x, ball_y, ball_state (raw detection)
  - every column in features.FEATURE_COLUMNS -- screen position relative
    to center, velocity in px/s, distance, closing speed, targeting flag,
    and the depth cues (apparent radius, its growth rate, time-to-contact).
    Computed by the same FeatureTracker that play_live.py uses, so the
    training features and live features can't drift apart.
  - one label column per action in LABELS (1 if held that frame, else 0)

Frames where the ball wasn't detected are dropped entirely -- there's
nothing useful to learn from a frame with no ball position, and we'd
rather have clean gaps than made-up data.

USAGE:
    python build_dataset.py recordings/session_A recordings/session_B ... -o dataset.csv

    (Run track_ball.py on each session first -- this script expects
    ball_tracks.jsonl, including the ball_r radius field, in each one.)
"""

import argparse
import csv
import json
from pathlib import Path

from features import FeatureTracker

# Labels are defined by what an action *does* in-game, not by which
# physical input did it: blocking works with either left click or F, so
# both count as a block. Keys here must match what record_gameplay.py
# records. Right mouse isn't a label on purpose -- in Roblox, holding it
# and dragging rotates the camera, so those frames are camera movement,
# not an action (play_live.py controls the camera separately).
LABELS = {
    "held_w": {"w"},
    "held_a": {"a"},
    "held_s": {"s"},
    "held_d": {"d"},
    "held_block": {"mouse_left", "f"},
    "held_ability": {"q"},
}


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
    first_frame = next((session_dir / "frames").glob("*.jpg"), None)
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

    tracks = {t["frame"]: t for t in load_jsonl(tracks_path)}
    if any(t.get("ball_x") is not None and "ball_r" not in t for t in tracks.values()):
        print(f"Skipping {session_dir}: ball_tracks.jsonl predates radius tracking "
              f"(re-run track_ball.py on it).")
        return []

    size = get_frame_size(session_dir)
    if size is None:
        print(f"Skipping {session_dir}: no frame images found.")
        return []
    center_x, center_y = size[0] / 2, size[1] / 2

    tracker = FeatureTracker()
    rows = []
    for inp in load_jsonl(inputs_path):
        track = tracks.get(inp["frame"])
        if track is None or track.get("ball_x") is None:
            continue  # ball not detected this frame -- drop it

        features = tracker.update(
            track["ball_x"], track["ball_y"], track["state"], track["ball_r"],
            inp["t"], center_x, center_y,
        )
        held = set(inp.get("held", []))
        row = {
            "session": session_dir.name,
            "frame": inp["frame"],
            "t": inp["t"],
            "ball_x": track["ball_x"],
            "ball_y": track["ball_y"],
            "ball_state": track["state"],
            **features,
        }
        for label, inputs in LABELS.items():
            row[label] = 1 if held & inputs else 0
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

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} total rows to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
