"""
Blade Ball ball tracker.

Detects the ball in recorded frames using plain color thresholding:
the ball appears as a white/grey disc when idle, and switches to red
when it's targeting you. No trained model needed for this part.

SETUP (run once):
    pip install opencv-python numpy

USAGE:

  1) TUNE the color ranges on one of your own frames first --
     defaults are a starting guess and will likely need adjustment:

       python track_ball.py --tune recordings/session_XXXX/frames/000100.jpg

     A window opens with sliders for the white-ball and red-ball hue
     ranges. Drag them until ONLY the ball is highlighted in the mask
     preview (not chat text, not UI, not other players). Press 's' to
     save the tuned values to color_config.json, or 'q' to quit
     without saving.

  2) RUN detection over a full recorded session:

       python track_ball.py recordings/session_XXXX

     This writes ball_tracks.jsonl into that session folder, one line
     per frame with the detected ball position, state (idle/targeting)
     and apparent radius in pixels (or nulls if not found).

  3) Optionally SPOT-CHECK detections visually:

       python track_ball.py recordings/session_XXXX --debug --debug-every 20

     Saves annotated copies of every 20th frame (with the detected
     ball circled) into session_XXXX/debug_frames/ so you can flip
     through them and confirm detection is actually working.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path("color_config.json")

# Default HSV ranges - starting guesses, tune these on your own footage.
DEFAULT_CONFIG = {
    # White/grey idle ball: low saturation, high brightness
    "white_sat_max": 60,
    "white_val_min": 180,
    # Red targeting ball: red wraps around hue 0, so two ranges
    "red_hue_low_max": 10,
    "red_hue_high_min": 170,
    "red_sat_min": 120,
    "red_val_min": 120,
    # Shape filtering (in pixels / ratio), tune if the ball is very
    # small/large on your screen resolution
    "min_area": 90,
    "max_area": 6000,
    "min_circularity": 0.75,
    # Fractional (0-1) screen regions to ignore entirely -- covers the
    # left-side HUD panel (coins/AFK/skills/quests/emote/level icon),
    # the top stats bar, the bottom BLOCK/ABILITY icons, the bottom-left
    # round timer/lucky-spin badge, and the bottom-right promo banner.
    # The last two are static and red/white-ish, so a lone jump filter
    # can't reject them -- they "confirm" themselves frame after frame
    # since they never move. Tune these if your UI is laid out
    # differently or your window is a different aspect ratio.
    "ui_mask_regions": [
        {"x0": 0.0, "y0": 0.0, "x1": 0.20, "y1": 1.0},    # left HUD panel + bottom timer badge
        {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 0.10},    # top stats bar
        {"x0": 0.38, "y0": 0.70, "x1": 0.62, "y1": 0.95}, # block/ability icons
        {"x0": 0.85, "y0": 0.72, "x1": 1.0, "y1": 1.0},   # bottom-right promo banner
    ],
    # Temporal sanity check: flag a detection if it's implausibly far
    # from the last trusted ball position (could be a VFX flash, or
    # could be the ball genuinely moving fast -- a late-round Blade
    # Ball can easily outrun this per-frame cap). Rather than reject it
    # outright, we check whether the *next* detected frame continues
    # near the new position: real ball movement keeps going, a VFX
    # flash snaps back or disappears. Scales with how many frames have
    # passed since the last trusted detection, and gives up enforcing
    # it after a long gap (round probably restarted / ball was
    # off-screen a while).
    "max_jump_px_per_frame": 120,
    "reset_after_missing_frames": 30,
    "confirm_lookahead_frames": 5,
    # Staleness watchdog: a real ball -- even sitting "idle" -- shouldn't
    # stay at the exact same pixel for seconds on end; that's a stronger
    # sign of having locked onto something static (a map decoration, a
    # standing player, a UI element we haven't masked) than any specific
    # color/shape check could catch, and unlike those checks it doesn't
    # need to know anything about which map or object is at fault. Once
    # the trusted position has gone unchanged (within jitter_px) for this
    # many consecutive frames, proximity is no longer trusted and the next
    # pick is a fresh whole-frame best-candidate search instead.
    "stale_after_frames": 30,
    "stale_jitter_px": 3,
}


def apply_ui_mask(mask, cfg):
    h, w = mask.shape[:2]
    for r in cfg.get("ui_mask_regions", []):
        x0, y0 = int(r["x0"] * w), int(r["y0"] * h)
        x1, y1 = int(r["x1"] * w), int(r["y1"] * h)
        mask[y0:y1, x0:x1] = 0
    return mask


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"Saved {CONFIG_PATH.resolve()}")


def build_masks(hsv, cfg):
    white_mask = cv2.inRange(
        hsv,
        (0, 0, cfg["white_val_min"]),
        (180, cfg["white_sat_max"], 255),
    )
    red_low = cv2.inRange(
        hsv,
        (0, cfg["red_sat_min"], cfg["red_val_min"]),
        (cfg["red_hue_low_max"], 255, 255),
    )
    red_high = cv2.inRange(
        hsv,
        (cfg["red_hue_high_min"], cfg["red_sat_min"], cfg["red_val_min"]),
        (180, 255, 255),
    )
    red_mask = cv2.bitwise_or(red_low, red_high)
    return white_mask, red_mask


def find_ball_candidates(frame_bgr, cfg):
    """Returns every ball-colored/shaped blob in the frame, as a list of
    (circularity, x, y, state, radius) -- not just the single best one.
    radius is the apparent size in pixels (from contour area), which grows
    as the ball gets closer. Color and shape alone can't tell the real ball
    apart from other round pale/red things on screen (another player's
    head, a skill effect), so callers that track the ball over time should
    disambiguate using where it was last seen (closest_to) rather than
    blindly trusting whichever candidate is most circular in isolation."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_mask, red_mask = build_masks(hsv, cfg)

    candidates = []
    for mask, state in ((white_mask, "idle"), (red_mask, "targeting")):
        mask = apply_ui_mask(mask, cfg)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg["min_area"] or area > cfg["max_area"]:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < cfg["min_circularity"]:
                continue
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            radius = (area / np.pi) ** 0.5
            candidates.append((circularity, cx, cy, state, radius))

    return candidates


def closest_to(candidates, point):
    px, py = point
    return min(candidates, key=lambda c: (c[1] - px) ** 2 + (c[2] - py) ** 2)


def most_circular(candidates):
    return max(candidates, key=lambda c: c[0])


# ---------------------------------------------------------------------------
# Tuning mode
# ---------------------------------------------------------------------------

def run_tune(image_path, cfg):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Couldn't read {image_path}")
        return

    win = "tune (s=save, q=quit)"
    cv2.namedWindow(win)

    def nothing(_):
        pass

    cv2.createTrackbar("white_sat_max", win, cfg["white_sat_max"], 255, nothing)
    cv2.createTrackbar("white_val_min", win, cfg["white_val_min"], 255, nothing)
    cv2.createTrackbar("red_hue_low_max", win, cfg["red_hue_low_max"], 60, nothing)
    cv2.createTrackbar("red_hue_high_min", win, cfg["red_hue_high_min"], 180, nothing)
    cv2.createTrackbar("red_sat_min", win, cfg["red_sat_min"], 255, nothing)
    cv2.createTrackbar("red_val_min", win, cfg["red_val_min"], 255, nothing)

    print("Adjust sliders until ONLY the ball glows white in the mask preview.")
    print("Press 's' to save, 'q' to quit without saving.")

    while True:
        for key in (
            "white_sat_max", "white_val_min", "red_hue_low_max",
            "red_hue_high_min", "red_sat_min", "red_val_min",
        ):
            cfg[key] = cv2.getTrackbarPos(key, win)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_mask, red_mask = build_masks(hsv, cfg)
        combined_mask = cv2.bitwise_or(white_mask, red_mask)
        combined_mask = apply_ui_mask(combined_mask, cfg)
        preview = cv2.bitwise_and(img, img, mask=combined_mask)
        stacked = np.hstack([img, preview])
        cv2.imshow(win, stacked)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            save_config(cfg)
            break
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Batch detection over a session
# ---------------------------------------------------------------------------

def run_session(session_dir, cfg, debug, debug_every):
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return

    out_path = session_dir / "ball_tracks.jsonl"
    debug_dir = session_dir / "debug_frames"
    if debug:
        debug_dir.mkdir(exist_ok=True)

    # Pass 1: collect every ball-colored/shaped candidate in each frame, with
    # no temporal reasoning yet. Color/shape alone can't tell the real ball
    # apart from other round pale/red things on screen (another player's
    # head, a skill effect) -- that ambiguity gets resolved in Pass 2 using
    # where the ball was last seen. Images aren't kept around afterward --
    # holding every frame of a full session in memory at once (thousands of
    # 1920x1080 images) would use several GB, so we re-read from disk later
    # for debug output instead.
    raw = [find_ball_candidates(cv2.imread(str(fp)), cfg) for fp in frame_paths]

    # Pass 2: sequentially decide which detection to trust each frame.
    # Whichever candidate is closest to the last trusted position wins, as
    # long as it's within the per-frame allowance -- this is what actually
    # disambiguates the real ball from a same-colored decoy, since the
    # decoy is rarely also the closest thing to where the ball just was.
    # If nothing is close enough, that's treated as a possible big jump
    # (real fast movement, not just noise) and we peek at the next
    # detected frame to see if anything there continues near it before
    # trusting it.
    found_count = 0
    rejected_count = 0
    last_good = None       # (x, y)
    last_good_idx = None   # frame index of last trusted detection
    stale_count = 0        # consecutive accepted frames barely-unchanged in position
    results = [None] * len(frame_paths)  # final (x, y, state, radius) or None, per frame

    for i, candidates in enumerate(raw):
        if not candidates:
            continue

        if last_good is None:
            x, y, state, radius = most_circular(candidates)[1:]
        else:
            elapsed = i - last_good_idx
            if elapsed > cfg["reset_after_missing_frames"] or stale_count >= cfg["stale_after_frames"]:
                x, y, state, radius = most_circular(candidates)[1:]
            else:
                allowed = cfg["max_jump_px_per_frame"] * max(elapsed, 1)
                near = [c for c in candidates
                        if ((c[1] - last_good[0]) ** 2 + (c[2] - last_good[1]) ** 2) ** 0.5 <= allowed]
                if near:
                    x, y, state, radius = closest_to(near, last_good)[1:]
                else:
                    # Nothing near the last trusted position -- could be
                    # real fast movement, could be a decoy. Only trust it
                    # if the next detected frame keeps going near it.
                    cx, cy, cstate, cradius = most_circular(candidates)[1:]
                    confirmed = False
                    lookahead_end = min(i + 1 + cfg["confirm_lookahead_frames"], len(raw))
                    for j in range(i + 1, lookahead_end):
                        nxt = raw[j]
                        if not nxt:
                            continue
                        elapsed2 = j - i
                        allowed2 = cfg["max_jump_px_per_frame"] * max(elapsed2, 1)
                        confirmed = any(
                            ((c[1] - cx) ** 2 + (c[2] - cy) ** 2) ** 0.5 <= allowed2
                            for c in nxt
                        )
                        break  # only the next detected frame counts as confirmation
                    if not confirmed:
                        rejected_count += 1
                        continue  # implausible jump, no confirmation -- drop this frame
                    x, y, state, radius = cx, cy, cstate, cradius

        if last_good is not None and \
                ((x - last_good[0]) ** 2 + (y - last_good[1]) ** 2) ** 0.5 <= cfg["stale_jitter_px"]:
            stale_count += 1
        else:
            stale_count = 0

        found_count += 1
        last_good = (x, y)
        last_good_idx = i
        results[i] = (x, y, state, radius)

    with open(out_path, "w") as out_file:
        for i, fp in enumerate(frame_paths):
            result = results[i]
            x, y, state, radius = result if result else (None, None, None, None)
            out_file.write(json.dumps({
                "frame": fp.name, "ball_x": x, "ball_y": y, "state": state,
                "ball_r": None if radius is None else round(radius, 2),
            }) + "\n")

            if debug and i % debug_every == 0:
                annotated = cv2.imread(str(fp))
                if result:
                    cv2.circle(annotated, (x, y), 12, (0, 255, 0), 2)
                    cv2.putText(annotated, state, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(str(debug_dir / fp.name), annotated)

    print(f"Processed {len(frame_paths)} frames, ball found in {found_count}, "
          f"{rejected_count} rejected as implausible jumps.")
    print(f"Wrote {out_path.resolve()}")
    if debug:
        print(f"Debug frames in {debug_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", help="Path to recordings/session_XXXX")
    parser.add_argument("--tune", metavar="IMAGE", help="Interactively tune color ranges on one frame")
    parser.add_argument("--debug", action="store_true", help="Save annotated debug frames")
    parser.add_argument("--debug-every", type=int, default=20, help="Save every Nth frame when --debug is on")
    args = parser.parse_args()

    cfg = load_config()

    if args.tune:
        run_tune(args.tune, cfg)
        return

    if not args.session:
        parser.error("Provide a session folder, or use --tune IMAGE to tune colors first.")

    run_session(args.session, cfg, args.debug, args.debug_every)


if __name__ == "__main__":
    main()