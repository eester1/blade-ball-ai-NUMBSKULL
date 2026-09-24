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
    # Stricter brightness used only inside white blobs that failed the ball
    # checks, to split the ball out of a pale hazy sky it merged with (barn
    # map: sky brightness 157-197, ball 238). ~95% of tracked white balls in
    # the recordings are at least this bright.
    "white_split_val_min": 215,
    # Red targeting ball: red wraps around hue 0, so two ranges
    "red_hue_low_max": 10,
    "red_hue_high_min": 170,
    "red_sat_min": 120,
    "red_val_min": 120,
    # Shape filtering (in pixels / ratio), tune if the ball is very
    # small/large on your screen resolution
    "min_area": 90,
    # A ball right next to you can be ~100px in radius on a 1080p screen --
    # this has to be big enough not to throw the ball away exactly when it's
    # about to hit.
    "max_area": 130000,
    "min_circularity": 0.75,
    # A glowing ball (the red targeting ball especially) has a ragged,
    # fuzzy edge that inflates its perimeter and sinks the plain
    # circularity score (0.59 for a perfectly round glowing ball seen live).
    # Its convex hull ignores the raggedness, so a blob also counts as round
    # if its hull's circularity reaches this. Real balls score 0.95-0.98
    # here; banner letters, sky patches and explosion flashes 0.83-0.89.
    "min_hull_circularity": 0.92,
    # Blob area / convex-hull area. A ball covers nearly all of its hull
    # (0.95-0.98); letters with a round outline and an opening -- a G or C in
    # an announcement banner -- don't (0.73-0.82), and the solid-fill check
    # below can't catch them because the opening lets their outline follow
    # the stroke.
    "min_solidity": 0.9,
    # Fraction of the blob's outline that's actually filled with ball
    # color. The ball is a solid disc (~1.0); round lettering in red
    # announcement banners ("STANDOFF", "NO ONE WON!") passes the color and
    # circularity checks but is a hollow ring (~0.75-0.85).
    "min_fill": 0.9,
    # A fast ball's glowing trail is pale enough to pass the color filter
    # and merges with the ball into one long blob that fails the round-shape
    # check. For such blobs, the largest solid circle that fits inside is
    # taken as the ball -- if it makes up at least core_min_share of the
    # blob and stands out from its surroundings: at most core_max_ring of a
    # ring just outside it may be ball-colored (a trail touches the ball on
    # one side; a wall, floor or cloud would surround it on all sides).
    "core_min_share": 0.5,
    "core_max_ring": 0.5,
    # Only close (big) balls drag a trail long enough to break the shape
    # check; below this radius, recovering a core just turns irregular
    # specks into extra decoys.
    "core_min_radius": 12,
    # The ball's apparent size changes smoothly -- it can't go from radius 40
    # to 6 in a fifth of a second. A core continues the track only if its
    # radius is within this factor range of the tracked ball's last radius
    # (the ball grows fast as it closes in, but a sky patch or effect usually
    # isn't its size), and for size_continuity_frames after the track was
    # last seen, a far "jump" candidate must be within it too -- otherwise a
    # speck elsewhere takes over the moment a close ball blinks out for a
    # frame or two. Past that window the ball may really have gone far away.
    "core_radius_ratio": [0.5, 2.5],
    "size_continuity_frames": 10,
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
    # When the ball targets you, Blade Ball tints your own character red.
    # The character always sits in about the same place on screen, so this
    # is readable even when the ball itself is off-screen or behind the
    # camera -- and it doesn't depend on the ball's own color being picked
    # up as red. Scored as the fraction of highlight-red pixels in this
    # fractional screen region, kept tight around the character so another
    # player standing next to you (highlighted when *they're* targeted)
    # mostly falls outside it. The highlight is a slightly pinkish red (hue
    # 170-180); orange-red things near your feet -- dirt, lava, a pumpkin
    # head -- sit on the other side of pure red (hue ~5-15) and don't count.
    "self_highlight_roi": {"x0": 0.47, "y0": 0.45, "x1": 0.53, "y1": 0.60},
    "self_highlight_hue_min": 170,
    "self_highlight_sat_min": 120,
    "self_highlight_val_min": 60,
    # Score above which you count as targeted. Real highlights scored
    # 0.15-0.36 in live logs; orange dirt and a highlighted neighbor, <= 0.03.
    # While targeted, a red ball on screen is trusted immediately rather than
    # having to match the previous track: when the camera turns to find the
    # ball, the whole scene sweeps ~300px per frame, so the ball never
    # looks like a continuation of anything and would otherwise be ignored
    # frame after frame while the camera swept right past it.
    "self_target_threshold": 0.08,
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
    """Returns (candidates, cores). candidates are every ball-colored/shaped
    blob in the frame, as (circularity, x, y, state, radius) -- not just the
    single best one; radius is the apparent size in pixels, which grows as
    the ball gets closer. Color and shape alone can't tell the real ball
    apart from other round pale/red things on screen (another player's
    head, a skill effect), so callers that track the ball over time should
    disambiguate using where it was last seen (near_track / closest_to)
    rather than blindly trusting whichever is most circular in isolation.

    cores are round cores recovered from blobs that failed the shape check
    (a close ball merged with its trail -- see core_* config), same format.
    Sky patches and glowing effects can yield convincing cores too, so they
    must only ever be used to *continue* an existing track (near_track),
    never to pick up a new ball."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_mask, red_mask = build_masks(hsv, cfg)

    h, w = frame_bgr.shape[:2]
    roi = cfg["self_highlight_roi"]
    own_x0, own_x1 = roi["x0"] * w, roi["x1"] * w
    own_y0, own_y1 = roi["y0"] * h, roi["y1"] * h

    def on_own_character(x, y):
        # Your own character sits in this fixed region: while you're
        # targeted its red-tinted body passes as a red ball, and its parts
        # (e.g. a white face mask) as a white one. The real ball is beside
        # or above you at contact, not centered on your body.
        return own_x0 <= x <= own_x1 and own_y0 <= y <= own_y1

    def prepare(mask):
        mask = apply_ui_mask(mask, cfg)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    candidates, cores = [], []
    for mask, state in ((white_mask, "idle"), (red_mask, "targeting")):
        mask = prepare(mask)
        failed = []  # blobs that failed the ball checks
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            blob = _blob(c, mask, cfg)
            if blob is None:
                continue
            if blob["ok"]:
                if not on_own_character(blob["x"], blob["y"]):
                    candidates.append((blob["circularity"], blob["x"], blob["y"], state, blob["radius"]))
                continue
            failed.append(c)
            core = _round_core(mask, blob["solid"], blob["bx"], blob["by"], blob["area"], cfg)
            if core is not None and not on_own_character(core[0], core[1]):
                cx, cy, radius = core
                cores.append((cfg["min_circularity"], cx, cy, state, radius))

        # A pale, washed-out sky (or anything bright and grey) also passes the
        # white filter, and a ball in front of it merges into one huge blob
        # that fails the shape check -- on a map with a hazy sky that hides
        # the ball entirely. The ball is still clearly *brighter* than such a
        # background, so look inside failed white blobs again with a stricter
        # brightness cutoff; normal (unmerged) detection is left as is.
        if state == "idle" and failed:
            strict = prepare(cv2.inRange(
                hsv, (0, 0, cfg["white_split_val_min"]), (180, cfg["white_sat_max"], 255)))
            for c in cv2.findContours(strict, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                blob = _blob(c, strict, cfg)
                if blob is None or not blob["ok"] or on_own_character(blob["x"], blob["y"]):
                    continue
                if any(cv2.pointPolygonTest(f, (blob["x"], blob["y"]), False) >= 0 for f in failed):
                    candidates.append((blob["circularity"], blob["x"], blob["y"], state, blob["radius"]))

    return candidates, cores


def _blob(contour, mask, cfg):
    """Shape measurements for one contour, and whether it passes as a ball
    ("ok"); None if it's too small to consider."""
    area = cv2.contourArea(contour)
    if area < cfg["min_area"]:
        return None
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    hull_circularity = 4 * np.pi * hull_area / (hull_perimeter * hull_perimeter)
    round_enough = (circularity >= cfg["min_circularity"] or
                    hull_circularity >= cfg["min_hull_circularity"]) \
        and area / hull_area >= cfg["min_solidity"]
    bx, by, bw, bh = cv2.boundingRect(contour)
    outline = np.zeros((bh, bw), np.uint8)
    cv2.drawContours(outline, [contour], -1, 255, -1, offset=(-bx, -by))
    solid = cv2.bitwise_and(mask[by:by + bh, bx:bx + bw], outline)
    inside = cv2.countNonZero(outline)
    fill = cv2.countNonZero(solid) / inside if inside else 0.0
    M = cv2.moments(contour)
    return {
        "ok": round_enough and fill >= cfg["min_fill"] and area <= cfg["max_area"],
        "x": int(M["m10"] / M["m00"]), "y": int(M["m01"] / M["m00"]),
        "circularity": max(circularity, hull_circularity),
        "radius": (area / np.pi) ** 0.5,
        "area": area, "solid": solid, "bx": bx, "by": by,
    }


def _round_core(mask, solid, bx, by, blob_area, cfg):
    """Largest solid circle inside a blob that failed the round-shape check,
    as (x, y, radius) -- or None if it isn't ball-like (see core_* config)."""
    padded = cv2.copyMakeBorder(solid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    _, radius, _, (px, py) = cv2.minMaxLoc(dist)
    disc_area = np.pi * radius * radius
    if radius < cfg["core_min_radius"] or disc_area < cfg["core_min_share"] * blob_area \
            or disc_area > cfg["max_area"]:
        return None

    cx, cy = bx + px - 1, by + py - 1
    h, w = mask.shape
    outer = int(np.ceil(radius * 1.5))
    x0, y0 = max(cx - outer, 0), max(cy - outer, 0)
    x1, y1 = min(cx + outer + 1, w), min(cy + outer + 1, h)
    ring = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(ring, (cx - x0, cy - y0), outer, 255, -1)
    cv2.circle(ring, (cx - x0, cy - y0), int(radius * 1.2), 0, -1)
    ring_px = cv2.countNonZero(ring)
    if ring_px == 0:
        return None
    covered = cv2.countNonZero(cv2.bitwise_and(mask[y0:y1, x0:x1], ring)) / ring_px
    if covered > cfg["core_max_ring"]:
        return None
    return cx, cy, radius


def self_highlight_score(frame_bgr, cfg):
    """Fraction (0-1) of strongly red pixels around your own character --
    high when the ball is targeting you (see self_highlight_roi)."""
    h, w = frame_bgr.shape[:2]
    r = cfg["self_highlight_roi"]
    roi = frame_bgr[int(r["y0"] * h):int(r["y1"] * h), int(r["x0"] * w):int(r["x1"] * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (cfg["self_highlight_hue_min"], cfg["self_highlight_sat_min"], cfg["self_highlight_val_min"]),
        (180, 255, 255),
    )
    return float(mask.mean() / 255)


def near_track(candidates, cores, point, radius, max_dist, cfg):
    """Candidates within max_dist of the tracked ball's position, plus cores
    that are near it *and* about the tracked ball's size -- the only way a
    core is ever trusted (see find_ball_candidates)."""
    px, py = point

    def close(c):
        return ((c[1] - px) ** 2 + (c[2] - py) ** 2) ** 0.5 <= max_dist

    return [c for c in candidates if close(c)] + \
        size_consistent([c for c in cores if close(c)], radius, cfg)


def size_consistent(candidates, radius, cfg):
    """Candidates whose radius is plausible for the same ball as `radius`."""
    lo, hi = cfg["core_radius_ratio"]
    return [c for c in candidates if lo <= c[4] / radius <= hi]


def targeting_ball(candidates, self_red, cfg):
    """While you're highlighted as targeted, the most ball-like red
    candidate on screen, else None (see self_target_threshold)."""
    if self_red < cfg["self_target_threshold"]:
        return None
    red = [c for c in candidates if c[3] == "targeting"]
    return most_circular(red) if red else None


def prefer_red(near, red_target, point):
    """While you're targeted (red_target not None), a red ball beats whatever
    else is being tracked -- otherwise a stuck white decoy keeps "continuing"
    itself while the red ball coming at you is ignored. Returns
    (candidate, continued_existing_track), or None when not targeted."""
    if red_target is None:
        return None
    near_red = [c for c in near if c[3] == "targeting"]
    if near_red:
        return closest_to(near_red, point), True
    return red_target, False


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
    raw, self_red = [], []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        raw.append(find_ball_candidates(img, cfg))
        self_red.append(self_highlight_score(img, cfg))

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
    last_radius = None     # apparent radius of the last trusted detection
    last_good_idx = None   # frame index of last trusted detection
    stale_count = 0        # consecutive accepted frames barely-unchanged in position
    results = [None] * len(frame_paths)  # final (x, y, state, radius) or None, per frame

    for i, (candidates, cores) in enumerate(raw):
        fresh = last_good is None or i - last_good_idx > cfg["reset_after_missing_frames"] \
            or stale_count >= cfg["stale_after_frames"]
        red_target = targeting_ball(candidates, self_red[i], cfg)
        if fresh:
            if not candidates:
                continue
            x, y, state, radius = (red_target or most_circular(candidates))[1:]
        else:
            elapsed = i - last_good_idx
            allowed = cfg["max_jump_px_per_frame"] * max(elapsed, 1)
            near = near_track(candidates, cores, last_good, last_radius, allowed, cfg)
            choice = prefer_red(near, red_target, last_good)
            if elapsed <= cfg["size_continuity_frames"]:
                candidates = size_consistent(candidates, last_radius, cfg)
            if choice is not None:
                x, y, state, radius = choice[0][1:]
            elif near:
                x, y, state, radius = closest_to(near, last_good)[1:]
            elif not candidates:
                continue
            else:
                # Nothing near the last trusted position -- could be
                # real fast movement, could be a decoy. Only trust it
                # if the next detected frame keeps going near it.
                cx, cy, cstate, cradius = most_circular(candidates)[1:]
                confirmed = False
                lookahead_end = min(i + 1 + cfg["confirm_lookahead_frames"], len(raw))
                for j in range(i + 1, lookahead_end):
                    nxt = raw[j][0]
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
        last_radius = radius
        last_good_idx = i
        results[i] = (x, y, state, radius)

    with open(out_path, "w") as out_file:
        for i, fp in enumerate(frame_paths):
            result = results[i]
            x, y, state, radius = result if result else (None, None, None, None)
            out_file.write(json.dumps({
                "frame": fp.name, "ball_x": x, "ball_y": y, "state": state,
                "ball_r": None if radius is None else round(radius, 2),
                "self_red": round(self_red[i], 4),
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