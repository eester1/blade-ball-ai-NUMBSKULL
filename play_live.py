"""
Live AI play for Blade Ball.

Watches your screen in real time, detects the ball (reusing track_ball.py's
detection code), builds the same features training used (features.py),
feeds them into your trained model, and presses/releases the predicted
actions for you. It also turns the camera to keep the ball in view: the
model can only react to a ball it can see.

SETUP (run once, if you haven't already):
    pip install mss pynput joblib pandas scikit-learn opencv-python numpy

USAGE:
    python play_live.py model.joblib

    - Press Insert to toggle AI control ON/OFF. Starts OFF -- nothing
      happens until you turn it on.
    - Press End to quit immediately. This always releases every key/
      button first, so nothing gets left stuck held down.

    Camera control (see CAMERA_* settings below):
    - --camera keys   (default) turns with the Left/Right arrow keys,
                      Roblox's built-in camera rotation keys.
    - --camera mouse  turns by holding right mouse and dragging, sent as
                      raw relative mouse input. Try this if arrow keys
                      don't rotate the camera in Blade Ball.
    - --camera off    never touches the camera.
    - --camera-invert flips the turn direction, if it turns the wrong way.
    - --camera-test   turns left for a second, then right, and exits --
                      a quick check that the camera method and direction
                      are right before letting the AI play.

    Diagnostics: --log (a new timestamped file in live_logs/ each run) or
    --log session.jsonl (a name you choose) writes one JSON line per frame the
    AI acted on (features, model confidences, actions, camera state) and
    saves each captured frame into session_frames/.

SAFETY: while ON, this sends REAL key/mouse input to whatever window is
focused. Keep Roblox focused and your hand near the keyboard the first
few times you try it, in case it does something you don't want -- End
stops everything instantly.
"""

import argparse
import ctypes
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import joblib
import mss
import numpy as np
from pynput import keyboard, mouse

import ball_classifier
import track_ball
from features import FeatureTracker

# Target capture rate -- caps how often the screen is polled. Velocity is
# measured in pixels/second from real timestamps, so this doesn't have to
# match the recording FPS exactly.
FPS = 15

# How confident the model needs to be (0-1) before actually taking each
# action. Blocking is a bit lower: missing a real block (death) is worse
# than an unnecessary one, and since blocks are re-tapped (below) an early
# one no longer costs the chance to block again. On held-out data 0.4 has
# an F1 as good as 0.5's (0.38 vs 0.37) and catches more of your real
# blocks (recall 0.43 vs 0.36).
THRESHOLDS = {
    "held_w": 0.5,
    "held_a": 0.5,
    "held_s": 0.5,
    "held_d": 0.5,
    "held_block": 0.4,
    "held_ability": 0.5,
}

# How each model label is carried out. Block uses left click (verified to
# block live); F would work too.
KEY_ACTIONS = {"w": "w", "a": "a", "s": "s", "d": "d", "ability": "q"}
MOUSE_ACTIONS = {"block": mouse.Button.left}

# Actions that only do something at the moment they're pressed -- holding
# them doesn't repeat them. Blade Ball's block triggers once per press, so
# holding it from an early press means it never fires again when the ball
# actually arrives. These are tapped instead, and re-tapped at most this
# often (seconds) for as long as the model keeps wanting them.
TAP_ACTIONS = {"block": 0.35}
TAP_DOWN_S = 0.03  # how long a tap holds the button, so the game registers it

# Blocking is gated on the ball being close, because timing is what the
# model gets wrong in both directions: it tapped with the ball ~1.1s away
# (block used up, then died), and right at contact it often *doesn't* want
# to block, since in your recordings you'd already pressed earlier. So:
# - a block only goes through once the ball is close to your character.
#   Whatever direction it comes from, a ball reaching you ends up at your
#   character's spot on screen: every successful live block was 121-191px
#   away, every contact 113-195px. But screen distance alone isn't enough
#   -- a ball far away in front of you sits just above your character on
#   screen (practice mode: ~190px away at radius 8-10, and it spammed block
#   for 3s) -- so it also has to look big enough to be near: radius at
#   least BLOCK_MIN_RADIUS (every contact seen was >= 15). And a ball
#   coming at you head-on stays up the screen until the last instant, so a
#   *big* ball (>= BLOCK_BIG_RADIUS) counts from BLOCK_BIG_MAX_CHAR_DIST.
# - once it's that close, block if the model wants to, *or* if you're
#   targeted and the ball is red -- that combination alone is enough.
# Raise the distances if it blocks too late, lower them if too early.
BLOCK_MAX_CHAR_DIST = 200
BLOCK_MIN_RADIUS = 14
BLOCK_BIG_RADIUS = 45
BLOCK_BIG_MAX_CHAR_DIST = 320

# Your character's red "targeted" tint can blink off while the ball is
# still coming. For this long after last seeing it, keep counting as
# targeted while the tracked ball is still red, instead of the model
# suddenly deciding the danger has passed -- but not once a block has been
# tapped since (then the tint going away most likely means it worked).
TARGET_LATCH_S = 1.5

TOGGLE_KEY = keyboard.Key.insert
QUIT_KEY = keyboard.Key.end

# If the ball goes undetected for this many consecutive frames, release
# every held action as a safety net (probably out of a round, or lost).
MISSING_FRAMES_RESET = 30

# --- Camera control ---------------------------------------------------------
# Only steer toward a detection that's been followed smoothly for at least
# this many consecutive frames. In live testing, most camera swings away
# from the real ball started on a detection that had just jumped hundreds
# of pixels -- a decoy -- rather than on a steady track.
CAMERA_MIN_TRACK_FRAMES = 3
# Turn toward the ball only once it's past this fraction of the way from
# screen center to the left/right edge. The dead zone in the middle keeps
# the ball's on-screen position meaningful to the model (which learned
# from your recordings, where the ball wasn't pinned to center), while
# still stopping it from drifting off-screen.
CAMERA_EDGE_ZONE = 0.5
# Turn duration per unit of distance past the edge zone, and its bounds.
CAMERA_TURN_GAIN_S = 0.3
CAMERA_MIN_TURN_S = 0.04
CAMERA_MAX_TURN_S = 0.15
# Once the ball has been missing this many frames, sweep toward the side
# it was last seen on to find it again...
CAMERA_SEARCH_AFTER_FRAMES = 8
CAMERA_SEARCH_TURN_S = 0.12
# ...but stop after this long: it's probably between rounds, not lost.
CAMERA_SEARCH_MAX_S = 6.0
# Drag speed in "mouse" mode, in mouse counts per second.
CAMERA_MOUSE_SPEED = 900
# Mouse mode: once the dragged cursor is this many pixels from the anchor,
# hop it back (see Camera.update) so it never leaves the game screen.
CAMERA_REANCHOR_PX = 250


def should_block(model_wants, ball, targeted, character_xy):
    """Final block decision for this frame (see BLOCK_* above)."""
    x, y, state, radius = ball
    distance = ((x - character_xy[0]) ** 2 + (y - character_xy[1]) ** 2) ** 0.5
    close = (distance <= BLOCK_MAX_CHAR_DIST and radius >= BLOCK_MIN_RADIUS) or \
        (distance <= BLOCK_BIG_MAX_CHAR_DIST and radius >= BLOCK_BIG_RADIUS)
    return close and (model_wants or (targeted and state == "targeting"))


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", _MOUSEINPUT)]


def move_mouse_relative(dx, dy):
    """Raw relative mouse movement via SendInput. Roblox rotates the camera
    from raw mouse deltas while right mouse is held; moving the cursor to an
    absolute position (what pynput does) isn't reliably seen as a drag."""
    if sys.platform != "win32":
        raise RuntimeError("--camera mouse needs Windows (SendInput)")
    inp = _INPUT(type=0, mi=_MOUSEINPUT(dx, dy, 0, 0x0001, 0, 0))  # MOUSEEVENTF_MOVE
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


class Camera:
    """Non-blocking camera turning: turn() starts or extends a timed turn,
    update() (called every loop iteration) keeps it going and ends it."""

    def __init__(self, method, invert, kb, ms, anchor=None):
        self.method = method
        self.invert = invert
        self.kb = kb
        self.ms = ms
        # Mouse mode: where the cursor is kept -- the game screen's center if
        # given, else wherever the cursor was when a turn started.
        self.anchor = anchor
        self.direction = 0      # -1 = turning left, +1 = right, 0 = idle
        self.until = 0.0
        self.last_active = 0.0  # last time the view was moving
        self._last_move = None  # mouse mode: time of the last drag step
        self._anchor = None     # mouse mode: anchor for the current turn

    def turn(self, direction, duration, now):
        if self.method == "off":
            return
        if self.invert:
            direction = -direction
        if direction != self.direction:
            self.stop()
            self._press(direction, now)
            self.direction = direction
            self.until = now + duration
        else:
            self.until = max(self.until, now + duration)
        self.last_active = now

    def update(self, now):
        if self.direction == 0:
            return
        if self.method == "mouse":
            dx = int(self.direction * CAMERA_MOUSE_SPEED * (now - self._last_move))
            if dx:
                move_mouse_relative(dx, 0)
            self._last_move = now
            # The drag really moves the cursor, which would otherwise wander
            # off the game screen (onto another monitor, where a block click
            # would land in some other window). Hop it back to the anchor
            # with right mouse released, so the hop doesn't turn the camera.
            x, y = self.ms.position
            if abs(x - self._anchor[0]) > CAMERA_REANCHOR_PX or \
                    abs(y - self._anchor[1]) > CAMERA_REANCHOR_PX:
                self.ms.release(mouse.Button.right)
                self.ms.position = self._anchor
                self.ms.press(mouse.Button.right)
        self.last_active = now
        if now >= self.until:
            self.stop()

    def stop(self):
        if self.direction == 0:
            return
        if self.method == "keys":
            self.kb.release(keyboard.Key.left if self.direction < 0 else keyboard.Key.right)
        elif self.method == "mouse":
            self.ms.release(mouse.Button.right)
            self.ms.position = self._anchor
        self.direction = 0

    def _press(self, direction, now):
        if self.method == "keys":
            self.kb.press(keyboard.Key.left if direction < 0 else keyboard.Key.right)
        elif self.method == "mouse":
            self._anchor = self.anchor or self.ms.position
            self.ms.position = self._anchor
            self.ms.press(mouse.Button.right)
            self._last_move = now


class AIController:
    def __init__(self, model_path, camera_method="keys", camera_invert=False, log_path=None,
                 use_classifier=True):
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.label_columns = data["label_columns"]
        unknown = [label for label in self.label_columns
                   if label.replace("held_", "") not in KEY_ACTIONS | MOUSE_ACTIONS]
        if unknown:
            raise SystemExit(f"{model_path} was trained with an older label set {unknown} this "
                             f"version can't act on -- rebuild the dataset and retrain:\n"
                             f"  python build_dataset.py recordings/*/ -o dataset.csv\n"
                             f"  python train_model.py dataset.csv -o model.joblib")

        self.cfg = track_ball.load_config()
        # Learned "is this blob really the ball?" filter, if it's been trained.
        self.scorer = ball_classifier.load_scorer() if use_classifier else None
        self.kb = keyboard.Controller()
        self.ms = mouse.Controller()
        self.camera = Camera(camera_method, camera_invert, self.kb, self.ms)

        self.enabled = False
        self.quit = False
        self._lock = threading.Lock()

        self.currently_held = set()
        self.features = FeatureTracker()
        self.last_feature_t = 0.0
        self.prev_ball = None     # (x, y, radius) of the last trusted detection
        self.last_seen_side = 1   # -1 = ball last seen left of center, +1 = right
        self.missing_streak = 0
        self.missing_since = None
        self.stale_count = 0      # consecutive frames barely-unchanged in position
        self.pending = None       # candidate awaiting next-frame confirmation
        self.track_len = 0        # consecutive frames the ball was followed smoothly
        self.self_red = 0.0       # latest self-highlight score (see track_ball)
        self.targeted_at = None   # (time, score) of the last frame you were highlighted

        self.last_tap = {}  # tap action -> time.time() it was last tapped

        self._sct = mss.mss()

        self._log_file = open(log_path, "w") if log_path else None
        self._log_frame_idx = 0
        self._log_frames_dir = None
        if log_path:
            self._log_frames_dir = Path(log_path).parent / (Path(log_path).stem + "_frames")
            self._log_frames_dir.mkdir(parents=True, exist_ok=True)

    # --- hotkeys -----------------------------------------------------

    def on_key_press(self, key):
        if key == TOGGLE_KEY:
            with self._lock:
                self.enabled = not self.enabled
                state = self.enabled
                # Start fresh -- tracking state from before a pause is stale
                # (and would block the camera search from kicking in).
                self.missing_streak = 0
                self.prev_ball = None
                self.pending = None
                self.targeted_at = None
            print(f"[AI {'ENABLED' if state else 'disabled'}]")
            if not state:
                self.release_all()
            return
        if key == QUIT_KEY:
            with self._lock:
                self.quit = True
            self.release_all()
            print("[quitting]")
            return False

    # --- action press/release -----------------------------------------

    def press_action(self, action):
        if action in KEY_ACTIONS:
            self.kb.press(KEY_ACTIONS[action])
        elif action in MOUSE_ACTIONS:
            self.ms.press(MOUSE_ACTIONS[action])

    def release_action(self, action):
        if action in KEY_ACTIONS:
            self.kb.release(KEY_ACTIONS[action])
        elif action in MOUSE_ACTIONS:
            self.ms.release(MOUSE_ACTIONS[action])

    def release_all(self):
        for action in list(self.currently_held):
            self.release_action(action)
        self.currently_held = set()
        self.camera.stop()

    def apply_actions(self, desired):
        """Holds/releases held actions to match `desired`, and taps any tap
        action in it that's due. Returns the set of actions tapped."""
        now = time.time()
        tapped = set()
        for action in desired & TAP_ACTIONS.keys():
            if now - self.last_tap.get(action, 0) >= TAP_ACTIONS[action]:
                self.press_action(action)
                time.sleep(TAP_DOWN_S)
                self.release_action(action)
                self.last_tap[action] = now
                tapped.add(action)

        held = desired - TAP_ACTIONS.keys()
        for action in held - self.currently_held:
            self.press_action(action)
        for action in self.currently_held - held:
            self.release_action(action)
        self.currently_held = held
        return tapped

    # --- ball tracking -------------------------------------------------

    def select_ball(self, candidates, cores):
        """Pick the real ball out of this frame's candidates (or, to continue
        an existing track, its trail-merged cores), or None. Mirrors
        track_ball.py's batch tracker, except that live play can't peek at
        the next frame -- a big jump is held as pending for one frame and
        only trusted if the next frame confirms it. Uses self.self_red, so
        compute that for the frame first."""
        # A real ball shouldn't sit at the exact same pixel for seconds --
        # if it has, stop trusting proximity to that spot (it's more likely
        # a static decoy: a map decoration, a standing player, an unmasked
        # UI element) and force a fresh whole-frame search instead.
        stale = self.stale_count >= self.cfg["stale_after_frames"]
        red_target = track_ball.targeting_ball(candidates, self.self_red, self.cfg)
        if self.prev_ball is None or stale:
            self.pending = None
            self.track_len = 0
            if red_target is not None:
                return red_target[1:]
            return track_ball.most_circular(candidates)[1:] if candidates else None

        px, py, pr = self.prev_ball
        max_dist = self.cfg["max_jump_px_per_frame"] * max(1, self.missing_streak + 1)
        near = track_ball.near_track(candidates, cores, (px, py), pr, max_dist, self.cfg)
        choice = track_ball.prefer_red(near, red_target, (px, py))
        if choice is not None:
            ball, continued = choice
            self.pending = None
            self.track_len = self.track_len + 1 if continued else 0
            return ball[1:]
        if near:
            self.pending = None
            self.track_len += 1
            return track_ball.closest_to(near, (px, py))[1:]
        if self.missing_streak < self.cfg["size_continuity_frames"]:
            candidates = track_ball.size_consistent(candidates, pr, self.cfg)
        if not candidates:
            return None

        best = track_ball.most_circular(candidates)[1:]
        if self.pending is not None and \
                ((best[0] - self.pending[0]) ** 2 + (best[1] - self.pending[1]) ** 2) ** 0.5 \
                <= self.cfg["max_jump_px_per_frame"]:
            self.pending = None
            self.track_len = 0
            return best
        self.pending = best
        return None

    # --- camera --------------------------------------------------------

    def steer_camera(self, ball, width, now, targeted):
        """ball is this frame's (x, y, state, radius) or None; targeted is
        whether your own character is highlighted as the ball's target."""
        # A red ball while you're highlighted counts as the real thing right
        # away -- it's what the search is looking for, so stop sweeping.
        stable = ball is not None and (self.track_len >= CAMERA_MIN_TRACK_FRAMES or
                                       (targeted and ball[2] == "targeting"))
        # You're the target but no ball is steadily tracked in view -- so the
        # ball that's coming for you is off-screen, usually behind. Find it
        # now instead of waiting to lose track. A steadily tracked ball is
        # kept in view instead, whatever color it reads as: it's usually the
        # real one even when it doesn't look red, and turning away from it
        # was exactly what sent the camera the wrong way in live testing.
        if targeted and not stable:
            self.camera.turn(self.last_seen_side, CAMERA_SEARCH_TURN_S, now)
            return
        if stable:
            offset = (ball[0] - width / 2) / (width / 2)  # -1 = left edge, +1 = right edge
            self.last_seen_side = -1 if offset < 0 else 1
            beyond = abs(offset) - CAMERA_EDGE_ZONE
            if beyond > 0:
                duration = min(max(beyond * CAMERA_TURN_GAIN_S / (1 - CAMERA_EDGE_ZONE),
                                   CAMERA_MIN_TURN_S), CAMERA_MAX_TURN_S)
                self.camera.turn(self.last_seen_side, duration, now)
        elif ball is None and self.missing_streak >= CAMERA_SEARCH_AFTER_FRAMES and \
                now - self.missing_since <= CAMERA_SEARCH_MAX_S:
            self.camera.turn(self.last_seen_side, CAMERA_SEARCH_TURN_S, now)
        # else: a detection that just jumped here -- often a decoy (a sparkle,
        # a lantern, another player's cosmetic), so don't swing the camera
        # toward it until it's held up for a few frames.

    # --- per-frame prediction -----------------------------------------

    def predict_actions(self, row):
        features = [[row[col] for col in self.feature_columns]]
        # predict_proba returns per-label confidence (0-1) instead of just
        # a yes/no guess, so each action can use its own threshold.
        proba = self.model.predict_proba(self.scaler.transform(features))[0]
        desired = {
            label.replace("held_", "")
            for label, confidence in zip(self.label_columns, proba)
            if confidence >= THRESHOLDS.get(label, 0.5)
        }
        return desired, proba

    def log_frame(self, t, frame_bgr, ball, row, proba, desired, tapped):
        frame_name = None
        if self._log_frames_dir is not None:
            frame_name = f"{self._log_frame_idx:06d}.jpg"
            cv2.imwrite(str(self._log_frames_dir / frame_name), frame_bgr)
            self._log_frame_idx += 1
        x, y, state, _ = ball if ball is not None else (None, None, None, None)
        entry = {
            "t": t, "frame": frame_name, "ball_x": x, "ball_y": y, "state": state,
            **(row or {}),
            "proba": None if proba is None else
                     {label: float(p) for label, p in zip(self.label_columns, proba)},
            "desired": sorted(desired),
            "tapped": sorted(tapped),
            "camera": self.camera.direction,
            "track_len": self.track_len,
            "self_red": round(self.self_red, 4),
        }
        self._log_file.write(json.dumps(entry) + "\n")
        self._log_file.flush()

    # --- main loop -------------------------------------------------

    def run(self):
        interval = 1.0 / FPS
        region = self._sct.monitors[1]
        width, height = region["width"], region["height"]
        center_x, center_y = width / 2, height / 2
        self.camera.anchor = (region["left"] + width // 2, region["top"] + height // 2)
        roi = self.cfg["self_highlight_roi"]
        character_xy = ((roi["x0"] + roi["x1"]) / 2 * width, (roi["y0"] + roi["y1"]) / 2 * height)

        print("Insert = toggle AI on/off, End = quit.")
        print("Starting in OFF state -- press Insert when you're ready.")

        while not self.quit:
            start = time.time()
            with self._lock:
                enabled = self.enabled

            if enabled:
                self.camera.update(start)
                capture_t = time.time()
                frame_bgr = cv2.cvtColor(np.array(self._sct.grab(region)), cv2.COLOR_BGRA2BGR)
                self.self_red = track_ball.self_highlight_score(frame_bgr, self.cfg)
                if self.self_red >= self.cfg["self_target_threshold"]:
                    self.targeted_at = (capture_t, self.self_red)
                candidates, cores = track_ball.find_ball_candidates(frame_bgr, self.cfg)
                if self.scorer is not None:
                    candidates, cores = self.scorer.filter(
                        frame_bgr, candidates, cores, self.cfg["min_ball_score"])
                ball = self.select_ball(candidates, cores)
                # Tint blinked off but the ball is still coming -- unless a block
                # was already tapped since, in which case it most likely worked
                # and whatever red thing is nearby isn't coming for you.
                if self.self_red < self.cfg["self_target_threshold"] and self.targeted_at \
                        and capture_t - self.targeted_at[0] <= TARGET_LATCH_S \
                        and self.last_tap.get("block", 0) < self.targeted_at[0] \
                        and ball is not None and ball[2] == "targeting":
                    self.self_red = self.targeted_at[1]
                targeted = self.self_red >= self.cfg["self_target_threshold"]
                row = proba = None
                desired = tapped = set()

                if ball is None:
                    if self.missing_streak == 0:
                        self.missing_since = capture_t
                    self.missing_streak += 1
                    if self.missing_streak >= MISSING_FRAMES_RESET:
                        self.release_all()
                        self.prev_ball = None
                        self.pending = None
                    self.steer_camera(None, width, capture_t, targeted)
                else:
                    self.missing_streak = 0
                    x, y, state, radius = ball
                    if self.prev_ball is not None and \
                            ((x - self.prev_ball[0]) ** 2 + (y - self.prev_ball[1]) ** 2) ** 0.5 \
                            <= self.cfg["stale_jitter_px"]:
                        self.stale_count += 1
                    else:
                        self.stale_count = 0
                    self.prev_ball = (x, y, radius)

                    # A turning camera sweeps the whole scene across the
                    # screen -- that motion isn't the ball's, so restart
                    # motion history instead of reading it as velocity.
                    if self.camera.last_active >= self.last_feature_t:
                        self.features.reset()
                    row = self.features.update(x, y, state, radius, self.self_red,
                                               capture_t, center_x, center_y)
                    self.last_feature_t = capture_t

                    desired, proba = self.predict_actions(row)
                    if should_block("block" in desired, ball, targeted, character_xy):
                        desired = desired | {"block"}
                    else:
                        desired = desired - {"block"}
                    tapped = self.apply_actions(desired)
                    self.steer_camera(ball, width, capture_t, targeted)

                if self._log_file is not None:
                    self.log_frame(capture_t, frame_bgr, ball, row, proba, desired, tapped)

            elapsed = time.time() - start
            time.sleep(max(0.0, interval - elapsed))


def run_camera_test(camera):
    print("Camera test: switch to Roblox now. Starting in 3 seconds...")
    time.sleep(3)
    for name, direction in (("LEFT", -1), ("RIGHT", 1)):
        print(f"Turning {name} for 1s -- the view should rotate to look {name.lower()}.")
        end = time.time() + 1.0
        camera.turn(direction, 1.0, time.time())
        while time.time() < end:
            camera.update(time.time())
            time.sleep(1 / 60)
        camera.stop()
        time.sleep(1.0)
    print("Done. Wrong direction? add --camera-invert. Didn't move? try --camera mouse.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="Path to model.joblib from train_model.py")
    parser.add_argument("--camera", choices=["keys", "mouse", "off"], default="keys",
                        help="How to turn the camera toward the ball (default: arrow keys)")
    parser.add_argument("--camera-invert", action="store_true",
                        help="Flip the camera turn direction")
    parser.add_argument("--camera-test", action="store_true",
                        help="Turn the camera left then right to check the setup, then exit")
    parser.add_argument("--no-classifier", action="store_true",
                        help="Don't use ball_classifier.joblib to filter out decoys "
                             "(to compare against it)")
    parser.add_argument("--log", nargs="?", const="auto",
                        help="Write a JSONL diagnostic log (features, predicted confidences, "
                             "camera) for every live-inferred frame while AI is enabled, and "
                             "save the captured frames into a <log>_frames/ folder. With no "
                             "name, a new timestamped file is made in live_logs/ each run, "
                             "so earlier logs are never overwritten.")
    args = parser.parse_args()
    if args.log == "auto":
        Path("live_logs").mkdir(exist_ok=True)
        args.log = str(Path("live_logs") / f"live_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

    if args.camera_test:
        run_camera_test(Camera(args.camera, args.camera_invert,
                               keyboard.Controller(), mouse.Controller()))
        return
    if not args.model:
        parser.error("Provide model.joblib (or use --camera-test).")

    controller = AIController(args.model, args.camera, args.camera_invert, log_path=args.log,
                              use_classifier=not args.no_classifier)
    print("Ball classifier: " + ("on" if controller.scorer else
                                 "off" if args.no_classifier else "not trained yet (off)"))
    if args.log:
        print(f"Logging live diagnostics to {args.log} (frames in {controller._log_frames_dir})")

    try:
        with keyboard.Listener(on_press=controller.on_key_press) as listener:
            controller.run()
            listener.join()
    finally:
        controller.release_all()  # belt-and-suspenders: never leave keys stuck
        if controller._log_file is not None:
            controller._log_file.close()
        print("Exited, all keys/buttons released.")


if __name__ == "__main__":
    main()
