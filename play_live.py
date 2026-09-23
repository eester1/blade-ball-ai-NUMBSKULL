"""
Live AI play for Blade Ball.

Watches your screen in real time, detects the ball (reusing track_ball.py's
detection code), builds the same features build_dataset.py used for
training, feeds them into your trained model, and presses/releases the
predicted keys and mouse buttons for you.

SETUP (run once, if you haven't already):
    pip install mss pynput joblib pandas scikit-learn opencv-python numpy

USAGE:
    python play_live.py model.joblib

    - Press Insert to toggle AI control ON/OFF. Starts OFF -- nothing
      happens until you turn it on.
    - Press End to quit immediately. This always releases every key/
      button first, so nothing gets left stuck held down.

SAFETY: while ON, this sends REAL key/mouse input to whatever window is
focused. Keep Roblox focused and your hand near the keyboard the first
few times you try it, in case it does something you don't want -- End
stops everything instantly.
"""

import argparse
import json
import time
import threading
from pathlib import Path

import cv2
import joblib
import mss
import numpy as np
from pynput import keyboard, mouse

import track_ball  # reuses find_ball / load_config / color_config.json

# Target capture rate. Velocity is now computed in pixels/second from
# real timestamps (see predict_actions), so this no longer needs to match
# the recording FPS exactly -- it just caps how often we poll the screen.
FPS = 15

# How confident the model needs to be (0-1) before actually pressing each
# action. Default is 0.5 (the model's own best guess). Lower this for an
# action to make it act on weaker hunches -- more false alarms, but fewer
# missed moments. Blocking is lowered here since missing a real block
# (death) is worse than an unnecessary one (mostly harmless).
THRESHOLDS = {
    "held_w": 0.5,
    "held_a": 0.5,
    "held_s": 0.5,
    "held_d": 0.5,
    "held_mouse_left": 0.3,
    "held_mouse_right": 0.5,
}

TOGGLE_KEY = keyboard.Key.insert
QUIT_KEY = keyboard.Key.end

KEY_ACTIONS = {"w": "w", "a": "a", "s": "s", "d": "d"}
MOUSE_ACTIONS = {"mouse_left": mouse.Button.left, "mouse_right": mouse.Button.right}

# If the ball goes undetected for this many consecutive frames, release
# everything as a safety net (probably out of a round, or tracking lost).
MISSING_FRAMES_RESET = 30


class AIController:
    def __init__(self, model_path, log_path=None):
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.label_columns = data["label_columns"]

        self.cfg = track_ball.load_config()
        self.kb = keyboard.Controller()
        self.ms = mouse.Controller()

        self.enabled = False
        self.quit = False
        self._lock = threading.Lock()

        self.currently_held = set()
        self.prev_ball = None  # (x, y, t) from the previous frame, for velocity
        self.missing_streak = 0
        self.stale_count = 0  # consecutive frames barely-unchanged in position
        self.pending = None    # (x, y, state) candidate awaiting next-frame confirmation

        # Below this dt, treat elapsed time as "basically zero" and fall
        # back to zero velocity instead of dividing by a near-zero number
        # (matches build_dataset.py's MIN_DT so train/live features agree).
        self.MIN_DT = 1e-3

        # Once pressed, some actions must stay held for at least this long
        # (seconds) before they're allowed to release, even if the model's
        # frame-by-frame confidence dips below threshold in between. This
        # stops a correctly-timed-but-brief block signal from letting go
        # right before the ball actually arrives.
        self.min_hold_seconds = {"mouse_left": 0.4}
        self.press_time = {}  # action -> time.time() it was last pressed

        self._sct = mss.mss()

        # Optional diagnostic log: one JSON line per live-inferred frame
        # (ball position/velocity as seen live, predicted confidences,
        # actions taken). Lets us compare what the model saw during an
        # actual match against the training data's feature distribution,
        # instead of guessing why live behavior diverges from offline
        # test-set metrics.
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

    def apply_actions(self, desired):
        now = time.time()
        to_press = desired - self.currently_held

        release_candidates = self.currently_held - desired
        to_release = set()
        for action in release_candidates:
            min_hold = self.min_hold_seconds.get(action, 0)
            pressed_at = self.press_time.get(action, 0)
            if now - pressed_at >= min_hold:
                to_release.add(action)
            # else: not held long enough yet -- keep it held a bit longer

        for action in to_press:
            self.press_action(action)
            self.press_time[action] = now
        for action in to_release:
            self.release_action(action)

        self.currently_held = (self.currently_held - to_release) | to_press

    # --- per-frame prediction -----------------------------------------

    def predict_actions(self, ball_x, ball_y, state, center_x, center_y, t, frame_bgr=None):
        if self.prev_ball is None:
            vel_x = vel_y = 0
        else:
            dt = t - self.prev_ball[2]
            if dt < self.MIN_DT:
                vel_x = vel_y = 0
            else:
                vel_x = (ball_x - self.prev_ball[0]) / dt
                vel_y = (ball_y - self.prev_ball[1]) / dt
        self.prev_ball = (ball_x, ball_y, t)

        rel_x, rel_y = ball_x - center_x, ball_y - center_y
        distance = (rel_x ** 2 + rel_y ** 2) ** 0.5
        closing_speed = 0.0 if distance < 1e-6 else -(rel_x * vel_x + rel_y * vel_y) / distance
        state_targeting = 1 if state == "targeting" else 0

        row = {
            "ball_rel_x": rel_x, "ball_rel_y": rel_y,
            "ball_vel_x": vel_x, "ball_vel_y": vel_y,
            "ball_distance": distance, "ball_closing_speed": closing_speed,
            "state_targeting": state_targeting,
        }
        features = [[row[col] for col in self.feature_columns]]
        scaled = self.scaler.transform(features)
        # predict_proba returns per-label confidence (0-1) instead of just
        # a yes/no guess, so each action can use its own threshold below.
        proba = self.model.predict_proba(scaled)[0]

        desired = set()
        for label, confidence in zip(self.label_columns, proba):
            threshold = THRESHOLDS.get(label, 0.5)
            if confidence >= threshold:
                desired.add(label.replace("held_", ""))

        if self._log_file is not None:
            frame_name = None
            if self._log_frames_dir is not None and frame_bgr is not None:
                frame_name = f"{self._log_frame_idx:06d}.jpg"
                cv2.imwrite(str(self._log_frames_dir / frame_name), frame_bgr)
                self._log_frame_idx += 1
            entry = {
                "t": t, "frame": frame_name, "ball_x": ball_x, "ball_y": ball_y, "state": state,
                **row,
                "proba": {label: float(p) for label, p in zip(self.label_columns, proba)},
                "desired": sorted(desired),
            }
            self._log_file.write(json.dumps(entry) + "\n")
            self._log_file.flush()

        return desired

    # --- main loop -------------------------------------------------

    def run(self):
        interval = 1.0 / FPS
        region = self._sct.monitors[1]
        width, height = region["width"], region["height"]
        center_x, center_y = width / 2, height / 2

        print("Insert = toggle AI on/off, End = quit.")
        print("Starting in OFF state -- press Insert when you're ready.")

        while not self.quit:
            start = time.time()
            with self._lock:
                enabled = self.enabled

            if enabled:
                capture_t = time.time()
                shot = self._sct.grab(region)
                frame = np.array(shot)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                candidates = track_ball.find_ball_candidates(frame_bgr, self.cfg)

                result = None
                if candidates:
                    # A real ball shouldn't sit at the exact same pixel for
                    # seconds -- if it has, stop trusting proximity to that
                    # spot (it's more likely a static decoy: a map decoration,
                    # a standing player, an unmasked UI element) and force a
                    # fresh whole-frame best-candidate search instead.
                    stale = self.stale_count >= self.cfg["stale_after_frames"]
                    if self.prev_ball is None or stale:
                        result = track_ball.most_circular(candidates)[1:]
                        self.pending = None
                    else:
                        px, py = self.prev_ball[:2]
                        # Scales with how many frames since the last trusted
                        # detection, same idea as track_ball.py's batch
                        # tracker -- keeps live inference from latching onto
                        # a same-colored decoy (another player's head, a
                        # skill effect) just because it's the most circular
                        # blob in a single frame.
                        max_dist = self.cfg["max_jump_px_per_frame"] * max(1, self.missing_streak + 1)
                        near = [c for c in candidates
                                if ((c[1] - px) ** 2 + (c[2] - py) ** 2) ** 0.5 <= max_dist]
                        if near:
                            result = track_ball.closest_to(near, (px, py))[1:]
                            self.pending = None
                        else:
                            # Nothing near the last trusted position. Batch
                            # tracking can peek at the next recorded frame
                            # before trusting a big jump; live play can't
                            # see the future, so instead we wait exactly one
                            # frame: hold the best candidate as "pending"
                            # and only trust it once the *next* frame keeps
                            # finding something near it. A one-off flash --
                            # a torch, a skill effect -- won't repeat there
                            # and gets dropped; real ball movement will.
                            _, cx, cy, cstate = track_ball.most_circular(candidates)
                            confirm_dist = self.cfg["max_jump_px_per_frame"]
                            if self.pending is not None and \
                                    ((cx - self.pending[0]) ** 2 + (cy - self.pending[1]) ** 2) ** 0.5 <= confirm_dist:
                                result = (cx, cy, cstate)
                                self.pending = None
                            else:
                                self.pending = (cx, cy, cstate)

                if result is None:
                    self.missing_streak += 1
                    if self.missing_streak >= MISSING_FRAMES_RESET:
                        self.release_all()
                        self.prev_ball = None
                        self.pending = None
                else:
                    self.missing_streak = 0
                    x, y, state = result
                    if self.prev_ball is not None and \
                            ((x - self.prev_ball[0]) ** 2 + (y - self.prev_ball[1]) ** 2) ** 0.5 <= self.cfg["stale_jitter_px"]:
                        self.stale_count += 1
                    else:
                        self.stale_count = 0
                    desired = self.predict_actions(x, y, state, center_x, center_y, capture_t, frame_bgr)
                    self.apply_actions(desired)

            elapsed = time.time() - start
            time.sleep(max(0.0, interval - elapsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model.joblib from train_model.py")
    parser.add_argument("--log", help="Optional path to write a JSONL diagnostic log "
                                       "(ball position/velocity, predicted confidences) "
                                       "for every live-inferred frame while AI is enabled. "
                                       "Also saves the captured frame images next to it, "
                                       "in a <log>_frames/ folder, for visual inspection.")
    args = parser.parse_args()

    controller = AIController(args.model, log_path=args.log)
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