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

    - Starts OFF: press Insert in Roblox to turn it on, and again to pause.
    - --auto: plays by itself instead -- it plays each round, waits in the
      lobby after you die or the round ends, and starts again when the next
      round begins (it can tell from Blade Ball's menu -- see game_state.py).
      Insert still pauses/resumes it. With --vote classic it also votes for
      that gamemode each time it's in the lobby.
    - It only ever acts while the Roblox window is the active one.
    - End quits immediately (--quit-key picks another key). This always
      releases every key/button first, so nothing gets left stuck held down.

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
import game_state
import track_ball
from features import FeatureTracker

# How often the screen is looked at. At 15 a fast late-round ball moved
# 100-180px between looks, so blocks landed a frame late. With dxcam capture
# (~1 ms instead of ~33 ms with mss) a frame costs ~25 ms, so 30 fits.
FPS = 30
# The recordings the model learned from were made at 15 fps, and its
# features (velocity, radius growth) are smoothed and differenced per
# sample -- so the model still sees the ball at that rate, and its last
# decision is reused in between. Tracking, block timing and the camera run
# at the full FPS.
RECORDING_FPS = 15
MODEL_INTERVAL_S = 1.0 / RECORDING_FPS - 0.005
# Settings below that count frames were tuned at 15 fps; this keeps them
# meaning the same amount of time.
FRAME_SCALE = FPS // RECORDING_FPS
# With --log, save a screenshot on every Nth frame only (15 a second), so
# logs don't double in size.
LOG_FRAME_EVERY = FRAME_SCALE

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
# ...but a red ball while you're targeted counts from this size: in one run
# (camera probably zoomed out a little) 42% of contacts were under radius
# 14, and a fast final ball was let through at 165px because it looked
# 10.5. The far practice-mode ball that spammed blocks measured 8-10. Over
# the logged targetings this makes the rule fire in 89% instead of 85%,
# and too late (<0.1s) in 15% instead of 24%.
BLOCK_MIN_RADIUS_TARGETED = 11
BLOCK_BIG_RADIUS = 45
BLOCK_BIG_MAX_CHAR_DIST = 320
# Judge "close" by where the ball will be this far ahead, from how fast
# it's been closing in on your character. A fast ball covers ~100-180px
# per frame at the end, so it can go from "too far" to hitting you
# between two frames -- both deaths in the 2026-09-23 runs were a tap one
# frame too late like that (at 112px and 121px, already inside the hit).
BLOCK_LEAD_S = 0.1
# ...but never more than this much closer, so a speed misread mid camera
# turn can't set off a block while the ball is still far away.
BLOCK_MAX_LEAD_PX = 150
# A ball coming at you head-on barely moves on screen -- it just grows (in
# one death its radius went 12 -> 14 -> 17 -> 24 in three frames), so its
# size is judged ahead the same way, from how fast it's growing, by at
# most half again its current size.
BLOCK_MAX_GROWTH_LEAD = 0.5
# A red ball this big while you're targeted is on top of you, wherever it
# is on screen: one coming from behind the camera shows up huge at the
# bottom of the screen, well away from your character (a death: radius 97,
# 408px away, never blocked).
BLOCK_HUGE_RADIUS = 70
# Screen distance from your character is a poor measure of "close" when the
# ball comes from in front of you, from above or near the camera: it can be
# on you while still 200-500px away on screen (two deaths, never blocked in
# time). Its apparent size gives its depth, so size + screen position give
# its real 3D distance from your character (see ball_distance_3d). Over 278
# logged targetings that separated "at contact" from "half a second before"
# 2.6x better than screen distance. While you're targeted and the ball is
# red, it also counts as close within this 3D distance (in ball radii):
# replayed on those targetings, the block rule then fires in 87% of them
# instead of 68%, and too late (<0.1s before contact) in 23% instead of 46%.
BLOCK_3D_DIST = 22
CAMERA_FOV_DEG = 70      # Roblox's default vertical field of view
CHARACTER_DEPTH = 30     # camera to your character, in ball radii (fitted to logged contacts)

# Your character's red "targeted" tint can blink off while the ball is
# still coming. For this long after last seeing it, keep counting as
# targeted while the tracked ball is still red, instead of the model
# suddenly deciding the danger has passed -- but not once a block has been
# tapped since (then the tint going away most likely means it worked).
TARGET_LATCH_S = 1.5

TOGGLE_KEY = keyboard.Key.insert
DEFAULT_QUIT_KEY = "end"

# Playing by itself (see game_state.py): whether you're in a round is
# checked every LOBBY_CHECK_EVERY frames while playing (every frame while
# waiting), and it only switches after this many checks in a row agree --
# so one odd frame (a menu flicker, a flash over the button) can't pause
# or start it.
LOBBY_CHECK_EVERY = 3 * FRAME_SCALE
LOBBY_CONFIRM_CHECKS = 2
ROUND_CONFIRM_CHECKS = 3 * FRAME_SCALE
# Auto vote: while in the lobby, look for the chosen gamemode's vote button
# this often (frames), and click it once per visit to the lobby.
VOTE_CHECK_EVERY = 8 * FRAME_SCALE
# A vote counts once the game shows its tick on that button; otherwise click
# again, this long after the last click, up to VOTE_MAX_TRIES times a visit.
VOTE_CHECK_AFTER_S = 1.0
VOTE_MAX_TRIES = 3
# While logging, save a lobby frame this often (seconds), so what the lobby
# looked like (vote screen included) can be checked afterwards.
LOBBY_FRAME_EVERY_S = 1.0

# What's shown for each phase when it changes (the control panel watches
# for these lines).
PHASE_MESSAGES = {
    "playing": "[in a round -- AI playing]",
    "lobby": "[in the lobby -- waiting for the next round]",
    "no_focus": "[waiting for the Roblox window -- click into Roblox]",
}


def parse_key(name):
    """A pynput key from a name like "end", "f8", "page_down" or "x"."""
    name = name.strip().lower().replace(" ", "_")
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(f"unknown key {name!r} (try end, home, delete, page_down, f8 ...)")


def roblox_focused():
    """Whether the active window is Roblox -- the AI never sends input
    anywhere else (e.g. into the control panel right after Start AI)."""
    if sys.platform != "win32":
        return True
    user32 = ctypes.windll.user32
    title = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(user32.GetForegroundWindow(), title, 256)
    return "roblox" in title.value.lower()

# If the ball goes undetected for this many consecutive frames, release
# every held action as a safety net (probably out of a round, or lost).
MISSING_FRAMES_RESET = 30 * FRAME_SCALE

# --- Camera control ---------------------------------------------------------
# Only steer toward a detection that's been followed smoothly for at least
# this many consecutive frames. In live testing, most camera swings away
# from the real ball started on a detection that had just jumped hundreds
# of pixels -- a decoy -- rather than on a steady track.
CAMERA_MIN_TRACK_FRAMES = 3 * FRAME_SCALE
# Turn toward the ball only once it's past this fraction of the way from
# screen center to the left/right edge. The dead zone in the middle keeps
# the ball's on-screen position meaningful to the model (which learned
# from your recordings, where the ball wasn't pinned to center), while
# still stopping it from drifting off-screen.
CAMERA_EDGE_ZONE = 0.5
# ...judged by where the ball will be this far ahead, from its motion on
# screen. A ball flying across the view at ~1500px/s was gone off the
# edge a frame or two after crossing the edge zone -- before the turn
# took effect -- and most deaths started exactly like that.
CAMERA_LEAD_S = 0.25
# The game shows a turn about a frame late, so motion measured across a
# frame this soon after the camera moved is still partly the camera's.
CAMERA_SETTLE_S = 0.07
# How long a ball's measured motion is still trusted while the camera turns
# and it can't be measured afresh.
MOTION_HOLD_S = 0.3
# Ball speed and growth are measured over at least this long (about one
# 15 fps frame) -- over a 30 fps frame the pixel jitter would double them.
MOTION_MIN_DT = 0.06
# Turn duration per unit of distance past the edge zone, and its bounds.
CAMERA_TURN_GAIN_S = 0.3
CAMERA_MIN_TURN_S = 0.04
CAMERA_MAX_TURN_S = 0.15
# Searching turns toward the side the ball was last seen on, in steps of this.
CAMERA_SEARCH_TURN_S = 0.12
# Mouse mode: search speed while you're targeted, relative to normal turns.
# Measured from live frames, a normal-speed turn already goes all the way
# round in under a second (~430 degrees/s); 1.5x only overshot more.
CAMERA_SEARCH_SPEED = 1.0
# A ball this small on screen is far away. While it isn't coming for you,
# it's followed calmly -- no leading, no quick search, no fast sweeps: a
# far ball bouncing between players on opposite sides swung the camera
# back and forth, and every sweep overshoots a bit (the game shows a turn
# about a frame late). If it does turn on you, your red highlight starts
# the fast search anyway.
CAMERA_FAR_RADIUS = 10
# Looking for a ball that isn't after you. Two ways this went wrong:
# - Searching whenever it was lost (for up to 6s, restarted by every brief
#   detection -- ability flashes, the ball darting between nearby players)
#   chained short searches into the camera circling.
# - Not searching at all: at the start of a round the ball usually isn't
#   in view, so it was never found -- seen in 12% of frames instead of ~60%,
#   and the model (which only acts when it sees the ball) barely moved.
# So: once it's been missing this long, one sweep of CAMERA_IDLE_SWEEP_S
# (about a full turn), stopped the moment the ball is found, and at most one
# sweep every CAMERA_IDLE_SWEEP_EVERY_S. Each round starts fresh. If the ball
# turns on you, your red highlight starts the (unlimited, faster) targeted
# search anyway.
CAMERA_IDLE_SEARCH_AFTER_S = 1.0
CAMERA_IDLE_SWEEP_S = 2.5
CAMERA_IDLE_SWEEP_EVERY_S = 7.0
# ...turning at this fraction of normal speed: ~150 degrees/s, one full turn
# in ~2.5s. At full speed (~430 degrees/s) the ball was on screen for only
# a few frames per turn -- too few to be recognized -- and a 1.5s sweep
# spun past it about twice.
CAMERA_IDLE_SWEEP_SPEED = 0.35
# Drag speed in "mouse" mode, in mouse counts per second.
CAMERA_MOUSE_SPEED = 900
# Mouse mode: once the dragged cursor is this many pixels from the anchor,
# hop it back (see Camera.update) so it never leaves the game screen.
CAMERA_REANCHOR_PX = 250


LEARNED_BLOCK_PATH = Path(__file__).resolve().parent / "learned_block.json"


def use_learned_block_timing():
    """--learned-block: take BLOCK_3D_DIST from learned_block.json -- the tap
    distance learn_block.py found works best in your own logged runs. The
    built-in value stays if nothing has been learned yet."""
    global BLOCK_3D_DIST
    try:
        learned = json.loads(LEARNED_BLOCK_PATH.read_text())
    except (OSError, ValueError):
        print(f"Learned block timing: none yet -- press 'Learn from my runs' (learn_block.py). "
              f"Using the built-in {BLOCK_3D_DIST}.")
        return
    BLOCK_3D_DIST = learned["block_3d_dist"]
    print(f"Learned block timing: taps within {BLOCK_3D_DIST} ball radii (3D), learned "
          f"{learned['learned_at']} from {learned['targetings']} targetings.")


def ball_distance_3d(x, y, radius, character_xy, screen_h):
    """The ball's distance from your character in 3D, in ball radii, from its
    screen position and apparent size (a perspective camera: size shrinks in
    proportion to depth). Approximate -- it assumes the default field of
    view and your camera zoom (CHARACTER_DEPTH)."""
    focal = (screen_h / 2) / np.tan(np.radians(CAMERA_FOV_DEG / 2))
    depth = focal / radius
    mid_x, mid_y = character_xy[0], screen_h / 2
    bx, by = (x - mid_x) / radius, (y - mid_y) / radius
    char_y = (character_xy[1] - mid_y) * CHARACTER_DEPTH / focal
    return (bx ** 2 + (by - char_y) ** 2 + (depth - CHARACTER_DEPTH) ** 2) ** 0.5


def should_block(model_wants, ball, targeted, character_xy, approach_speed=0.0, growth=0.0,
                 screen_h=1080):
    """Final block decision for this frame (see BLOCK_* above).
    approach_speed: how fast the ball is closing in on your character on
    screen, in px/s; growth: how fast its radius is growing, in px/s (0 if
    unknown)."""
    x, y, state, radius = ball
    coming = targeted and state == "targeting"
    if coming and ball_distance_3d(x, y, radius, character_xy, screen_h) <= BLOCK_3D_DIST:
        return True
    distance = ((x - character_xy[0]) ** 2 + (y - character_xy[1]) ** 2) ** 0.5
    distance -= min(max(approach_speed, 0.0) * BLOCK_LEAD_S, BLOCK_MAX_LEAD_PX)
    radius += min(max(growth, 0.0) * BLOCK_LEAD_S, radius * BLOCK_MAX_GROWTH_LEAD)
    min_radius = BLOCK_MIN_RADIUS_TARGETED if coming else BLOCK_MIN_RADIUS
    close = (distance <= BLOCK_MAX_CHAR_DIST and radius >= min_radius) or \
        (distance <= BLOCK_BIG_MAX_CHAR_DIST and radius >= BLOCK_BIG_RADIUS)
    return (close and (model_wants or coming)) or (coming and radius >= BLOCK_HUGE_RADIUS)


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


def click_at(x, y, screen_w, screen_h, from_xy):
    """Move the mouse to (x, y) on the primary screen and left-click, as real
    Windows mouse input (SendInput), in a few steps like a hand would.
    Setting the cursor position directly (pynput) isn't seen by Roblox: the
    gamemode vote clicks landed wherever its cursor already was -- the
    middle button, or nothing."""
    if sys.platform != "win32":
        raise RuntimeError("clicking needs Windows (SendInput)")
    send = ctypes.windll.user32.SendInput

    def move(px, py):  # MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0-65535 scale
        inp = _INPUT(type=0, mi=_MOUSEINPUT(int(px * 65535 / (screen_w - 1)),
                                            int(py * 65535 / (screen_h - 1)), 0, 0x8001, 0, 0))
        send(1, ctypes.byref(inp), ctypes.sizeof(inp))

    fx, fy = from_xy
    for i in range(1, 6):
        move(fx + (x - fx) * i / 5, fy + (y - fy) * i / 5)
        time.sleep(0.02)
    time.sleep(0.08)
    for flag in (0x0002, 0x0004):  # left down, left up
        inp = _INPUT(type=0, mi=_MOUSEINPUT(0, 0, 0, flag, 0, 0))
        send(1, ctypes.byref(inp), ctypes.sizeof(inp))
        time.sleep(0.04)


class Screen:
    """Grabs the primary monitor as a BGR image. Uses dxcam (Windows Desktop
    Duplication: ~1 ms a frame) when it's installed and works, else mss
    (~33 ms a frame, which alone takes half of a 30 fps frame)."""

    def __init__(self, region):
        self.region = region
        self.method = "mss"
        self._cam = None
        try:
            import dxcam
            cam = dxcam.create(output_color="BGR")
            if cam is not None and (cam.width, cam.height) == (region["width"], region["height"]):
                cam.start(target_fps=60, video_mode=True)
                self._cam, self.method = cam, "dxcam"
        except Exception:  # not installed, or no Desktop Duplication here
            self._cam = None
        if self._cam is None:
            self._sct = mss.mss()

    def grab(self):
        if self._cam is not None:
            # A copy: dxcam keeps writing into its own buffer while we work.
            return self._cam.get_latest_frame().copy()
        return cv2.cvtColor(np.array(self._sct.grab(self.region)), cv2.COLOR_BGRA2BGR)

    def close(self):
        if self._cam is not None:
            self._cam.stop()
            self._cam = None


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
        self.speed = 1.0
        self.until = 0.0
        self.last_active = 0.0  # last time the view was moving
        self._last_move = None  # mouse mode: time of the last drag step
        self._anchor = None     # mouse mode: anchor for the current turn

    def turn(self, direction, duration, now, speed=1.0):
        """speed scales the drag rate in mouse mode (keys turn at one rate)."""
        if self.method == "off":
            return
        if self.invert:
            direction = -direction
        self.speed = speed
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
            dx = int(self.direction * CAMERA_MOUSE_SPEED * self.speed * (now - self._last_move))
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
                 use_classifier=True, auto=False, quit_key=DEFAULT_QUIT_KEY, vote=None):
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
        # Counted in frames, tuned at 15 fps (see FRAME_SCALE).
        for key in ("stale_after_frames", "size_continuity_frames"):
            self.cfg[key] *= FRAME_SCALE
        # Learned "is this blob really the ball?" filter, if it's been trained.
        self.scorer = ball_classifier.load_scorer() if use_classifier else None
        self.kb = keyboard.Controller()
        self.ms = mouse.Controller()
        self.camera = Camera(camera_method, camera_invert, self.kb, self.ms)

        self.quit_key_name = quit_key
        self.quit_key = parse_key(quit_key)
        # Plays by itself between rounds if the lobby can be recognized.
        self.lobby = game_state.LobbyDetector() if auto else None
        self.enabled = self.lobby is not None  # auto mode starts armed
        # Auto vote (needs auto mode, which knows when you're in the lobby).
        self.vote_mode = vote if auto else None
        self.vote_button = None
        if self.vote_mode:
            try:
                self.vote_button = game_state.VoteButton(self.vote_mode)
            except FileNotFoundError as e:
                print(f"[auto vote off: {e} -- see README, Auto vote]")
        self.voted = False           # the vote registered this lobby visit
        self.vote_tries = 0          # clicks tried this lobby visit
        self.vote_next_t = 0.0       # earliest time for the next click / check
        self.last_lobby_frame_t = 0.0
        self.phase = None          # "playing", "lobby", "no_focus" or None (not yet known)
        self.lobby_streak = 0      # checks in a row that saw the lobby
        self.round_streak = 0      # ... and that saw a round
        self.frame_count = 0
        self.screen = None         # Screen, made when run() starts
        self.last_model = None     # (row, desired, proba) of the last model run
        self.last_model_t = 0.0
        self.quit = False
        self._lock = threading.Lock()

        self.currently_held = set()
        self.features = FeatureTracker()
        self.last_feature_t = 0.0
        self.prev_ball = None     # (x, y, radius) of the last trusted detection
        self.last_seen_side = 1   # -1 = ball last seen left of center, +1 = right
        self.searching = False    # the camera is sweeping to find a lost ball
        self.last_far = False     # the ball was far away and not after you (CAMERA_FAR_RADIUS)
        self.idle_search_from = None  # when the last idle sweep began (see CAMERA_IDLE_SWEEP_S)
        self.prev_motion = None   # (t, x, distance to character) of the last detection
        self.screen_vx = 0.0      # ball's sideways speed on screen, px/s
        self.approach = 0.0       # how fast the ball closes in on your character, px/s
        self.motion_t = 0.0       # when those two were last measured
        self.growth = 0.0         # how fast the ball's radius grows, px/s
        self.missing_streak = 0
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
                # (and would block the camera search from kicking in) -- and
                # work out again whether we're in a round.
                self.reset_tracking()
                self.phase = None
                self.lobby_streak = self.round_streak = 0
            print(f"[AI {'ENABLED' if state else 'disabled'}]")
            if not state:
                self.release_all()
            return
        if key == self.quit_key:
            with self._lock:
                self.quit = True
            self.release_all()
            print("[quitting]")
            return False

    def reset_tracking(self):
        """Forget the ball -- after a pause or between rounds it's stale."""
        self.missing_streak = 0
        self.prev_ball = None
        self.pending = None
        self.targeted_at = None
        self.track_len = 0
        self.stale_count = 0
        self.prev_motion = None
        self.screen_vx = self.approach = self.growth = 0.0
        self.idle_search_from = None
        self.searching = False
        self.features.reset()
        self.last_model = None

    def set_phase(self, phase, t):
        """Switch between playing / waiting in the lobby / waiting for the
        Roblox window, letting go of everything when it stops playing."""
        if phase == self.phase:
            return
        self.phase = phase
        if phase != "playing":
            self.release_all()
            self.reset_tracking()
        if phase == "playing":
            self.voted = False  # vote again next time in the lobby
            self.vote_tries = 0
        # Without auto mode it can't tell rounds from the lobby.
        print("[AI playing]" if phase == "playing" and self.lobby is None
              else PHASE_MESSAGES[phase])
        self.log_event(t, phase)

    def check_round(self, frame_bgr, t):
        """Auto mode: pause in the lobby, play when a round starts."""
        if self.phase == "playing" and self.frame_count % LOBBY_CHECK_EVERY:
            return
        if self.lobby.in_lobby(frame_bgr):
            self.lobby_streak += 1
            self.round_streak = 0
        else:
            self.round_streak += 1
            self.lobby_streak = 0
        if self.lobby_streak >= LOBBY_CONFIRM_CHECKS:
            self.set_phase("lobby", t)
        elif self.round_streak >= ROUND_CONFIRM_CHECKS:
            self.set_phase("playing", t)

    def in_lobby(self, frame_bgr, t, region):
        """Auto mode, while waiting in the lobby: vote, and keep a few frames."""
        if self.vote_button is not None and not self.voted and \
                self.vote_tries <= VOTE_MAX_TRIES and t >= self.vote_next_t and \
                self.frame_count % VOTE_CHECK_EVERY == 0:
            spot = self.vote_button.find(frame_bgr)
            if spot is not None and self.vote_button.ticked(frame_bgr, spot):
                # The game shows a tick on the mode you voted for.
                self.voted = True
                print(f"[voted for {self.vote_mode}]")
                self.log_event(t, "vote")
            elif spot is not None and self.vote_tries < VOTE_MAX_TRIES:
                click_at(region["left"] + spot[0], region["top"] + spot[1],
                         region["width"], region["height"], self.ms.position)
                self.vote_tries += 1
                self.vote_next_t = t + VOTE_CHECK_AFTER_S  # then look for the tick
                self.log_event(t, "vote_click")
            elif spot is not None:
                self.vote_tries += 1  # gave up for this lobby visit
                print(f"[vote for {self.vote_mode} didn't register after {VOTE_MAX_TRIES} clicks]")
        if self._log_frames_dir is not None and t - self.last_lobby_frame_t >= LOBBY_FRAME_EVERY_S:
            self.last_lobby_frame_t = t
            name = f"{self._log_frame_idx:06d}.jpg"
            cv2.imwrite(str(self._log_frames_dir / name), frame_bgr)
            self._log_frame_idx += 1
            self.log_event(t, "lobby_frame", frame=name)

    def log_event(self, t, event, **extra):
        if self._log_file is not None:
            self._log_file.write(json.dumps({"t": t, "event": event, **extra}) + "\n")
            self._log_file.flush()

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
        choice = track_ball.prefer_red(near, red_target, (px, py), cores)
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

    def measure_motion(self, ball, t, character_xy, targeted):
        """Updates self.screen_vx and self.approach from this detection and
        the last one -- only when it's the same track, a frame or so apart,
        and the camera held still (else the motion is partly the camera's).
        While the camera is turning, the last clean measurement is kept for
        a moment instead -- the ball keeps flying the same way, and dropping
        its speed mid-turn would stop the camera following it.

        Exception: while you're targeted, two red detections in a row are
        the ball coming at you even when the track broke between them (it
        does, when it rushes in during a camera turn), so how fast it's
        closing in is still measured -- that's what times the block."""
        x, y, state, radius = ball
        distance = ((x - character_xy[0]) ** 2 + (y - character_xy[1]) ** 2) ** 0.5
        prev = self.prev_motion
        if prev is not None and 0 < t - prev[0] < MOTION_MIN_DT and self.track_len >= 1:
            return  # too soon to measure well -- keep the last values
        recent = prev is not None and 0 < t - prev[0] <= 0.2
        continued = recent and self.track_len >= 1
        incoming = recent and targeted and state == prev[3] == "targeting"
        # A turn hardly changes the ball's size, so that's measured either way.
        self.growth = (radius - prev[4]) / (t - prev[0]) if continued or incoming else 0.0
        if continued and self.camera.last_active < prev[0] - CAMERA_SETTLE_S:
            dt = t - prev[0]
            self.screen_vx = (x - prev[1]) / dt
            self.approach = (prev[2] - distance) / dt
            self.motion_t = t
        elif incoming:
            self.approach = (prev[2] - distance) / (t - prev[0])
        elif not continued or t - self.motion_t > MOTION_HOLD_S:
            self.screen_vx = self.approach = 0.0
        self.prev_motion = (t, x, distance, state, radius)

    # --- camera --------------------------------------------------------

    def search(self, now, speed):
        self.camera.turn(self.last_seen_side, CAMERA_SEARCH_TURN_S, now, speed)
        self.searching = True

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
            self.search(now, CAMERA_SEARCH_SPEED)
            return
        if ball is not None and not stable and self.searching:
            # Something ball-like came into view mid-sweep: hold still so it
            # can be followed for long enough to count. If it was nothing,
            # the sweep picks up again when it's gone.
            self.camera.stop()
            self.searching = False
        if stable:
            # Found it: stop sweeping at once. A sweep left running after the
            # ball came back into view swung it right across the screen and
            # broke the track just as it was coming at you.
            if self.searching:
                self.camera.stop()
                self.searching = False
            # Where it's heading, not just where it is (see CAMERA_LEAD_S) --
            # unless it's far away and not after you (CAMERA_FAR_RADIUS).
            self.last_far = ball[3] < CAMERA_FAR_RADIUS and not targeted
            lead = 0.0 if self.last_far else CAMERA_LEAD_S
            predicted_x = ball[0] + self.screen_vx * lead
            offset = (predicted_x - width / 2) / (width / 2)  # -1 = left edge, +1 = right edge
            self.last_seen_side = -1 if offset < 0 else 1
            beyond = abs(offset) - CAMERA_EDGE_ZONE
            if beyond > 0:
                duration = min(max(beyond * CAMERA_TURN_GAIN_S / (1 - CAMERA_EDGE_ZONE),
                                   CAMERA_MIN_TURN_S), CAMERA_MAX_TURN_S)
                self.camera.turn(self.last_seen_side, duration, now)
        elif ball is None:
            last = self.idle_search_from  # when the last idle sweep began
            if last is not None and now - last <= CAMERA_IDLE_SWEEP_S:
                self.search(now, CAMERA_IDLE_SWEEP_SPEED)  # keep sweeping
            elif self.missing_streak >= CAMERA_IDLE_SEARCH_AFTER_S * FPS and \
                    (last is None or now - last >= CAMERA_IDLE_SWEEP_EVERY_S):
                self.idle_search_from = now
                self.search(now, CAMERA_IDLE_SWEEP_SPEED)
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
        if self._log_frames_dir is not None and self.frame_count % LOG_FRAME_EVERY == 0:
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
            "screen_vx": round(self.screen_vx, 1),
            "approach": round(self.approach, 1),
            "growth": round(self.growth, 1),
            "track_len": self.track_len,
            "self_red": round(self.self_red, 4),
        }
        self._log_file.write(json.dumps(entry) + "\n")
        self._log_file.flush()

    # --- main loop -------------------------------------------------

    def run(self):
        interval = 1.0 / FPS
        region = self._sct.monitors[1]
        if self.screen is None:
            self.screen = Screen(region)
        print(f"Screen capture: {self.screen.method}, {FPS} fps")
        width, height = region["width"], region["height"]
        center_x, center_y = width / 2, height / 2
        self.camera.anchor = (region["left"] + width // 2, region["top"] + height // 2)
        roi = self.cfg["self_highlight_roi"]
        character_xy = ((roi["x0"] + roi["x1"]) / 2 * width, (roi["y0"] + roi["y1"]) / 2 * height)

        quit_name = self.quit_key_name.replace("_", " ").title()
        if self.lobby is not None:
            print(f"Auto mode: plays each round by itself and waits in the lobby between "
                  f"them. Insert = pause/resume, {quit_name} = quit.")
            print(f"Auto vote: {self.vote_mode or 'off'}"
                  + (" (votes once the vote panel shows in the lobby)" if self.vote_button else ""))
        else:
            print(f"Insert = toggle AI on/off, {quit_name} = quit.")
            print("Starting in OFF state -- press Insert when you're ready.")

        while not self.quit:
            start = time.time()
            with self._lock:
                enabled = self.enabled

            if enabled and not roblox_focused():
                self.set_phase("no_focus", start)
            elif enabled:
                self.frame_count += 1
                self.camera.update(start)
                capture_t = time.time()
                frame_bgr = self.screen.grab()
                if self.lobby is not None:
                    if self.phase == "no_focus":
                        self.phase = None  # back in Roblox: work out where we are
                    self.check_round(frame_bgr, capture_t)
                else:
                    self.set_phase("playing", capture_t)
                if self.phase == "lobby":
                    self.in_lobby(frame_bgr, capture_t, region)
                if self.phase != "playing":
                    time.sleep(max(0.0, interval - (time.time() - start)))
                    continue
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
                    self.last_model = None  # start fresh when it's seen again
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
                    if self.last_model is None or capture_t - self.last_model_t >= MODEL_INTERVAL_S:
                        if self.camera.last_active >= self.last_feature_t:
                            self.features.reset()
                        row = self.features.update(x, y, state, radius, self.self_red,
                                                   capture_t, center_x, center_y)
                        self.last_feature_t = capture_t
                        desired, proba = self.predict_actions(row)
                        self.last_model, self.last_model_t = (row, desired, proba), capture_t
                    else:
                        row, desired, proba = self.last_model  # see MODEL_INTERVAL_S
                    self.measure_motion(ball, capture_t, character_xy, targeted)

                    if should_block("block" in desired, ball, targeted, character_xy,
                                    self.approach, self.growth, height):
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
    parser.add_argument("--auto", action="store_true",
                        help="Play by itself: play each round, wait in the lobby between rounds, "
                             "start again when the next round begins (default: start OFF, "
                             "Insert toggles)")
    parser.add_argument("--vote", choices=["none", *game_state.VOTE_MODES], default="none",
                        help="With --auto: vote for this gamemode each time in the lobby")
    parser.add_argument("--learned-block", action="store_true",
                        help="Use the block timing learned from your own logged runs "
                             "(learned_block.json, made by learn_block.py) instead of the "
                             "built-in BLOCK_3D_DIST")
    parser.add_argument("--quit-key", default=DEFAULT_QUIT_KEY,
                        help="Key that quits the AI (default: end). E.g. home, delete, "
                             "page_down, f8")
    args = parser.parse_args()
    try:
        parse_key(args.quit_key)
    except ValueError as e:
        parser.error(str(e))
    if args.log == "auto":
        Path("live_logs").mkdir(exist_ok=True)
        args.log = str(Path("live_logs") / f"live_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")

    if args.camera_test:
        run_camera_test(Camera(args.camera, args.camera_invert,
                               keyboard.Controller(), mouse.Controller()))
        return
    if not args.model:
        parser.error("Provide model.joblib (or use --camera-test).")

    if args.learned_block:
        use_learned_block_timing()

    controller = AIController(args.model, args.camera, args.camera_invert, log_path=args.log,
                              use_classifier=not args.no_classifier, auto=args.auto,
                              quit_key=args.quit_key,
                              vote=None if args.vote == "none" else args.vote)
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
        if controller.screen is not None:
            controller.screen.close()
        if controller._log_file is not None:
            controller._log_file.close()
        print("Exited, all keys/buttons released.")


if __name__ == "__main__":
    main()
