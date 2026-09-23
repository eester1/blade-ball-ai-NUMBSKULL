"""
Blade Ball gameplay recorder.

Captures your screen at a fixed rate and logs which keys/mouse buttons
are held at each captured frame. This produces a dataset you can later
use to train a model to predict actions from screen state
(imitation learning / behavior cloning).

SETUP (run once):
    pip install mss pynput pillow

USAGE:
    python record_gameplay.py

    - Press Insert to start/stop a recording session.
    - Press End to quit the program entirely.
    - (These are chosen to avoid colliding with Roblox's own hotkeys,
      e.g. F9 opens Roblox's performance stats overlay and ESC opens
      Roblox's in-game menu.)
    - Each session is saved to recordings/session_<timestamp>/
        frames/000001.jpg, 000002.jpg, ...
        inputs.jsonl   <- one JSON line per frame with the input state

NOTES:
    - Tune MONITOR_REGION below if Roblox isn't fullscreen on your
      primary monitor, or leave it as None to capture the whole
      primary monitor.
    - Tune FPS. 15-20 is usually plenty for a reaction-timing game
      like Blade Ball and keeps file sizes/training time manageable.
    - Tune TRACKED_KEYS / TRACKED_MOUSE_BUTTONS to match whatever
      keybinds you actually use in-game.
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime

import mss
from PIL import Image
from pynput import keyboard, mouse

# ---------------------------------------------------------------------------
# Config - edit these to match your setup
# ---------------------------------------------------------------------------

FPS = 15
OUTPUT_ROOT = Path("recordings")
JPEG_QUALITY = 80

# None = capture the whole primary monitor. Or set explicit region, e.g.:
# MONITOR_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080}
MONITOR_REGION = None

TRACKED_KEYS = {"w", "a", "s", "d"}
TRACKED_MOUSE_BUTTONS = {"left", "right"}  # left = block/deflect, right = abilities

START_STOP_KEY = keyboard.Key.insert
QUIT_KEY = keyboard.Key.end

# ---------------------------------------------------------------------------

class InputState:
    """Thread-safe record of which tracked keys/buttons are currently held."""

    def __init__(self):
        self._lock = threading.Lock()
        self._held = set()

    def press(self, name):
        with self._lock:
            self._held.add(name)

    def release(self, name):
        with self._lock:
            self._held.discard(name)

    def snapshot(self):
        with self._lock:
            return sorted(self._held)


def key_to_name(key):
    """Normalize a pynput key event to a simple string, or None if untracked."""
    try:
        name = key.char.lower()
    except AttributeError:
        name = str(key).replace("Key.", "").lower()
    return name


class Recorder:
    def __init__(self):
        self.input_state = InputState()
        self.recording = False
        self.quit = False
        self.session_dir = None
        self.frames_dir = None
        self.log_file = None
        self.frame_index = 0
        self._sct = mss.mss()
        self._lock = threading.Lock()  # guards recording/log_file against the capture thread

    # --- session lifecycle ------------------------------------------------

    def start_session(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = OUTPUT_ROOT / f"session_{ts}"
        frames_dir = session_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(session_dir / "inputs.jsonl", "a")
        with self._lock:
            self.session_dir = session_dir
            self.frames_dir = frames_dir
            self.log_file = log_file
            self.frame_index = 0
            self.recording = True
        print(f"[recording started] -> {session_dir.resolve()}")

    def stop_session(self):
        with self._lock:
            self.recording = False
            log_file = self.log_file
            self.log_file = None
        if log_file:
            log_file.close()
        print(f"[recording stopped] {self.frame_index} frames saved to {self.session_dir.resolve()}")

    def toggle(self):
        if self.recording:
            self.stop_session()
        else:
            self.start_session()

    # --- capture loop -------------------------------------------------

    def capture_loop(self):
        interval = 1.0 / FPS
        region = MONITOR_REGION or self._sct.monitors[1]
        while not self.quit:
            start = time.time()
            with self._lock:
                recording = self.recording
                log_file = self.log_file
                frames_dir = self.frames_dir
            if recording and log_file is not None:
                self._capture_frame(region, log_file, frames_dir)
            elapsed = time.time() - start
            time.sleep(max(0.0, interval - elapsed))

    def _capture_frame(self, region, log_file, frames_dir):
        shot = self._sct.grab(region)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        self.frame_index += 1
        frame_name = f"{self.frame_index:06d}.jpg"
        img.save(frames_dir / frame_name, quality=JPEG_QUALITY)

        record = {
            "frame": frame_name,
            "t": time.time(),
            "held": self.input_state.snapshot(),
        }
        try:
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
        except (ValueError, AttributeError):
            pass  # recording was stopped mid-write -- just drop this last frame safely

    # --- input listeners -------------------------------------------------

    def on_key_press(self, key):
        if key == START_STOP_KEY:
            self.toggle()
            return
        if key == QUIT_KEY:
            self.quit = True
            if self.recording:
                self.stop_session()
            return False  # stop the keyboard listener

        name = key_to_name(key)
        if name in TRACKED_KEYS:
            self.input_state.press(name)

    def on_key_release(self, key):
        name = key_to_name(key)
        if name in TRACKED_KEYS:
            self.input_state.release(name)

    def on_click(self, x, y, button, pressed):
        name = str(button).replace("Button.", "").lower()
        if name in TRACKED_MOUSE_BUTTONS:
            if pressed:
                self.input_state.press(f"mouse_{name}")
            else:
                self.input_state.release(f"mouse_{name}")


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    recorder = Recorder()

    print("Blade Ball gameplay recorder")
    print(f"  Insert = start/stop recording (FPS={FPS})")
    print("  End    = quit")
    print("Switch to Roblox now. Recording will save under ./recordings/\n")

    capture_thread = threading.Thread(target=recorder.capture_loop, daemon=True)
    capture_thread.start()

    mouse_listener = mouse.Listener(on_click=recorder.on_click)
    mouse_listener.start()

    with keyboard.Listener(
        on_press=recorder.on_key_press, on_release=recorder.on_key_release
    ) as kb_listener:
        kb_listener.join()

    mouse_listener.stop()
    print("Exited.")


if __name__ == "__main__":
    main()