"""
Blade Ball AI control panel -- a small window with buttons for everything:
play, record, update the model, and score how runs went.

USAGE:
    python app.py

It runs the same scripts you'd run by hand (play_live.py, record_gameplay.py,
track_ball.py, ball_classifier.py, build_dataset.py, train_model.py,
score_logs.py) and shows their output in the window. The in-game hotkeys still work while they run:
Insert pauses/resumes, and the stop key chosen in the panel (End by default) quits the AI.
"""

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from pynput import keyboard

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


# The big banner at the top of the panel: (text, background, text color).
# It says plainly what the AI or recorder is doing right now -- loaded but
# waiting, playing, paused, recording... ({key} = the AI's stop key.)
BANNERS = {
    "idle": ("Nothing running.", "#e4e4e4", "#555555"),
    "ai_loading": ("Loading the AI...", "#fff1c2", "#6b4d00"),
    "ai_off": ("AI is loaded but NOT playing yet.\n"
               "Switch to Roblox and press Insert to turn it on.", "#ffd966", "#4d3800"),
    "ai_armed": ("AI is ON -- click into Roblox.\n"
                 "It plays rounds by itself and waits in the lobby between them.",
                 "#c9e0f7", "#123a5c"),
    "ai_on": ("AI is PLAYING.\nInsert = pause,  {key} (or Stop) = quit.", "#b6e3a8", "#1e4d12"),
    "ai_lobby": ("AI is waiting in the lobby.\n"
                 "It starts by itself when the next round begins.  {key} = quit.",
                 "#c9e0f7", "#123a5c"),
    "ai_no_focus": ("AI is ON but waiting for Roblox.\n"
                    "Click into the Roblox window -- it only plays while Roblox is active.",
                    "#ffd966", "#4d3800"),
    "ai_paused": ("AI is PAUSED.\nPress Insert in Roblox to turn it back on.",
                  "#ffd966", "#4d3800"),
    "rec_loading": ("Loading the recorder...", "#fff1c2", "#6b4d00"),
    "rec_ready": ("Recorder is ready but NOT recording yet.\n"
                  "Switch to Roblox and press Insert to start recording.", "#ffd966", "#4d3800"),
    "rec_on": ("RECORDING.\nPress Insert in Roblox to stop (and save) this recording.",
               "#f4a6a6", "#5c0f0f"),
    "rec_saved": ("Recording saved -- recorder still ready.\n"
                  "Press Insert in Roblox to record another, or Stop when done.",
                  "#ffd966", "#4d3800"),
    "camera_test": ("Camera test -- switch to Roblox now.\n"
                    "The view should turn left, then right.", "#c9e0f7", "#123a5c"),
    "working": ("Updating the model... (this can take a while)", "#c9e0f7", "#123a5c"),
}

# Lines the scripts print that mean their state changed -> banner to show.
BANNER_TRIGGERS = [
    ("Starting in OFF state", "ai_off"),
    ("Auto mode:", "ai_armed"),
    ("[AI ENABLED]", "ai_armed"),
    ("[AI playing]", "ai_on"),
    ("[in a round", "ai_on"),
    ("[in the lobby", "ai_lobby"),
    ("[waiting for the Roblox window", "ai_no_focus"),
    ("[AI disabled]", "ai_paused"),
    ("Blade Ball gameplay recorder", "rec_ready"),
    ("[recording started]", "rec_on"),
    ("[recording stopped]", "rec_saved"),
]


# Keys the AI can be told to quit on: shown name -> play_live.py --quit-key.
# Keys games rarely use, so the AI can't be stopped by accident.
STOP_KEYS = {
    "End": "end", "Home": "home", "Delete": "delete", "Page Up": "page_up",
    "Page Down": "page_down", "Pause": "pause", "Scroll Lock": "scroll_lock",
    "F6": "f6", "F7": "f7", "F8": "f8", "F9": "f9", "F10": "f10", "F12": "f12",
}

# The panel's choices (camera, stop key, ...) are remembered between runs.
SETTINGS_PATH = HERE / "panel_settings.json"


def load_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def sessions_needing_tracking(retrack_all):
    """Recording sessions with no ball_tracks.jsonl, or one made by an older
    track_ball.py (missing fields build_dataset.py needs)."""
    needing = []
    for session in sorted((HERE / "recordings").glob("*/")):
        tracks = session / "ball_tracks.jsonl"
        if retrack_all or not tracks.exists():
            needing.append(session)
            continue
        with open(tracks) as f:
            first = f.readline()
        if not first.strip() or "self_red" not in json.loads(first):
            needing.append(session)
    return needing


class App:
    def __init__(self, root):
        self.root = root
        root.title("Blade Ball AI")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.output = queue.Queue()
        self.proc = None
        self.stopping = False
        self.busy_buttons = []

        self.proc_quit_key = keyboard.Key.end  # what Stop presses for the running script
        saved = load_settings()
        pad = {"padx": 8, "pady": 4}
        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)

        # --- What's happening right now (see BANNERS) ---------------------
        self.banner = tk.Label(main, font=("Segoe UI", 13, "bold"), justify="center",
                               padx=10, pady=10)
        self.banner.pack(fill="x", **pad)
        self.set_banner("idle")

        # --- Play -------------------------------------------------------
        play = ttk.LabelFrame(main, text="Play", padding=8)
        play.pack(fill="x", **pad)
        ttk.Label(play, text="Camera:").grid(row=0, column=0, sticky="w")
        self.camera = tk.StringVar(value=saved.get("camera", "mouse"))
        ttk.Combobox(play, textvariable=self.camera, values=["mouse", "keys", "off"],
                     state="readonly", width=8).grid(row=0, column=1, sticky="w")
        self.invert = tk.BooleanVar(value=saved.get("invert", False))
        ttk.Checkbutton(play, text="Invert camera", variable=self.invert).grid(row=0, column=2, sticky="w", padx=8)
        self.log = tk.BooleanVar(value=saved.get("log", True))
        ttk.Checkbutton(play, text="Save a log of this run", variable=self.log).grid(row=0, column=3, sticky="w")
        self.auto = tk.BooleanVar(value=saved.get("auto", True))
        ttk.Checkbutton(play, text="Play rounds by itself (waits in the lobby between rounds)",
                        variable=self.auto, command=self.update_play_hint).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(play, text="Stop key:").grid(row=2, column=0, sticky="w")
        self.stop_key = tk.StringVar(value=saved.get("stop_key", "End"))
        if self.stop_key.get() not in STOP_KEYS:
            self.stop_key.set("End")
        stop_box = ttk.Combobox(play, textvariable=self.stop_key, values=list(STOP_KEYS),
                                state="readonly", width=11)
        stop_box.grid(row=2, column=1, sticky="w")
        stop_box.bind("<<ComboboxSelected>>", lambda _: self.update_play_hint())
        self.add_button(play, "Start AI", self.start_ai).grid(row=3, column=0, columnspan=2, sticky="we", pady=4)
        self.add_button(play, "Test camera", self.test_camera).grid(row=3, column=2, sticky="we", padx=8, pady=4)
        self.play_hint = tk.StringVar()
        ttk.Label(play, textvariable=self.play_hint, foreground="gray").grid(
            row=4, column=0, columnspan=4, sticky="w")
        self.update_play_hint()

        # --- Record -------------------------------------------------------
        record = ttk.LabelFrame(main, text="Record your own gameplay (to teach the AI)", padding=8)
        record.pack(fill="x", **pad)
        self.add_button(record, "Start recorder (then press Insert in Roblox)", self.start_recorder).grid(row=0, column=0, sticky="w")
        ttk.Label(record, text="In Roblox: Insert = start/stop a recording, End = quit",
                  foreground="gray").grid(row=0, column=1, sticky="w", padx=8)

        # --- Update ------------------------------------------------------
        update = ttk.LabelFrame(main, text="Update the AI from recordings", padding=8)
        update.pack(fill="x", **pad)
        self.retrack_all = tk.BooleanVar(value=False)
        self.retrain_detector = tk.BooleanVar(value=False)
        self.add_button(update, "Update model", self.update_model).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(update, text="Re-track all recordings (slow; only after detection changes)",
                        variable=self.retrack_all).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Checkbutton(update, text="Also retrain the ball detector (slow; after new recordings)",
                        variable=self.retrain_detector).grid(row=1, column=1, sticky="w", padx=8)

        # --- Results -----------------------------------------------------
        # These only read log files, so they work any time -- even while the
        # AI is playing -- instead of waiting for the running script.
        results = ttk.LabelFrame(main, text="How did it go?", padding=8)
        results.pack(fill="x", **pad)
        ttk.Button(results, text="Score latest run",
                   command=lambda: self.score(all_logs=False)).grid(row=0, column=0)
        ttk.Button(results, text="Score all runs",
                   command=lambda: self.score(all_logs=True)).grid(row=0, column=1, padx=8)
        ttk.Button(results, text="Open logs folder", command=self.open_logs).grid(row=0, column=2)

        # --- Status / output ---------------------------------------------
        bar = ttk.Frame(main)
        bar.pack(fill="x", **pad)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(bar, textvariable=self.status).pack(side="left")
        self.stop_button = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="right")
        # Says why the other buttons are greyed out while something runs.
        self.hint = tk.StringVar(value="")
        tk.Label(main, textvariable=self.hint, fg="#c05800", anchor="w").pack(fill="x", padx=8)

        self.text = ScrolledText(main, height=18, width=100, font=("Consolas", 9))
        self.text.pack(fill="both", expand=True, **pad)

        self.root.after(100, self.pump_output)

    # --- running scripts ---------------------------------------------------

    def add_button(self, parent, label, command):
        button = ttk.Button(parent, text=label, command=command)
        self.busy_buttons.append(button)
        return button

    def run(self, title, steps, what, banner):
        """Run steps -- a list of (description, argv) -- one after another
        in the background, streaming their output into the window. `what`
        names it in the "press Stop" hint (e.g. "The AI"); `banner` is the
        BANNERS entry to show until the script reports otherwise."""
        self.stopping = False
        self.set_busy(True, title)
        self.set_banner(banner)
        self.hint.set(f"{what} is running -- press Stop before using the other buttons. "
                      f"Scoring works any time.")
        threading.Thread(target=self._run_steps, args=(steps,), daemon=True).start()

    def _run_steps(self, steps):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        ok = True
        try:
            for description, argv in steps:
                if self.stopping:
                    break
                self.output.put(f"\n>>> {description}\n")
                self.proc = subprocess.Popen(
                    argv, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                for line in self.proc.stdout:
                    self.output.put(line)
                if self.proc.wait() != 0 and not self.stopping:
                    self.output.put(f"\n!!! {description} failed (exit code {self.proc.returncode})\n")
                    ok = False
                    break
        except Exception as e:  # e.g. the script couldn't be started at all
            self.output.put(f"\n!!! {e}\n")
            ok = False
        finally:
            # Always unlock the buttons, whatever happened.
            self.proc = None
            self.output.put(("done", "Stopped." if self.stopping else "Done." if ok else "Failed -- see output."))

    def pump_output(self):
        try:
            while True:
                item = self.output.get_nowait()
                if isinstance(item, tuple) and item[0] == "close":
                    self.on_close()  # a newer panel was opened
                elif isinstance(item, tuple):
                    self.set_busy(False, item[1])
                    self.hint.set("")
                    self.set_banner("idle")
                else:
                    for trigger, banner in BANNER_TRIGGERS:
                        if trigger in item:
                            self.set_banner(banner)
                    self.text.insert("end", item)
                    self.text.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self.pump_output)

    def set_banner(self, name):
        text, bg, fg = BANNERS[name]
        key = self.stop_key.get() if hasattr(self, "stop_key") else "End"
        self.banner.config(text=text.replace("{key}", key), bg=bg, fg=fg)

    def update_play_hint(self):
        key = self.stop_key.get()
        if self.auto.get():
            self.play_hint.set(f"Plays each round, pauses in the lobby, carries on next round. "
                               f"In Roblox: Insert = pause/resume, {key} = stop.")
        else:
            self.play_hint.set(f"In Roblox: Insert = AI on/off, {key} = stop.")

    def save_settings(self):
        try:
            SETTINGS_PATH.write_text(json.dumps({
                "camera": self.camera.get(), "invert": self.invert.get(), "log": self.log.get(),
                "auto": self.auto.get(), "stop_key": self.stop_key.get()}, indent=2))
        except OSError:
            pass  # not worth failing a start over

    def set_busy(self, busy, status):
        self.status.set(status)
        for button in self.busy_buttons:
            button.state(["disabled"] if busy else ["!disabled"])
        self.stop_button.state(["!disabled"] if busy else ["disabled"])

    def stop(self):
        """Press End (the scripts' own quit key -- play_live.py releases every
        key/button on it), and only force-kill if that didn't work."""
        self.stopping = True
        proc = self.proc
        if proc is None:
            return
        self.status.set("Stopping...")
        kb = keyboard.Controller()
        kb.press(self.proc_quit_key)
        kb.release(self.proc_quit_key)

        def force_if_needed():
            if proc.poll() is None:
                proc.terminate()
        self.root.after(3000, force_if_needed)

    def on_close(self):
        if self.proc is not None:
            self.stop()
            self.root.after(3500, self.root.destroy)
        else:
            self.root.destroy()

    # --- button actions -----------------------------------------------------

    def start_ai(self):
        if not (HERE / "model.joblib").exists():
            self.status.set("No model.joblib yet -- record some gameplay, then Update model.")
            return
        self.save_settings()
        quit_key = STOP_KEYS[self.stop_key.get()]
        argv = [PYTHON, "play_live.py", "model.joblib", "--camera", self.camera.get(),
                "--quit-key", quit_key]
        if self.invert.get():
            argv.append("--camera-invert")
        if self.log.get():
            argv.append("--log")
        if not self.auto.get():
            argv.append("--no-auto")
        self.proc_quit_key = getattr(keyboard.Key, quit_key)
        self.run("AI running." if self.auto.get() else
                 "AI loaded -- press Insert in Roblox to turn it on.",
                 [("Starting the AI", argv)], "The AI", "ai_loading")

    def test_camera(self):
        self.proc_quit_key = keyboard.Key.end
        argv = [PYTHON, "play_live.py", "--camera-test", "--camera", self.camera.get()]
        if self.invert.get():
            argv.append("--camera-invert")
        self.run("Camera test -- switch to Roblox now.", [("Testing the camera", argv)],
                 "The camera test", "camera_test")

    def start_recorder(self):
        self.proc_quit_key = keyboard.Key.end
        self.run("Recorder loaded -- press Insert in Roblox to start recording.",
                 [("Starting the recorder", [PYTHON, "record_gameplay.py"])], "The recorder",
                 "rec_loading")

    def update_model(self):
        self.proc_quit_key = keyboard.Key.end
        steps = [(f"Tracking the ball in {s.name}", [PYTHON, "track_ball.py", str(s)])
                 for s in sessions_needing_tracking(self.retrack_all.get())]
        if self.retrain_detector.get():
            steps.append(("Training the ball detector", [PYTHON, "ball_classifier.py"]))
        steps += [
            ("Building the dataset", [PYTHON, "build_dataset.py", *sorted(
                str(s) for s in (HERE / "recordings").glob("*/")), "-o", "dataset.csv"]),
            ("Training the model", [PYTHON, "train_model.py", "dataset.csv", "-o", "model.joblib"]),
        ]
        self.run(f"Updating the model ({len(steps)} steps)...", steps, "Updating the model",
                 "working")

    def score(self, all_logs):
        """Runs alongside whatever else is running (it only reads logs)."""
        argv = [PYTHON, "score_logs.py"] + (["--all"] if all_logs else [])

        def work():
            result = subprocess.run(
                argv, cwd=HERE, capture_output=True, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.output.put("\n>>> Scoring runs\n" + result.stdout + result.stderr)
        threading.Thread(target=work, daemon=True).start()

    def open_logs(self):
        logs = HERE / "live_logs"
        logs.mkdir(exist_ok=True)
        os.startfile(logs)


# The open panel listens on this local port, so a newly opened one can tell
# it to close -- only one panel runs at a time.
INSTANCE_PORT = 48213


def close_previous_panel():
    """If a panel is already open, ask it to close and wait until it has."""
    try:
        with socket.create_connection(("127.0.0.1", INSTANCE_PORT), timeout=1) as conn:
            conn.sendall(b"close")
    except OSError:
        return  # none open
    # It stops whatever it's running first (up to ~3.5s), then exits.
    deadline = time.time() + 6
    while time.time() < deadline:
        time.sleep(0.2)
        try:
            socket.create_connection(("127.0.0.1", INSTANCE_PORT), timeout=0.2).close()
        except OSError:
            return


def listen_for_newer_panel(app):
    server = socket.socket()
    try:
        server.bind(("127.0.0.1", INSTANCE_PORT))
    except OSError:
        return  # port taken by something else -- just skip the one-panel check
    server.listen()

    def serve():
        while True:
            conn, _ = server.accept()
            with conn:
                if conn.recv(16) == b"close":
                    app.output.put(("close",))  # handled on the window's own thread

    threading.Thread(target=serve, daemon=True).start()


def main():
    close_previous_panel()
    root = tk.Tk()
    app = App(root)
    listen_for_newer_panel(app)
    root.mainloop()


if __name__ == "__main__":
    main()
