"""
Blade Ball AI control panel -- a small window with buttons for everything:
play, record, update the model, and score how runs went.

USAGE:
    python app.py

It runs the same scripts you'd run by hand (play_live.py, record_gameplay.py,
track_ball.py, ball_classifier.py, build_dataset.py, train_model.py,
score_logs.py) and shows their output in the window. The in-game hotkeys still work while they run:
Insert toggles, End quits.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from pynput import keyboard

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


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

        pad = {"padx": 8, "pady": 4}
        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)

        # --- Play -------------------------------------------------------
        play = ttk.LabelFrame(main, text="Play", padding=8)
        play.pack(fill="x", **pad)
        ttk.Label(play, text="Camera:").grid(row=0, column=0, sticky="w")
        self.camera = tk.StringVar(value="mouse")
        ttk.Combobox(play, textvariable=self.camera, values=["mouse", "keys", "off"],
                     state="readonly", width=8).grid(row=0, column=1, sticky="w")
        self.invert = tk.BooleanVar(value=False)
        ttk.Checkbutton(play, text="Invert camera", variable=self.invert).grid(row=0, column=2, sticky="w", padx=8)
        self.log = tk.BooleanVar(value=True)
        ttk.Checkbutton(play, text="Save a log of this run", variable=self.log).grid(row=0, column=3, sticky="w")
        self.add_button(play, "Start AI", self.start_ai).grid(row=1, column=0, columnspan=2, sticky="we", pady=4)
        self.add_button(play, "Test camera", self.test_camera).grid(row=1, column=2, sticky="we", padx=8, pady=4)
        ttk.Label(play, text="In Roblox: Insert = AI on/off, End = quit",
                  foreground="gray").grid(row=2, column=0, columnspan=4, sticky="w")

        # --- Record -------------------------------------------------------
        record = ttk.LabelFrame(main, text="Record your own gameplay (to teach the AI)", padding=8)
        record.pack(fill="x", **pad)
        self.add_button(record, "Start recorder", self.start_recorder).grid(row=0, column=0, sticky="w")
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
        results = ttk.LabelFrame(main, text="How did it go?", padding=8)
        results.pack(fill="x", **pad)
        self.add_button(results, "Score latest run", lambda: self.score(all_logs=False)).grid(row=0, column=0)
        self.add_button(results, "Score all runs", lambda: self.score(all_logs=True)).grid(row=0, column=1, padx=8)
        ttk.Button(results, text="Open logs folder", command=self.open_logs).grid(row=0, column=2)

        # --- Status / output ---------------------------------------------
        bar = ttk.Frame(main)
        bar.pack(fill="x", **pad)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(bar, textvariable=self.status).pack(side="left")
        self.stop_button = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="right")

        self.text = ScrolledText(main, height=18, width=100, font=("Consolas", 9))
        self.text.pack(fill="both", expand=True, **pad)

        self.root.after(100, self.pump_output)

    # --- running scripts ---------------------------------------------------

    def add_button(self, parent, label, command):
        button = ttk.Button(parent, text=label, command=command)
        self.busy_buttons.append(button)
        return button

    def run(self, title, steps):
        """Run steps -- a list of (description, argv) -- one after another
        in the background, streaming their output into the window."""
        self.stopping = False
        self.set_busy(True, title)
        threading.Thread(target=self._run_steps, args=(steps,), daemon=True).start()

    def _run_steps(self, steps):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        ok = True
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
        self.proc = None
        self.output.put(("done", "Stopped." if self.stopping else "Done." if ok else "Failed -- see output."))

    def pump_output(self):
        try:
            while True:
                item = self.output.get_nowait()
                if isinstance(item, tuple):
                    self.set_busy(False, item[1])
                else:
                    self.text.insert("end", item)
                    self.text.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self.pump_output)

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
        kb.press(keyboard.Key.end)
        kb.release(keyboard.Key.end)

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
        argv = [PYTHON, "play_live.py", "model.joblib", "--camera", self.camera.get()]
        if self.invert.get():
            argv.append("--camera-invert")
        if self.log.get():
            argv.append("--log")
        self.run("AI running -- switch to Roblox and press Insert to turn it on.",
                 [("Starting the AI", argv)])

    def test_camera(self):
        argv = [PYTHON, "play_live.py", "--camera-test", "--camera", self.camera.get()]
        if self.invert.get():
            argv.append("--camera-invert")
        self.run("Camera test -- switch to Roblox now.", [("Testing the camera", argv)])

    def start_recorder(self):
        self.run("Recorder running -- switch to Roblox and press Insert to start recording.",
                 [("Starting the recorder", [PYTHON, "record_gameplay.py"])])

    def update_model(self):
        steps = [(f"Tracking the ball in {s.name}", [PYTHON, "track_ball.py", str(s)])
                 for s in sessions_needing_tracking(self.retrack_all.get())]
        if self.retrain_detector.get():
            steps.append(("Training the ball detector", [PYTHON, "ball_classifier.py"]))
        steps += [
            ("Building the dataset", [PYTHON, "build_dataset.py", *sorted(
                str(s) for s in (HERE / "recordings").glob("*/")), "-o", "dataset.csv"]),
            ("Training the model", [PYTHON, "train_model.py", "dataset.csv", "-o", "model.joblib"]),
        ]
        self.run(f"Updating the model ({len(steps)} steps)...", steps)

    def score(self, all_logs):
        argv = [PYTHON, "score_logs.py"] + (["--all"] if all_logs else [])
        self.run("Scoring...", [("Scoring runs", argv)])

    def open_logs(self):
        logs = HERE / "live_logs"
        logs.mkdir(exist_ok=True)
        os.startfile(logs)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
