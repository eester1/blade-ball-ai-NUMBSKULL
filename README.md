# Blade Ball AI

An imitation-learning bot for [Blade Ball](https://www.roblox.com/games) (Roblox). It watches your screen, learns from recordings of you playing, and can then play (dodge/block) on its own by mimicking what you did in similar situations.

There is no game-engine integration and no memory reading — everything is done by taking screenshots and detecting the ball with computer vision, then predicting keyboard/mouse actions with a small neural network trained on your own recorded games.

> **Status:** functional but not reliable enough to play unattended yet. See [Known Limitations](#known-limitations) before expecting too much of it.

## How it works

```
record_gameplay.py          track_ball.py            build_dataset.py        train_model.py         play_live.py
─────────────────────       ─────────────────        ──────────────────      ─────────────────      ─────────────────
Screenshot + input     -->  Detect ball position  --> Merge into one     --> Train a small     --> Detect ball live,
logger while you play       (color/shape based)        dataset.csv           MLP classifier         predict + press
                             per recorded session                             (model.joblib)         keys/mouse live
```

1. **Record** — `record_gameplay.py` captures your screen at a fixed rate while logging which keys/mouse buttons are held, producing one folder per session under `recordings/`.
2. **Track** — `track_ball.py` re-processes each recorded session's frames offline to find the ball's pixel position (and whether it's idle/white or targeting/red) in each frame, using color thresholding + shape filtering + temporal heuristics (see [Ball Detection](#ball-detection-track_ballpy)).
3. **Build dataset** — `build_dataset.py` merges the recorded inputs and tracked ball positions from one or more sessions into a single `dataset.csv`, computing derived features (relative position, velocity in px/s, distance, closing speed).
4. **Train** — `train_model.py` trains a small multi-label neural network (`model.joblib`) to predict which keys/buttons you'd hold, given the ball's state.
5. **Play live** — `play_live.py` runs the same ball-detection code in real time, feeds the same features into the trained model, and presses/releases keys and mouse buttons to match its predictions.

## Setup

```
pip install mss pynput joblib pandas scikit-learn opencv-python numpy pillow
```

Tested on Windows with Roblox running in a window (or fullscreen) on the primary monitor.

## Full workflow

### 1. Record gameplay

```
python record_gameplay.py
```

- Press **Insert** to start/stop a recording session, **End** to quit the program.
- Each session is saved to `recordings/session_<timestamp>/`:
  - `frames/000001.jpg, 000002.jpg, ...` — one screenshot per captured frame
  - `inputs.jsonl` — one JSON line per frame: `{"frame": "...", "t": <unix timestamp>, "held": [...]}`
- Play normally. Record several sessions across different maps and rounds — the model can only learn patterns that actually appear in your recordings, including rare ones like blocking and using your ability.
- Tunables at the top of the file: `FPS` (capture rate), `MONITOR_REGION` (defaults to the whole primary monitor), `TRACKED_KEYS`/`TRACKED_MOUSE_BUTTONS`.

### 2. Tune ball color detection (once, or whenever detection looks wrong)

```
python track_ball.py --tune recordings/session_XXXX/frames/000100.jpg
```

Opens a window with sliders for the white-ball (idle) and red-ball (targeting) HSV ranges. Adjust until **only** the ball is highlighted in the mask preview — not chat text, not UI icons, not other players. Press `s` to save to `color_config.json` (used automatically after that), or `q` to discard.

### 3. Detect the ball in each recorded session

```
python track_ball.py recordings/session_XXXX
```

Writes `ball_tracks.jsonl` into that session folder — one line per frame with the detected `ball_x`, `ball_y`, and `state` (`"idle"` or `"targeting"`), or `null` if nothing was found that frame.

Optional visual spot-check:

```
python track_ball.py recordings/session_XXXX --debug --debug-every 20
```

Saves every 20th frame, annotated with a circle around the detection, into `session_XXXX/debug_frames/`.

### 4. Build the training dataset

```
python build_dataset.py recordings/session_A recordings/session_B ... -o dataset.csv
```

(Or `recordings/*/` to include everything.) Merges inputs + ball tracks from each session, drops frames where the ball wasn't detected, and computes:

| Column | Meaning |
|---|---|
| `ball_x`, `ball_y` | Raw detected pixel position |
| `ball_rel_x`, `ball_rel_y` | Position relative to screen center |
| `ball_vel_x`, `ball_vel_y` | Velocity in **pixels per second** (real elapsed time between detected frames, not frame count — matters because live inference doesn't run at a perfectly steady rate) |
| `ball_distance` | Distance from screen center |
| `ball_closing_speed` | Positive = ball closing in on you, negative = moving away |
| `ball_state` | `"idle"` or `"targeting"` |
| `held_w`/`a`/`s`/`d`/`mouse_left`/`mouse_right` | 1 if held that frame, else 0 (the training labels) |

### 5. Train the model

```
python train_model.py dataset.csv -o model.joblib
```

- Trains an `MLPClassifier` (scikit-learn) with hidden layers `[32, 16]` by default (`--hidden-sizes` to change), on 7 input features and 6 output labels (see [Model](#model-train_modelpy)).
- Splits each session's data by time (last 20% held out as test set, not a random split — avoids testing on near-duplicate frames the model basically already saw).
- Oversamples rare rows in the *training* set only: frames with blocking/ability held (5x), and frames in the fastest 15% of ball speed (3x) — both are underrepresented relative to how much they matter.
- Prints a per-action classification report (precision/recall/F1) on the held-out test set.

Re-run this any time you've recorded more sessions and rebuilt a bigger `dataset.csv` — it always trains fresh from scratch.

### 6. Play live

```
python play_live.py model.joblib
```

- **Insert** toggles AI control on/off (starts off). **End** quits immediately and always releases every key/button first, so nothing gets stuck held down.
- **Safety:** while ON, this sends real keyboard/mouse input to whatever window is focused. Keep Roblox focused and your hand near the keyboard the first few times, in case it does something unwanted — End stops everything instantly.
- Optional diagnostics: `--log path.jsonl` writes one JSON line per live-inferred frame (ball position/velocity/state, the model's per-action confidence, actions taken) **and** saves every captured frame as a JPEG into `path_frames/`. This is the main tool for root-causing "why did it do that" after a session — see [Diagnosing bad behavior](#diagnosing-bad-behavior).

## Ball detection (`track_ball.py`)

The ball is found by plain color thresholding, not a trained model: it appears as a white/grey disc when idle and switches to red when it's targeting you. `find_ball_candidates()` returns *every* blob in the frame that passes color + area + circularity filtering — often more than one, since other round pale/red things (another player's head, a skill effect, map decorations) can pass the same filter.

Because a single frame's color/shape signal alone can't reliably tell the real ball apart from those decoys, there's a layer of temporal reasoning on top, controlled by `DEFAULT_CONFIG` in `track_ball.py` (or `color_config.json` if you've saved tuned values):

| Config key | What it does |
|---|---|
| `white_sat_max`, `white_val_min` | HSV range for the idle (white/grey) ball |
| `red_hue_low_max`, `red_hue_high_min`, `red_sat_min`, `red_val_min` | HSV range for the targeting (red) ball — red wraps around hue 0, so it's two ranges |
| `min_area`, `max_area`, `min_circularity` | Shape filter — rejects blobs that are the wrong size or not round enough |
| `ui_mask_regions` | Fractional screen regions ignored entirely (HUD panels, stat bar, block/ability icons, a couple of static decorations found to cause false positives) |
| `max_jump_px_per_frame`, `reset_after_missing_frames`, `confirm_lookahead_frames` | A detection far from the last trusted position isn't rejected outright — it's trusted if the *next* detected frame keeps going near it (real fast movement/bounces continue, a one-off flash doesn't) |
| `stale_after_frames`, `stale_jitter_px` | If the trusted position hasn't moved at all for this many frames, stop trusting proximity to it and force a fresh whole-frame search — catches long lock-ons onto something static (a decoration, a standing player) that a one-frame check can't |

`track_ball.py` (batch, processing a whole recorded session) and `play_live.py` (real-time) both use this same candidate list + proximity/staleness logic, via shared functions (`find_ball_candidates`, `find_ball_near`, `closest_to`, `most_circular`) — the batch version can additionally peek one frame into the future to confirm a jump, which live inference can't do (it uses a one-frame "pending" delay instead).

## Model (`train_model.py`)

A small multi-label MLP — no vision component. `track_ball.py` handles "seeing" the ball; the model only ever sees 7 numeric features:

**Inputs:** `ball_rel_x`, `ball_rel_y`, `ball_vel_x`, `ball_vel_y`, `ball_distance`, `ball_closing_speed`, `state_targeting`

**Outputs (independent per-action confidence, each with its own decision threshold in `play_live.py`):** `held_w`, `held_a`, `held_s`, `held_d`, `held_mouse_left` (block), `held_mouse_right` (ability)

Features are normalized with a `StandardScaler` fit on the training data (saved alongside the model in `model.joblib`, so live inference scales consistently).

## Diagnosing bad behavior

If the AI does something confusing live, don't guess — capture it:

```
python play_live.py model.joblib --log session.jsonl
```

This gives you, for every frame the AI acted on: the detected ball position/velocity/state, the model's confidence for each action, and what it actually pressed — plus the actual frame image in `session_frames/`, so you can visually confirm what was really being tracked at any point. This has already been essential for catching cases where detection locked onto the wrong thing entirely (see below).

## Known Limitations

- **Ball detection can still lock onto decoys.** Map decorations (e.g. lit torches), other players' head cosmetics, and similar red/white objects can pass the same color/shape filter as the real ball. Heuristic fixes (proximity tracking, a staleness watchdog, one-frame confirmation) catch most cases but not all — a decoy that's visually stable for a couple of consecutive frames can still slip through. The durable fix would be a small trained classifier scoring candidate crops instead of picking by circularity/proximity alone (data for this is nearly free to generate from existing recordings — every non-selected candidate in an already-tracked frame is an automatic negative example). Not yet built.
- **No camera control.** `play_live.py` never moves the mouse to look around — whatever direction Roblox's camera happens to be facing is what it sees. Since movement keys are camera-relative, this can bias which direction it moves. Not yet addressed; needs either steering the camera toward the ball (which would fight the model's existing position-based features) or toward a fixed reference point (which needs a way to know that direction that this vision-only pipeline doesn't currently have).
- **Model accuracy is modest**, especially for blocking and rarer actions — recall on `held_mouse_left` (block) and `held_s` fluctuates in the 0.15–0.45 range across retrains, and `held_mouse_right` has very few training examples. This is a data-volume problem more than an architecture problem: record more sessions, especially ones where you actually block/use your ability a fair amount, and retrain.
- **Single-monitor, fixed-resolution assumption.** Ball detection and screen-center calculations assume the capture region matches between recording and live play.
