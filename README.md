# Blade Ball AI

An imitation-learning bot for [Blade Ball](https://www.roblox.com/games) (Roblox). It watches your screen, learns from recordings of you playing, and can then play (dodge/block) on its own by mimicking what you did in similar situations.

There is no game-engine integration and no memory reading — everything is done by taking screenshots and detecting the ball with computer vision, then predicting keyboard/mouse actions with a small neural network trained on your own recorded games.

> **Status:** functional but not reliable enough to play unattended yet. See [Known Limitations](#known-limitations) before expecting too much of it.

## How it works

```
record_gameplay.py          track_ball.py            build_dataset.py        train_model.py         play_live.py
─────────────────────       ─────────────────        ──────────────────      ─────────────────      ─────────────────
Screenshot + input     -->  Detect ball position, --> Merge into one     --> Train a small     --> Detect ball live,
logger while you play       state and size             dataset.csv           MLP classifier         turn camera to keep
                            per recorded session       (features.py)         (model.joblib)         it in view, predict
                                                                                                    + press actions
```

1. **Record** — `record_gameplay.py` captures your screen at a fixed rate while logging which keys/mouse buttons are held, producing one folder per session under `recordings/`.
2. **Track** — `track_ball.py` re-processes each recorded session's frames offline to find the ball's pixel position, whether it's idle (white) or targeting (red), and its apparent size in each frame, using color thresholding + shape filtering + temporal heuristics (see [Ball Detection](#ball-detection-track_ballpy)).
3. **Build dataset** — `build_dataset.py` merges the recorded inputs and tracked ball positions from one or more sessions into a single `dataset.csv`, computing features with `features.py` (see [Features](#features-featurespy)).
4. **Train** — `train_model.py` trains a small multi-label neural network (`model.joblib`) to predict which actions you'd take, given the ball's state.
5. **Play live** — `play_live.py` runs the same ball detection and the same `features.py` code in real time, turns the camera to keep the ball on screen, and presses/releases keys and mouse buttons to match the model's predictions.

## Setup

```
pip install mss pynput joblib pandas scikit-learn opencv-python numpy pillow
```

Tested on Windows with Roblox running in a window (or fullscreen) on the primary monitor.

## The easy way: the control panel

```
python app.py
```

This opens a window with a button for everything, so you never have to type the commands further down. Keep it open while you play. Starting a script from it always runs the latest version of that script, so you only need to reopen the panel after `app.py` itself changes. Opening a second panel closes the first one, and stops whatever it was running.

### What's on the panel

- **Status banner (top).** Large coloured text saying what's happening right now. It follows the running script's own output, so it stays correct when you press the hotkeys in Roblox as well as the panel's buttons.

  | Banner | Meaning |
  |---|---|
  | grey — *Nothing running* | Nothing is running |
  | yellow — *loaded but NOT playing yet* | The AI or recorder is waiting for you to press **Insert** in Roblox |
  | green — *AI is PLAYING* | The AI is pressing keys for you |
  | yellow — *AI is PAUSED* | You pressed Insert again; press it once more to resume |
  | red — *RECORDING* | Your gameplay is being recorded |
  | blue | Camera test or model update in progress |

- **Play**
  - **Camera** — how the AI turns the camera: `mouse` (right-drag, fastest, recommended), `keys` (arrow keys) or `off`.
  - **Invert camera** — tick this if the camera test turns the wrong way.
  - **Save a log of this run** — keeps a log and the frames the AI saw, so the run can be scored and problems diagnosed. Leave it on.
  - **Start AI (then press Insert in Roblox)** — loads the AI. It stays **off** until you press Insert in Roblox.
  - **Test camera** — turns the camera left for a second, then right, to check the camera setting.
- **Record your own gameplay** — **Start recorder (then press Insert in Roblox)** loads the recorder. Insert starts and stops each recording.
- **Update the AI from recordings** — **Update model** teaches the AI from everything in `recordings/` (see below). The options:
  - **Re-track all recordings** — tracks the ball again in every recording, not just new ones. Slow; only needed after the ball detection code changes.
  - **Also retrain the ball detector** — retrains the learned ball detector too. Slow; worth doing after recording new sessions, especially on new maps.
- **How did it go?** — **Score latest run**, **Score all runs** (see [Scoring runs](#scoring-runs-score_logspy)) and **Open logs folder**. These work at any time, even while the AI is playing.
- **Stop** — ends whatever is running, safely: it presses End (the scripts' own quit key, which lets go of every held key and button), and only force-closes the script if it doesn't respond.
- **Output box** — everything the running script prints.

Only one script runs at a time. While one is running, the other buttons are greyed out, and a message says to press **Stop** first. Scoring works at any time.

**Hotkeys in Roblox** (the AI and recorder listen for these even though the panel isn't focused): **Insert** turns the AI or recording on and off, and **End** quits the script.

### First time: check the camera

1. In Roblox, join a game.
2. In the panel, set **Camera** to `mouse` and press **Test camera**, then click into Roblox within 3 seconds.
3. The view should turn left for a second, then right. If it turns the wrong way, tick **Invert camera**. If it doesn't move at all, try `keys`.

### Let the AI play

1. Press **Start AI (then press Insert in Roblox)**. The banner turns yellow: the AI is loaded but **not playing yet**.
2. Click into Roblox and press **Insert**. The banner turns green, and the AI is playing.
3. Press **Insert** again to pause it, whenever you want to play yourself. Press **End** in Roblox, or **Stop** in the panel, to quit.
4. Afterwards, press **Score latest run** to see how often it got targeted, blocked and survived.

While it's on, the AI sends real key and mouse input to whatever window is focused, so keep Roblox focused.

### Teach the AI with your own gameplay

The AI learns by copying what you do, so the more of your own play it sees, the better it gets.

1. **Record.** Press **Start recorder (then press Insert in Roblox)**. The banner turns yellow. Click into Roblox and press **Insert** to start recording, and the banner turns red. Play normally, then press **Insert** again to stop and save that recording. You can record several in a row. Press **Stop** when you're done.
2. **Update the model.** Press **Update model**. Tick **Also retrain the ball detector** if you recorded on maps it hasn't seen before. The panel then:
   1. finds the ball in every new recording (the slow part — it can take several minutes);
   2. retrains the ball detector, if ticked;
   3. rebuilds the training data from **all** your recordings;
   4. retrains the AI from scratch and saves it as `model.joblib`.

   Wait until the banner goes grey and the status says *Done*.
3. **Play.** Press **Start AI** as usual. It uses the new model straight away.

What makes recordings useful:

- **Block the way you normally do.** Use **F** or left click; both count the same.
- **Use your ability with Q.** The AI can only learn to use abilities if you do in your recordings; none of the current ones have any.
- **Mix maps and situations.** The AI only learns what it has seen, so record on several maps, in close fights, and when the ball comes at you from behind.
- **Record whole rounds.** Keep recording through the calm parts too, because the AI also needs to learn when *not* to block. Stop recording between matches, in menus and lobbies.
- **Keep the same camera zoom and screen resolution** that the AI plays at. Detection and the block distances assume them.

Each recording is saved to `recordings/session_<date>_<time>/`. To remove a bad recording (for example, you were in a menu the whole time), delete its folder, then press **Update model**.

The rest of this README describes the scripts that the panel runs, for anyone who wants to run them by hand or change them.

## Full workflow (by hand)

What the panel does, step by step, as commands.

### 1. Record gameplay

```
python record_gameplay.py
```

- Press **Insert** to start/stop a recording session, **End** to quit the program.
- Each session is saved to `recordings/session_<timestamp>/`:
  - `frames/000001.jpg, 000002.jpg, ...` — one screenshot per captured frame
  - `inputs.jsonl` — one JSON line per frame: `{"frame": "...", "t": <unix timestamp>, "held": [...]}`
- Play normally. Record several sessions across different maps and rounds — the model can only learn patterns that actually appear in your recordings, including rare ones like blocking and using your ability.
- Records W/A/S/D, **F** (block) and **Q** (ability) — Blade Ball's default keybinds — plus left and right mouse. Left click also blocks. Right mouse is recorded but not used as a training label: in Roblox, holding right mouse and dragging rotates the camera, so it's camera movement, not an action. (Sessions recorded before F/Q tracking was added have no F blocks or Q abilities in them.)
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

Writes `ball_tracks.jsonl` into that session folder — one line per frame with the detected `ball_x`, `ball_y`, `state` (`"idle"` or `"targeting"`) and `ball_r` (apparent radius in pixels), or `null`s if nothing was found that frame, plus `self_red` (how strongly your character is tinted red, i.e. targeted) for every frame.

Optional visual spot-check:

```
python track_ball.py recordings/session_XXXX --debug --debug-every 20
```

Saves every 20th frame, annotated with a circle around the detection, into `session_XXXX/debug_frames/`.

### 4. Build the training dataset

```
python build_dataset.py recordings/session_A recordings/session_B ... -o dataset.csv
```

(Or `recordings/*/` to include everything.) Merges inputs + ball tracks from each session, drops frames where the ball wasn't detected, and writes one row per frame: the raw detection (`ball_x`, `ball_y`, `ball_state`), every feature from [Features](#features-featurespy), and one label column per action:

| Label | 1 when… |
|---|---|
| `held_w`, `held_a`, `held_s`, `held_d` | that movement key is held |
| `held_block` | left mouse **or** F is held (both block) |
| `held_ability` | Q is held |

Labels describe what an action *does* rather than which button did it, so blocking with F and blocking with a click teach the model the same thing. Sessions tracked before radius tracking existed are skipped with a message to re-run `track_ball.py` on them.

### 5. Train the model

```
python train_model.py dataset.csv -o model.joblib
```

- Trains an `MLPClassifier` (scikit-learn) with hidden layers `[32, 16]` by default (`--hidden-sizes` to change), on the 11 features from `features.py` (see [Model](#model-train_modelpy)).
- Skips any label with no positive examples yet (e.g. `held_ability` until you've recorded sessions where you pressed Q) — a label that's always 0 can't be learned.
- Splits each session's data by time (last 20% held out as test set, not a random split — avoids testing on near-duplicate frames the model basically already saw).
- Oversamples rare rows in the *training* set only: frames with block/ability held (5x), and frames in the fastest 15% of ball speed (3x) — both are underrepresented relative to how much they matter.
- Prints a per-action classification report (precision/recall/F1) on the held-out test set.

Re-run this any time you've recorded more sessions and rebuilt a bigger `dataset.csv` — it always trains fresh from scratch.

### 6. Check the camera setup (once)

```
python play_live.py --camera-test
```

Switch to Roblox within 3 seconds. The camera should turn to look left for a second, then right. If it turns the wrong way, add `--camera-invert` to your play command. If it doesn't move at all, run `python play_live.py --camera-test --camera mouse` to try right-mouse dragging instead of the arrow keys. See [Camera control](#camera-control).

### 7. Play live

```
python play_live.py model.joblib
```

- **Insert** toggles AI control on/off (starts off). **End** quits immediately and always releases every key/button first, so nothing gets stuck held down.
- **Safety:** while ON, this sends real keyboard/mouse input to whatever window is focused. Keep Roblox focused and your hand near the keyboard the first few times, in case it does something unwanted — End stops everything instantly.
- `--camera keys|mouse|off` and `--camera-invert` pick how the camera is turned (whatever worked in step 6).
- **Block timing is decided by distance to your character.** The model gets block timing wrong in both directions: it once tapped with the ball ~1.1 s away (block used up, died), and right at contact it often *doesn't* want to block, since in the recordings you'd already pressed earlier. So a block only goes through once the ball is close to your character: within `BLOCK_MAX_CHAR_DIST` (200 px) on screen *and* at least `BLOCK_MIN_RADIUS` (14 px) in apparent size — every successful live block was 121–191 px away and every contact ≥ 15 px in radius, while a ball far away in front of you sits just above your character on screen (in practice mode it spammed block for 3 s at radius 8–10). A ball coming head-on stays up the screen until the last instant, so a big ball (≥ `BLOCK_BIG_RADIUS`, 45) counts from `BLOCK_BIG_MAX_CHAR_DIST` (320 px). "Close" is judged 0.1 s ahead (`BLOCK_LEAD_S`, at most 150 px closer): a fast ball covers 100–180 px per frame at the end, so it can go from "too far" to hitting you between two frames — the likely deaths in the 2026-09-23 runs were taps one frame too late like that. Size is judged ahead the same way, from how fast the ball is growing (`BLOCK_MAX_GROWTH_LEAD`): a ball coming head-on barely moves on screen and just grows — radius 12 → 14 → 17 → 24 in its last three frames in one death, where the size gate held the block back a frame. While you're targeted, the closing speed is measured between two red detections even if tracking broke in between (it does, when the ball rushes in during a camera turn). Once close, it blocks if the model wants to **or** if you're targeted and the ball is red. Raise the distances if it blocks too late, lower them if too early.
- **Red wins while you're targeted.** While your character is highlighted, a red ball on screen takes over tracking from anything else (a stuck white decoy used to keep "continuing" itself while the red ball coming at you was ignored), and blobs centered on your own character are never taken as the ball — while targeted, your red-tinted body otherwise passes for one.
- **Targeted memory.** Your red "targeted" tint can blink off while the ball is still coming (e.g. during a block animation); for 1.5 s after last seeing it (`TARGET_LATCH_S`) you keep counting as targeted as long as the tracked ball is still red.
- Movement keys are held for as long as the model wants them. Block is **tapped** instead, and re-tapped at most every 0.35 s (`TAP_ACTIONS`) while the model keeps wanting it — Blade Ball's block only triggers at the moment of the press, so holding it from an early press would mean it never fires again when the ball actually arrives.
- Per-action confidence thresholds are in `THRESHOLDS` (block is 0.4: on held-out data that fires on about the same share of targeting frames as you actually blocked).
- Refuses to start with a `model.joblib` trained on an older label set — rebuild the dataset and retrain after updating.
- Optional diagnostics: `--log` writes one JSON line per live-inferred frame (every feature, the model's per-action confidence, actions taken, camera turn direction) **and** saves every captured frame as a JPEG into a matching `_frames/` folder. Plain `--log` makes a new timestamped file in `live_logs/` each run (e.g. `live_logs/live_20260923_213000.jsonl`), so earlier logs are never overwritten; `--log name.jsonl` uses the name you give. This is the main tool for root-causing "why did it do that" after a session — see [Diagnosing bad behavior](#diagnosing-bad-behavior).

## Ball detection (`track_ball.py`)

The ball is found by plain color thresholding, not a trained model: it appears as a white/grey disc when idle and switches to red when it's targeting you. `find_ball_candidates()` returns *every* blob in the frame that passes color + area + circularity filtering — often more than one, since other round pale/red things (another player's head, a skill effect, map decorations) can pass the same filter.

Because a single frame's color/shape signal alone can't reliably tell the real ball apart from those decoys, there's a layer of temporal reasoning on top, controlled by `DEFAULT_CONFIG` in `track_ball.py` (or `color_config.json` if you've saved tuned values):

| Config key | What it does |
|---|---|
| `white_sat_max`, `white_val_min` | HSV range for the idle (white/grey) ball |
| `white_split_val_min` | A pale, hazy sky (e.g. the barn map: brightness 157–197) also passes the white filter, and a ball in front of it merges into one huge blob that fails the shape check. White blobs that fail are re-examined with this stricter brightness (215; the ball is ~238), which splits the ball out. Normal detection is unaffected |
| `red_hue_low_max`, `red_hue_high_min`, `red_sat_min`, `red_val_min` | HSV range for the targeting (red) ball — red wraps around hue 0, so it's two ranges |
| `min_hull_circularity`, `min_solidity` | A glowing ball (the red targeting ball especially) has a ragged edge that sinks the plain circularity score — a perfectly round glowing ball scored 0.59 live and was missed. So a blob also counts as round if its convex hull is round (real balls 0.95–0.98; banner letters, sky patches, explosion flashes 0.83–0.89). It must also cover ≥90% of its hull (balls 0.95–0.98), which rejects round-outlined letters with an opening like a G or C |
| `min_area`, `max_area`, `min_circularity` | Shape filter — rejects blobs that are the wrong size or not round enough. `max_area` is deliberately large: a ball right next to you can be ~100 px in radius, and an earlier, smaller limit threw the ball away exactly when it was about to hit |
| `core_min_share`, `core_max_ring`, `core_min_radius` | A close, fast ball's pale trail merges with it into one long blob that fails the round-shape check. For such blobs the largest solid circle inside is recovered as a "core" — if it's at least 12 px in radius, makes up at least half the blob, and stands out from its surroundings (a trail touches one side; a wall or floor would surround it). Sky patches and glowing effects can produce convincing cores too, so a core is only ever used to *continue* an existing track, never to pick up a new ball |
| `core_radius_ratio`, `size_continuity_frames` | The ball's apparent size changes smoothly. A core continues the track only if it's 0.5–2.5× the tracked ball's last radius, and for 10 frames after a track was last seen a far "jump" candidate has to be too — otherwise a speck elsewhere takes over the moment a close ball blinks out for a frame or two |
| `min_fill` | Fraction of a blob's outline actually filled with ball color. The ball is a solid disc (~1.0, and ≥0.9 for ~97% of tracked balls in the recordings); round lettering in red announcement banners ("STANDOFF", "NO ONE WON!") is a hollow ring (~0.75–0.85) and gets rejected |
| `ui_mask_regions` | Fractional screen regions ignored entirely (HUD panels, stat bar, block/ability icons, a couple of static decorations found to cause false positives) |
| `max_jump_px_per_frame`, `reset_after_missing_frames`, `confirm_lookahead_frames` | A detection far from the last trusted position isn't rejected outright — it's trusted if the *next* detected frame keeps going near it (real fast movement/bounces continue, a one-off flash doesn't) |
| `stale_after_frames`, `stale_jitter_px` | If the trusted position hasn't moved at all for this many frames, stop trusting proximity to it and force a fresh whole-frame search — catches long lock-ons onto something static (a decoration, a standing player) that a one-frame check can't |
| `self_target_threshold` | Highlight score above which you count as targeted. While targeted, a red ball on screen is trusted immediately instead of having to match the previous track — while the camera turns to find the ball the whole scene sweeps ~300 px per frame, so otherwise the ball never looks like a continuation of anything and the camera sweeps right past it (this is what happened in live testing) |
| `min_ball_score` | If `ball_classifier.joblib` exists, candidates the learned ball detector scores below this probability are dropped (see [Learned ball detector](#learned-ball-detector-ball_classifierpy)) |
| `self_highlight_roi`, `self_highlight_hue_min`, `self_highlight_sat_min`, `self_highlight_val_min` | Region around your own character, and color thresholds, for Blade Ball's red "you're targeted" tint. The region is kept tight so a highlighted player standing next to you mostly falls outside it, and only the slightly pinkish side of red (hue 170–180) counts — orange-red dirt, lava or a pumpkin head near your feet sits on the other side of pure red and caused false alarms. Adjust the region if your camera zoom puts your character somewhere else on screen |

`track_ball.py` (batch, processing a whole recorded session) and `play_live.py` (real-time) both use this same candidate list + proximity/staleness logic, via shared functions (`find_ball_candidates`, `closest_to`, `most_circular`) — the batch version can additionally peek one frame into the future to confirm a jump, which live inference can't do (it uses a one-frame "pending" delay instead). Each candidate also carries its apparent radius (from contour area), which feeds the depth features below.

## Learned ball detector (`ball_classifier.py`)

Color and shape rules keep getting fooled by things that are also round and white/red on some map — sparkles, lanterns, torches, banner letters, players' cosmetics — and each rule above fixes one decoy. `ball_classifier.py` learns what separates the real ball from all of them at once: for every candidate it looks at the color and brightness inside it, how uniform the inside is, how sharp its edge is, and how it contrasts with its surroundings, and outputs the probability it's the ball.

```
python ball_classifier.py
```

Trains `ball_classifier.joblib` from the existing recordings with no manual labeling: positives are the ball the tracker followed, but only on frames where it was actually moving (static decoys sit still); negatives are every other candidate in the same frame, well away from the ball. Tested on whole held-out sessions (unseen maps), at the default `min_ball_score` of 0.1 it keeps 98.5% of real balls and rejects 73% of decoys. On hand-checked live frames, real balls scored 0.40–0.995 and decoys (a stray speck, a stuck white object, an explosion's center, a sky patch) 0.02–0.19.

`track_ball.py` and `play_live.py` use it automatically once trained (live costs ~3 ms per frame), and work without it. `python play_live.py model.joblib --no-classifier` turns it off to compare. Retrain it after recording new sessions (the control panel's *Also retrain the ball detector* option).

## Features (`features.py`)

Both `build_dataset.py` and `play_live.py` compute features through the same `FeatureTracker`, so training and live play can't disagree about what a feature means.

| Feature | Meaning |
|---|---|
| `ball_rel_x`, `ball_rel_y` | Position relative to screen center |
| `ball_vel_x`, `ball_vel_y` | Screen velocity in **pixels per second** (real elapsed time, not frame count — live inference doesn't run at a perfectly steady rate) |
| `ball_distance` | Distance from screen center |
| `ball_closing_speed` | Positive = ball moving toward screen center, negative = away |
| `state_targeting` | 1 when the ball is red (targeting) |
| `ball_radius` | Apparent radius in pixels (smoothed) — bigger means closer |
| `ball_radius_rate` | How fast the radius is growing, in px/s — positive means approaching |
| `ball_ttc` | Estimated time to contact in seconds (radius ÷ growth rate, capped at 3 s) |
| `self_red` | How strongly your own character is tinted red — Blade Ball's "you're targeted" highlight |

Screen position alone can't tell how *close* the ball is — a ball at screen center can be next to you or across the arena. Its apparent size can: it grows as the ball approaches, and radius ÷ growth rate estimates how many seconds until it arrives. That's the cue that says *when* to block, which position and velocity only approximate.

`self_red` answers *whether* the ball is coming for you. The ball's own color (`state_targeting`) turned out to be an unreliable signal for that: in the recordings it was only picked up as red on about half the frames where your character was actually highlighted as the target, while 84% of your blocks happened while highlighted (block rate 9.2% highlighted vs 0.8% not).

Motion history (velocity, radius growth) restarts after gaps longer than 1 second, and live play also restarts it whenever the camera turns — a turning camera sweeps the whole scene across the screen, which would otherwise read as ball velocity.

## Model (`train_model.py`)

A small multi-label MLP — no vision component. `track_ball.py` handles "seeing" the ball; the model only ever sees the 11 numeric features above.

**Outputs (independent per-action confidence, each with its own decision threshold in `play_live.py`):** `held_w`, `held_a`, `held_s`, `held_d`, `held_block`, `held_ability` (only labels that have examples in the training data — see step 5).

Features are normalized with a `StandardScaler` fit on the training data (saved alongside the model in `model.joblib`, so live inference scales consistently).

## Camera control

The model can only react to a ball it can see, so `play_live.py` turns the camera horizontally to keep the ball on screen:

- **You're targeted and no ball is steadily tracked in view:** turn right away to find it. (A steadily tracked ball is always kept in view instead, whatever color it reads as — an earlier version searched whenever the tracked ball wasn't red, and in live testing that turned the camera away from the real ball.) When the ball targets you, Blade Ball tints your own character red — `track_ball.self_highlight_score` measures that in a fixed region around your character, so it works even when the ball is behind the camera, which is exactly when you'd otherwise die without seeing it.
- **Ball steadily tracked, heading to stay inside the middle half of the screen:** the camera doesn't move. This dead zone keeps the ball's on-screen position meaningful to the model, which learned from recordings where the ball moved freely around the screen rather than being pinned to center.
- **Ball steadily tracked, heading past the dead zone toward the left/right edge:** a short turn toward it, longer the further out it's heading. "Heading" means where it will be 0.25 s from now (`CAMERA_LEAD_S`), from its motion on screen — reacting only to where it *is* was too late: a ball flying across the view at ~1500 px/s was off the edge a frame or two after crossing the dead zone, before the turn took effect, and most deaths in the 2026-09-23 runs started exactly like that. Motion is only measured while the camera holds still (the game shows a turn about a frame late); mid-turn, the last measurement is kept for up to 0.3 s so the camera keeps following.
- **Ball lost:** turn toward the side it was last heading, to find it again — after 2 frames if it was heading off the edge of the screen, otherwise after ~half a second (a small far ball is often just missed for a few frames) — for up to 6 seconds, after which it's probably between rounds. In mouse mode searches drag 1.5x faster than normal turns (`CAMERA_SEARCH_SPEED`).
- **Far-away ball that isn't after you** (radius under `CAMERA_FAR_RADIUS`, 10 px): followed calmly — no leading, no quick search, no fast sweep. A far ball bouncing between players on opposite sides swung the camera back and forth, and every sweep overshoots a bit because the game shows a turn about a frame late. If it turns on you, your red highlight starts the fast search anyway.
- **Ball found again:** the search stops at once. A search left running after the ball came back into view swung it across the screen and broke tracking just as it was coming in.
- **A detection that just jumped to a new spot:** ignored for steering until it's been followed smoothly for 3 frames (`CAMERA_MIN_TRACK_FRAMES`). In live testing, most camera swings away from the real ball started on exactly this kind of detection — a sparkle, a lantern, another player's cosmetic.

Two ways to turn, chosen with `--camera`:

- `keys` (default) — Roblox's Left/Right arrow keys, which rotate the default camera. Works, but turns slowly: in testing, swinging around to find a ball that was targeting you from behind took almost a second — about as long as the ball took to arrive.
- `mouse` — holds right mouse and drags, sent as raw relative mouse input (Windows `SendInput`). Roblox reads camera drags from raw mouse deltas, which ordinary cursor movement doesn't reliably produce. Turn speed is set by `CAMERA_MOUSE_SPEED` (and your Roblox mouse sensitivity), so it can turn much faster than the arrow keys. The drag really moves the cursor, so it's kept near the center of the game screen: whenever it drifts more than `CAMERA_REANCHOR_PX` it hops back (with right mouse briefly released, so the hop doesn't turn the camera), and it returns to center when a turn ends — otherwise it wanders onto a second monitor, where a block click would land in another window.

Tunables (`CAMERA_*` at the top of `play_live.py`): dead-zone width, turn duration per unit of offset, search timing, and mouse drag speed. The camera only turns horizontally; there's no up/down control yet.

## Diagnosing bad behavior

If the AI does something confusing live, don't guess — capture it:

```
python play_live.py model.joblib --log
```

This gives you, for every frame the AI acted on: the detected ball position/velocity/state, the model's confidence for each action, and what it actually pressed — plus the actual frame image in the matching `live_logs/live_<date>_<time>_frames/` folder, so you can visually confirm what was really being tracked at any point. This has already been essential for catching cases where detection locked onto the wrong thing entirely (see below).

### Scoring runs (`score_logs.py`)

```
python score_logs.py          # newest log in live_logs/
python score_logs.py --all    # every log, plus a total
```

Lists every time you were targeted during a run, whether block was tapped, when and how far away the ball was, and how it ended, plus taps made while you weren't targeted. The log can't see "you died" directly (a death ends your red highlight just like a successful block), so results are:

- **survived** — tapped, and you were targeted again soon after, so you were alive
- **blocked or died** — tapped in time, but no later evidence either way
- **tapped, unclear** — tapped, but the highlight stayed on long after (possibly too early)
- **no tap** — targeted but never blocked; most deaths look like this

Use it to compare before/after a change instead of judging from one memorable match. Anything but *survived* is worth a look in the saved frames.

## Known Limitations

- **Ball detection can still lock onto decoys.** Map decorations, other players' cosmetics, glow near your own character, and similar red/white objects can pass the same color/shape filter as the real ball. The learned ball detector plus the tracking rules catch most cases, but the detector is only as good as its training labels, which come from the rule-based tracker itself (no hand-labeled data) — so it learns the tracker's habits along with the ball's look. Hand-labeling a few hundred tricky frames would make it sharper.
- **Camera control is rule-based, not learned.** The camera controller is a fixed policy (turn toward an edge ball, search when lost), not something imitated from your play — recordings don't capture how much you dragged the camera, only whether right mouse was held. It only turns horizontally.
- **Movement is imitated, not planned.** WASD comes from the model copying your recordings; it has no idea of walls, other players or what's a good position, and can look like "running away" from the ball.
- **Ability has no training data yet.** Q wasn't recorded before, so the model can't use abilities until you record new sessions where you do.
- **The model only acts when it can see the ball.** Frames with no ball detected produce no features, so while you're targeted with the ball off-screen the camera searches for it but nothing blocks blind.
- **Block timing is rule-based.** The model learned *whether* to block reasonably well, but not *when* (your own recorded presses are spread out, so the label is diffuse). Timing is set by the close-enough gate in `play_live.py` (`BLOCK_*` constants), tuned from a handful of live runs; odd approach angles or map lighting may need further tuning — `score_logs.py` shows the ball's distance and size at every tap.
- **The targeted highlight depends on map lighting.** On strongly orange-lit maps your red "targeted" tint shifts toward orange and only just clears the threshold. The opposite problem also happened: on the orange desert arena a dark outfit (lit orange, plus a pink sword glow) looked dimly reddish and kept reading as "targeted" — only *bright* pinkish-red counts now (`self_highlight_val_min` 110), which removed most of those false alarms.
- **Single-monitor, fixed-resolution assumption.** Ball detection and screen-center calculations assume the capture region matches between recording and live play.
