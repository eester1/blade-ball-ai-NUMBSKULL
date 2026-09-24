# Changelog

## Unreleased

### New

- **Auto** — tick it (or use `--auto`) and the AI plays round after round by itself. It plays while you're in a round, lets go of everything and waits in the lobby after you die or the round ends, and starts again when the next round begins. It recognises the lobby from Blade Ball's green TRADE menu button (`game_state.py`). It's off unless you choose it: the tick box starts unticked every time the panel opens. See [Auto](README.md#auto).
- **Auto vote** — with Auto on, set it to `Classic` (`--vote classic`) and the AI votes for the Classic gamemode each time it's in the lobby. The default is `None`. It finds the Classic button on the "Vote for the next gamemode" panel and clicks it once per lobby visit; see [Auto vote](README.md#auto-vote).
- **Lobby screenshots in logs** — while Auto waits in the lobby with logging on, it saves one screenshot per second.
- **Choose your stop key** — pick the key that stops the AI from inside Roblox in the panel (End, Home, Delete, Page Up/Down, Pause, Scroll Lock, F6–F12), or with `--quit-key`. The panel's Stop button presses the same key.
- **Only acts in Roblox** — the AI sends input only while the Roblox window is the active window. Click into another window and it lets go of everything and waits.
- **The panel remembers your choices** (camera, invert, log, stop key) between runs, in `panel_settings.json`. Auto and Auto vote always start off.
- **Scores confirm deaths** — logs record when the AI went to the lobby, so being sent there right after you were targeted is reported as **died/round over** instead of a guess.
- New banners for the new states: waiting in the lobby, and waiting for the Roblox window.

### Docs

- New README section, [Windowed or fullscreen?](README.md#windowed-or-fullscreen-play-the-way-you-record): the AI plays best in the window mode its recordings were made in, why, and how to switch modes.
- A warning that botting is against Roblox's Terms of Use, and a new README section, [Long runs and log size](README.md#long-runs-and-log-size): logged runs use about 29 GB per hour of play, so untick **Save a log of this run** for long runs.

### Fixed

- **Camera spinning in circles** — after a block the ball often flies off far away, and the camera kept searching for up to 6 seconds, restarted by every brief detection. A search for a ball that isn't after you is now limited to about a second in total, and resets only once the ball is properly tracked again. Searching while you're targeted is unchanged.
- **False "you're targeted" alarms on orange maps** — on the orange desert arena, a dark outfit lit orange (plus a pink sword glow) looked dimly red and kept setting off the targeted check, which sent the camera searching. Only bright red counts now; real highlights read the same as before.
- **Tracking pale sky instead of the ball** — gaps of pale sky between rocks or trees could pass as a white ball (a "core", the check that recovers a ball merged with its trail), and the tracker stayed locked on them while the real ball came in. White cores now have to be almost pure white, like the real ball.
- **Stuck on red scenery while the ball came in** — a red coral decoration, picked up as a "core", kept the track for 1.5 s while you were targeted, even once the real ball was clearly detected. While you're targeted, a red core that isn't moving now gives way to a properly round red ball.
- **Scores marked several targetings as deaths** — only the last targeting before going to the lobby can be the death.
- **Unblocked ball from behind the camera** — a ball coming from behind shows up huge at the bottom of the screen, far from your character, so it was never blocked. A very large red ball while you're targeted is now blocked wherever it is on screen.

## v0.1.0 — First release

- Control panel (`app.py`) with a status banner showing whether the AI or recorder is loaded, playing, paused or recording.
- Recorder, ball tracking (colour and shape, plus a learned ball detector that filters out look-alikes), and a small neural network trained on your own recordings.
- Live play:
  - turns the camera to keep the ball in view, leading fast balls, and searches as soon as you're targeted;
  - blocks when the ball is about to reach you, judged from its distance, size and approach speed;
  - calmly follows far-away balls that aren't after you.
- Run scoring (`score_logs.py`) with automatically named logs.
