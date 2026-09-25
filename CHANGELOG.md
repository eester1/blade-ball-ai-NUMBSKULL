# Changelog

## Unreleased

### New

- **Faster reactions:** the AI looks at the screen **45 times a second** (was 30). The ball search runs its white and red halves side by side (about 21 ms a frame instead of 27, identical results). A block tap no longer pauses it for 30 ms, and log screenshots are saved in the background. Logged frames record their processing time (`frame_ms`).
- **Camera doesn't spin away from a ball it just had:** when you're targeted and a tracked ball vanishes mid-screen, it waits 0.25 s for it to reappear before searching. In the logs, 11 of 15 targetings where the ball wasn't on screen when it arrived had it in view just before.
- **Spam block in fast exchanges now triggers only when you're targeted again within 0.9 s** (was 1.2 s): in the logs, those exchanges died 35% of the time against 19% for 0.9–1.2 s. Tested live over 20 rounds it did much worse (20% survived in those exchanges, against 67% without), so leave it off.
- **Auto** — tick it (or use `--auto`) and the AI plays round after round by itself. It plays while you're in a round, lets go of everything and waits in the lobby after you die or the round ends, and starts again when the next round begins. It recognises the lobby from Blade Ball's green TRADE menu button (`game_state.py`). It's off unless you choose it: the tick box starts unticked every time the panel opens. See [Auto](README.md#auto).
- **Auto vote** — with Auto on, set it to `Classic` (`--vote classic`) and the AI votes for the Classic gamemode each time it's in the lobby. The default is `None`. It finds the Classic button on the "Vote for the next gamemode" panel and clicks it once per lobby visit; see [Auto vote](README.md#auto-vote).
- **Lobby screenshots in logs** — while Auto waits in the lobby with logging on, it saves one screenshot per second.
- **Choose your stop key** — pick the key that stops the AI from inside Roblox in the panel (End, Home, Delete, Page Up/Down, Pause, Scroll Lock, F6–F12), or with `--quit-key`. The panel's Stop button presses the same key.
- **Only acts in Roblox** — the AI sends input only while the Roblox window is the active window. Click into another window and it lets go of everything and waits.
- **The panel remembers your choices** (camera, invert, log, stop key) between runs, in `panel_settings.json`. Auto and Auto vote always start off.
- **Scores confirm deaths** — logs record when the AI went to the lobby, so being sent there right after you were targeted is reported as **died/round over** instead of a guess.
- New banners for the new states: waiting in the lobby, and waiting for the Roblox window.

- **30 looks per second instead of 15** — screen capture now uses `dxcam` when it's installed (about 1 ms instead of about 33 ms), so the AI runs at 30 fps: measured 29.8 fps with all detection. Ball tracking, block timing and the camera use every frame. The model still sees the ball 15 times a second, like the recordings it learned from. Settings that count frames were doubled to mean the same time, and logs still save 15 screenshots a second, so they don't get bigger. Falls back to `mss` at the old speed without dxcam.

- **Blocks balls that are close in 3D, not just on screen** — while you're targeted, the ball's real distance from your character (worked out from its screen position and size) also counts as close. A ball coming from in front, from above or near the camera could hit you while still far from your character on screen. Across 278 logged targetings, the block rule now fires in 87% instead of 68%, and too late in 23% instead of 46%.

- **Blocks a smaller-looking red ball** — while you're targeted, a red ball counts as big enough to block from radius 11 (was 14). In one run, 42% of contacts were under 14. Over the logged targetings the rule now fires too late in 15% instead of 24%.

- **Auto vote clicked the wrong button** — the click moved the cursor in a way Roblox doesn't see, so votes landed on whichever button the cursor was already over (usually the middle one), or nowhere. It now sends real Windows mouse input, and only counts the vote once the game shows its tick on Classic, retrying up to 3 times.

- **Lava counted as "you're targeted"** — on a lava map, the bright red lava next to your character passed the targeted check, faking targetings the AI then searched for and never blocked. Only the darker red of the real tint counts now (brightness at most 180): lava false alarms drop from about 0.12 to 0.02, while real targetings barely change. On a replay of that run, the "no tap" targetings go from 6 to 3.

### Docs

- New README section, [Camera zoom](README.md#camera-zoom-keep-it-the-same-every-time): zoom changes how big everything looks to the AI, so it has to stay the same every time (12 notches out from first person here). Zooming out further breaks the "you're targeted" check.

- New README section, [Windowed or fullscreen?](README.md#windowed-or-fullscreen-play-the-way-you-record): the AI plays best in the window mode its recordings were made in, why, and how to switch modes.
- A warning that botting is against Roblox's Terms of Use, and a new README section, [Long runs and log size](README.md#long-runs-and-log-size): logged runs use about 29 GB per hour of play, so untick **Save a log of this run** for long runs.

- **Learned block timing** — `learn_block.py` (the panel's **Learn from my runs** button) learns when to block from the AI's own logged games: which taps survived and which died, by the ball's 3D distance at the tap. The first time, taps inside the built-in distance survived 66–74%, and taps a bit farther out 87–93%, so it learned to tap earlier (39 instead of 22). It's switchable with the **Learned block timing** tick box (`--learned-block`), so it can be compared with the built-in rules and turned off any time. It only learns from runs played with **Save a log of this run** and **Auto** on.

- **Hold block while the ball hovers is now on by default** — over 16 live rounds, survival per targeting rose from 77% to 84% (slow balls 68% to 87%, normal 82% to 87%, fast exchanges unchanged at 71%).
- **A red ball closing in counts as targeting you** — for blocking, even when your red tint doesn't show (it didn't at all on a dark map, and a death followed).
- **Spam block keeps going between hits** while the ball stays close, since in a fast exchange the tint shows only about 0.1 s before the ball is back.
- **Experimental block options** (off by default, tick boxes in the panel, recorded in each run's log): **Spam block in fast exchanges** (`--spam-block`) taps block every 0.1 s while you're targeted, when targeted again within 1.2 s of the last time. **Hold block while the ball hovers** (`--hold-hover`) doesn't tap while a red ball coming at you is barely moving in, since slow-ball deaths mostly came from blocking over a second too early. See [Experimental block options](README.md#experimental-block-options).
- **Each run's log records its settings**, and scoring shows them, so runs with and without an option can be compared.
- **Free space button** — lists logged runs oldest first with their screenshot sizes, and deletes the screenshots of the ones you pick (they're 99% of a log's size), keeping the `.jsonl` logs that scoring and Learn from my runs use.
- **Round counter in the panel** — under the banner: rounds Auto has started playing this run, and all time.

### Fixed

- **Auto vote couldn't see its own vote** — with the mouse over it (after clicking, or left there from the round before), the Classic button turns bright yellow and a little bigger, which the AI's picture of it didn't match. So the vote went through but was never confirmed, and a button left lit up from the previous round was never clicked. There's now a second picture for that look (`assets/vote_classic_selected.png`).

- **Camera circling in busy fights, and not finding the ball at round start** — searching for a ball that isn't after you is now one slow sweep, once it has been missing for 1 second: about 150° a second, one full turn in about 2.5 seconds, at most once every 7 seconds. It pauses as soon as anything ball-like comes into view. At normal speed the camera goes all the way round in under a second, too fast to recognise the ball on the way past. Searching while you're targeted is now normal speed instead of 1.5×. Searching every time chained into circling. Switching searching off entirely, tried briefly, left the ball unfound at the start of rounds (seen in 12% of frames instead of about 60%), so the AI barely moved.
- **Camera spinning in circles** — after a block the ball often flies off far away, and the camera kept searching for up to 6 seconds, restarted by every brief detection. A search for a ball that isn't after you is now limited to about a second in total, and resets only once the ball is properly tracked again. Searching while you're targeted is unchanged.
- **False "you're targeted" alarms on orange maps** — on the orange desert arena, a dark outfit lit orange (plus a pink sword glow) looked dimly red and kept setting off the targeted check, which sent the camera searching. Only bright red counts now; real highlights read the same as before.
- **Tracking pale sky instead of the ball** — gaps of pale sky between rocks or trees could pass as a white ball (a "core", the check that recovers a ball merged with its trail), and the tracker stayed locked on them while the real ball came in. White cores now have to be almost pure white, like the real ball.
- **Sand maps** — pale sand (and some cloudy skies) passed the white-ball colour check over up to a fifth of the screen, burying the white ball and making decoys. When that much of the screen passes, only nearly colourless pixels count now. Sand is warm-tinted and the ball isn't, so the real ball is still found as often as before.
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
