"""
Tells whether you're playing a round or not (in the lobby, dead, or
spectating), from Blade Ball's own menu.

The left-hand menu shows a green TRADE button whenever you're *not* in a
round -- in the lobby, in the lobby's practice area, and a second or so
after you die -- and swaps it for "(R) Emote" while you're alive in a
round. So this looks for the TRADE button: only its bright green pixels
are matched against a picture of it (assets/trade_button.png), which
ignores whatever map shows through behind the menu. On every saved live
frame so far, frames with the button scored >= 0.93 and frames without it
<= 0.52; it costs ~1.5 ms per check.
"""

from pathlib import Path

import cv2

TEMPLATE_PATH = Path(__file__).resolve().parent / "assets" / "trade_button.png"

# Where the menu button can be, as fractions of the screen (x0, y0, x1, y1).
# Generous, because the menu sits a little differently in windowed and
# fullscreen Roblox.
SEARCH_REGION = (0.0, 0.44, 0.21, 0.76)
# Work at half size -- plenty for a button this big, and 4x less work.
WORK_SCALE = 0.5
# Button sizes to try, relative to its size in a 1080p frame.
SIZES = (0.9, 0.95, 1.0, 1.05)
# Score above which the button counts as there.
THRESHOLD = 0.75

# Bright green of the button's text and handshake icon (OpenCV HSV).
GREEN_LOW = (40, 120, 120)
GREEN_HIGH = (85, 255, 255)


def _green(img_bgr):
    return cv2.inRange(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), GREEN_LOW, GREEN_HIGH)


class LobbyDetector:
    def __init__(self, template_path=TEMPLATE_PATH):
        template = cv2.imread(str(template_path))
        if template is None:
            raise FileNotFoundError(f"missing {template_path}")
        self._mask = _green(template)
        self._templates = {}  # frame height -> scaled templates

    def _templates_for(self, frame_h):
        if frame_h not in self._templates:
            base = frame_h / 1080 * WORK_SCALE
            self._templates[frame_h] = [
                cv2.resize(self._mask, None, fx=base * s, fy=base * s, interpolation=cv2.INTER_AREA)
                for s in SIZES]
        return self._templates[frame_h]

    def score(self, frame_bgr):
        """How much the menu looks like it has the TRADE button (0-1)."""
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = SEARCH_REGION
        region = _green(frame_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)])
        if not region.any():
            return 0.0
        region = cv2.resize(region, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
        best = 0.0
        for t in self._templates_for(h):
            if t.shape[0] <= region.shape[0] and t.shape[1] <= region.shape[1]:
                best = max(best, float(cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED).max()))
        return best

    def in_lobby(self, frame_bgr):
        """True when you're not playing a round (lobby, dead, spectating)."""
        return self.score(frame_bgr) >= THRESHOLD


# --- Gamemode vote ---------------------------------------------------------
# Between rounds Blade Ball lets players vote for the next gamemode. Auto can
# click one for you: it looks for a picture of that mode's vote button,
# assets/vote_<mode>.png, anywhere on screen.
VOTE_MODES = ("classic",)
VOTE_THRESHOLD = 0.8


def vote_template_path(mode):
    return Path(__file__).resolve().parent / "assets" / f"vote_{mode}.png"


class VoteButton:
    """Finds one gamemode's vote button on screen."""

    def __init__(self, mode):
        self.mode = mode
        template = cv2.imread(str(vote_template_path(mode)))
        if template is None:
            raise FileNotFoundError(f"missing {vote_template_path(mode)}")
        self._template = template
        self._templates = {}  # frame height -> scaled templates

    def _templates_for(self, frame_h):
        if frame_h not in self._templates:
            base = frame_h / 1080 * WORK_SCALE
            self._templates[frame_h] = [
                cv2.resize(self._template, None, fx=base * s, fy=base * s,
                           interpolation=cv2.INTER_AREA) for s in SIZES]
        return self._templates[frame_h]

    def find(self, frame_bgr):
        """(x, y) of the button's center in frame pixels, or None."""
        small = cv2.resize(frame_bgr, None, fx=WORK_SCALE, fy=WORK_SCALE,
                           interpolation=cv2.INTER_AREA)
        best, best_xy = 0.0, None
        for t in self._templates_for(frame_bgr.shape[0]):
            result = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(result)
            if score > best:
                best = score
                best_xy = ((x + t.shape[1] / 2) / WORK_SCALE, (y + t.shape[0] / 2) / WORK_SCALE)
        return best_xy if best >= VOTE_THRESHOLD else None
