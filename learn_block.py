"""
Learn block timing from the AI's own logged games.

Every logged run (play_live.py --log) records, for each time you were
targeted, what the ball looked like when the AI tapped block and whether
you survived or died. This looks at the decisive tap of every targeting
with a known outcome -- the last tap before the ball arrived -- and how
survival depends on the ball's 3D distance from you at that moment (see
play_live.ball_distance_3d). Taps made while the ball was still farther
away survive more often. The block fires as soon as the ball comes within
the learned distance, so the tap lands just inside it; the learned setting
is the *closest* distance for which taps just inside it (within BAND) still
survive at least TARGET_SURVIVAL of the time -- blocks come as early as
they need to and no earlier.

It writes learned_block.json, which play_live.py uses with
--learned-block (the "Learned block timing" tick box in the control
panel). Without that switch nothing changes.

USAGE:
    python learn_block.py            # learn from every log in live_logs/
    python learn_block.py --dry-run  # just print what it would learn

Outcomes come from Auto's lobby markers: back in the lobby within 3 s of a
targeting = died (or the round ended); targeted again within 15 s = alive.
Targetings with neither are left out.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

import play_live

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "learned_block.json"

TARGET_SURVIVAL = 0.90  # taps in the chosen band must survive at least this often
BAND = 15               # ...where the band is [distance - BAND, distance) ball radii
MIN_TAPS = 15           # ...and holds at least this many taps
SEARCH = range(20, 61)  # distances considered, in ball radii


def decisive_taps(log_paths):
    """(3D distance at the decisive tap, died) for every targeting with a
    known outcome and a tap."""
    rows = []
    for path in log_paths:
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        lobby = [e["t"] for e in lines if e.get("event") == "lobby"]
        frames = [e for e in lines if "event" not in e]
        if not lobby or not frames or "approach" not in frames[0]:
            continue  # no outcome markers (not an Auto run) or an older log
        episodes = []
        for e in frames:
            if e["self_red"] >= 0.08:
                if episodes and e["t"] - episodes[-1][-1]["t"] <= 0.5:
                    episodes[-1].append(e)
                else:
                    episodes.append([e])
        episodes = [ep for ep in episodes if len(ep) >= 3]
        for i, ep in enumerate(episodes):
            start, end = ep[0]["t"], ep[-1]["t"]
            nxt = episodes[i + 1][0]["t"] if i + 1 < len(episodes) else float("inf")
            died = any(0 <= t - end <= 3 and t < nxt for t in lobby)
            alive = not died and nxt - end <= 15
            if not died and not alive:
                continue
            taps = [e for e in frames if start - 0.3 <= e["t"] <= end + 0.3
                    and e["tapped"] and e["ball_x"] is not None]
            if taps:
                e = taps[-1]
                d3 = play_live.ball_distance_3d(e["ball_x"], e["ball_y"], e["ball_radius"],
                                                (960, 567), 1080)
                rows.append((d3, died))
    return rows


def learn(rows):
    """The closest 3D distance whose band of taps just inside it survives
    >= TARGET_SURVIVAL."""
    d3 = np.array([r[0] for r in rows])
    died = np.array([r[1] for r in rows], bool)
    for dist in SEARCH:
        band = (d3 >= dist - BAND) & (d3 < dist)
        if band.sum() >= MIN_TAPS and 1 - died[band].mean() >= TARGET_SURVIVAL:
            return dist, int(band.sum()), float(1 - died[band].mean())
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print, don't save")
    args = parser.parse_args()

    rows = decisive_taps(sorted(glob.glob(str(HERE / "live_logs" / "*.jsonl"))))
    if len(rows) < 50:
        raise SystemExit(f"Only {len(rows)} targetings with a known outcome -- play more logged "
                         f"Auto runs first (at least 50).")
    d3 = np.array([r[0] for r in rows])
    died = np.array([r[1] for r in rows], bool)
    print(f"{len(rows)} targetings with a tap and a known outcome ({died.sum()} died).")
    print("Survival by the ball's 3D distance at the decisive tap:")
    for lo, hi in ((0, 12), (12, 17), (17, 22), (22, 27), (27, 33), (33, 40), (40, 60), (60, 999)):
        m = (d3 >= lo) & (d3 < hi)
        if m.sum():
            print(f"  {lo:3d}-{hi if hi < 999 else '':<3}: {m.sum():3d} taps, survived {100 * (1 - died[m].mean()):3.0f}%")

    result = learn(rows)
    if result is None:
        raise SystemExit("No distance reached the target survival rate -- nothing learned.")
    dist, n, rate = result
    print(f"\nLearned: tap block while the ball is within {dist} ball radii (3D) -- taps at "
          f"{dist - BAND}-{dist} survived {100 * rate:.0f}% ({n} taps). "
          f"The built-in rule uses {play_live.BLOCK_3D_DIST}.")
    if args.dry_run:
        return
    OUT_PATH.write_text(json.dumps({
        "block_3d_dist": dist, "band_survival": round(rate, 3), "band_taps": n,
        "targetings": len(rows), "learned_at": time.strftime("%Y-%m-%d %H:%M"),
    }, indent=2))
    print(f"Saved to {OUT_PATH.name}. Tick 'Learned block timing' in the panel (or use "
          f"play_live.py --learned-block) to play with it.")


if __name__ == "__main__":
    main()
