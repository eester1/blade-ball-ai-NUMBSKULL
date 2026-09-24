"""
Score live-play logs (from play_live.py --log) so changes can be judged by
numbers instead of by one memorable match.

For each "targeted" episode -- a stretch where your character was
highlighted as the ball's target -- it reports whether the AI tapped block,
how far the ball was when it did, and how the episode ended. It also counts
taps made while you weren't targeted (wasted blocks, which can put block on
cooldown right before you need it).

USAGE:
    python score_logs.py                  # newest log in live_logs/
    python score_logs.py a.jsonl b.jsonl  # specific logs
    python score_logs.py --all            # every log found, plus a total

How an episode is classified. The log can't see "you died" directly -- a
death ends your highlight just like a successful block does -- but being
targeted again soon afterwards proves you were still alive:
  - "survived": block was tapped, and you were targeted again within
    SURVIVAL_WINDOW_S (a death keeps you out until the next round).
  - "blocked or died": tapped and the highlight ended right after, but no
    later evidence either way (e.g. the log ended, or a long quiet spell).
  - "tapped, unclear": tapped, but the highlight stayed on well after --
    the tap may have been early.
  - "no tap": targeted, but block was never tapped. Most deaths look like
    this.
Anything but "survived" is worth a look frame by frame.
"""

import argparse
import json
import sys
from pathlib import Path

import track_ball

EPISODE_GAP_S = 0.5      # targeted frames closer than this belong to one episode
EPISODE_MIN_FRAMES = 3   # shorter highlight blips are flicker, not a targeting
TAP_MARGIN_S = 0.3       # taps this close to an episode's edges count toward it
BLOCK_RESULT_S = 0.8     # highlight ending this soon after a tap = the tap landed in time
SURVIVAL_WINDOW_S = 15   # targeted again within this = you were still alive


def character_point(cfg, width=1920, height=1080):
    roi = cfg["self_highlight_roi"]
    return (roi["x0"] + roi["x1"]) / 2 * width, (roi["y0"] + roi["y1"]) / 2 * height


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score(path, cfg):
    entries = load(path)
    if not entries or "self_red" not in entries[0] or "tapped" not in entries[0]:
        return None  # made by an older play_live.py that didn't log these
    threshold = cfg["self_target_threshold"]
    cx, cy = character_point(cfg)
    t0 = entries[0]["t"]

    episodes = []
    for e in entries:
        if e["self_red"] >= threshold:
            if episodes and e["t"] - episodes[-1][-1]["t"] <= EPISODE_GAP_S:
                episodes[-1].append(e)
            else:
                episodes.append([e])
    episodes = [ep for ep in episodes if len(ep) >= EPISODE_MIN_FRAMES]

    taps = [e for e in entries if e["tapped"]]
    counted_taps = set()
    rows = []
    for i, ep in enumerate(episodes):
        start, end = ep[0]["t"], ep[-1]["t"]
        ep_taps = [e for e in taps if start - TAP_MARGIN_S <= e["t"] <= end + TAP_MARGIN_S]
        counted_taps.update(id(e) for e in ep_taps)
        targeted_again = i + 1 < len(episodes) and \
            episodes[i + 1][0]["t"] - end <= SURVIVAL_WINDOW_S
        if not ep_taps:
            result = "no tap"
        elif end - ep_taps[-1]["t"] > BLOCK_RESULT_S:
            result = "tapped, unclear"
        elif targeted_again:
            result = "survived"
        else:
            result = "blocked or died"
        first = ep_taps[0] if ep_taps else None
        dist = radius = None
        if first is not None and first["ball_x"] is not None:
            dist = ((first["ball_x"] - cx) ** 2 + (first["ball_y"] - cy) ** 2) ** 0.5
            radius = first.get("ball_radius")
        rows.append({
            "start": start - t0, "end": end - t0, "taps": len(ep_taps), "result": result,
            "first_tap_after": None if first is None else first["t"] - start,
            "tap_dist": dist, "tap_radius": radius,
            "red_ball_seen": sum(e["state"] == "targeting" for e in ep) / len(ep),
        })

    return {
        "path": str(path),
        "duration": entries[-1]["t"] - t0,
        "ball_seen": sum(e["ball_x"] is not None for e in entries) / len(entries),
        "episodes": rows,
        "taps": len(taps),
        "wasted_taps": sum(id(e) not in counted_taps for e in taps),
    }


def summarize(results):
    episodes = [r for res in results for r in res["episodes"]]
    counts = {k: sum(r["result"] == k for r in episodes)
              for k in ("survived", "blocked or died", "tapped, unclear", "no tap")}
    taps = sum(res["taps"] for res in results)
    wasted = sum(res["wasted_taps"] for res in results)
    return episodes, counts, taps, wasted


def print_report(res):
    print(f"\n=== {res['path']} ===")
    print(f"length {res['duration']:.0f}s | ball detected in {100 * res['ball_seen']:.0f}% "
          f"of frames | {res['taps']} block taps, {res['wasted_taps']} while not targeted")
    if not res["episodes"]:
        print("  (never targeted)")
        return
    print(f"  {'targeted':>15}  {'taps':>4}  {'result':16} {'tap after':>9} "
          f"{'ball dist':>9} {'radius':>6} {'red seen':>8}")
    for r in res["episodes"]:
        after = "-" if r["first_tap_after"] is None else f"{r['first_tap_after']:.2f}s"
        dist = "-" if r["tap_dist"] is None else f"{r['tap_dist']:.0f}px"
        radius = "-" if r["tap_radius"] is None else f"{r['tap_radius']:.0f}"
        print(f"  {r['start']:6.1f}-{r['end']:6.1f}s  {r['taps']:>4}  {r['result']:16} "
              f"{after:>9} {dist:>9} {radius:>6} {100 * r['red_ball_seen']:7.0f}%")
    _, counts, _, _ = summarize([res])
    print("  " + ", ".join(f"{v} {k}" for k, v in counts.items()))


def find_logs(all_logs):
    logs = sorted(Path("live_logs").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if all_logs:
        return sorted(Path(".").glob("live_session*.jsonl")) + logs
    return logs[-1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*", help="Log files (default: newest in live_logs/)")
    parser.add_argument("--all", action="store_true", help="Score every log found")
    args = parser.parse_args()

    paths = [Path(p) for p in args.logs] or find_logs(args.all)
    if not paths:
        sys.exit("No logs found. Play with: python play_live.py model.joblib --log")

    cfg = track_ball.load_config()
    results = []
    for path in paths:
        res = score(path, cfg)
        if res is None:
            print(f"\n(skipping {path}: made by an older version that didn't log targeting/taps)")
            continue
        results.append(res)
        print_report(res)

    if len(results) > 1:
        episodes, counts, taps, wasted = summarize(results)
        print(f"\n=== TOTAL over {len(results)} logs ===")
        print(f"{len(episodes)} times targeted: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
        print(f"{taps} block taps, {wasted} while not targeted")


if __name__ == "__main__":
    main()
