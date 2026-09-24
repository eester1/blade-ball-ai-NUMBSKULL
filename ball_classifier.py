"""
A small learned "is this blob really the ball?" scorer, layered on top of
track_ball.py's color/shape detection.

Color and shape rules alone keep getting fooled by things that are also
round and white/red on some map: sparkles, lanterns, torches, banner
letters, players' cosmetics. Each fix so far targeted one decoy. This
learns what separates the real ball from all of them at once, from the
look of each candidate: color and brightness inside it, how uniform its
inside is, how sharp its edge is, and how it contrasts with its
surroundings.

Training data is generated from the existing recordings -- no manual
labeling. Positives: the ball the tracker followed, but only on frames
where it was actually moving (static decoys the tracker sometimes locked
onto sit still). Negatives: every other candidate detected in the same
frame, well away from the ball.

USAGE:
    python ball_classifier.py        # (re)train ball_classifier.joblib from recordings/

track_ball.py and play_live.py use the saved model automatically if it
exists (candidates it scores below min_ball_score are dropped), and work
without it.
"""

import json
from pathlib import Path

import cv2
import joblib
import numpy as np

import track_ball

MODEL_PATH = Path("ball_classifier.joblib")
# A positive must have moved at least this far since the previous tracked
# frame (static decoys don't move) but not implausibly far.
MIN_MOVE_PX, MAX_MOVE_PX = 2, 400
# Negatives must be at least this far from the tracked ball.
NEGATIVE_MIN_DIST_PX = 60

FEATURE_NAMES = [
    "radius", "circularity", "is_red", "y_frac",
    "in_s_mean", "in_s_std", "in_v_mean", "in_v_std", "in_hue_sin", "in_hue_cos",
    "ring_s_mean", "ring_s_std", "ring_v_mean", "ring_v_std", "ring_hue_sin", "ring_hue_cos",
    "v_contrast", "s_contrast", "edge_strength", "inner_gradient",
]


def candidate_features(hsv, candidate):
    """Feature vector for one (circularity, x, y, state, radius) candidate,
    from the frame in HSV."""
    circularity, x, y, state, radius = candidate
    h, w = hsv.shape[:2]
    r = max(float(radius), 3.0)
    x0, x1 = max(0, int(x - 2 * r)), min(w, int(x + 2 * r) + 1)
    y0, y1 = max(0, int(y - 2 * r)), min(h, int(y + 2 * r) + 1)
    patch = hsv[y0:y1, x0:x1].astype(np.float32)
    # Edge strength of the V channel. Sobel only looks at immediate
    # neighbours, so computing it on the patch plus a 1px margin gives the
    # same values as on the whole frame, far faster.
    mx0, mx1, my0, my1 = max(0, x0 - 1), min(w, x1 + 1), max(0, y0 - 1), min(h, y1 + 1)
    g = frame_gradient(hsv[my0:my1, mx0:mx1])[y0 - my0:y0 - my0 + (y1 - y0),
                                              x0 - mx0:x0 - mx0 + (x1 - x0)]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r
    inner, ring = d <= 0.7, (d >= 1.3) & (d <= 2.0)
    edge = (d >= 0.85) & (d <= 1.15)

    def stats(mask):
        if not mask.any():
            return [0.0] * 6
        hue = patch[..., 0][mask] * (2 * np.pi / 180)
        s, v = patch[..., 1][mask], patch[..., 2][mask]
        return [s.mean(), s.std(), v.mean(), v.std(), np.sin(hue).mean(), np.cos(hue).mean()]

    inside, around = stats(inner), stats(ring)
    return [
        r, circularity, 1.0 if state == "targeting" else 0.0, y / h,
        *inside, *around,
        inside[2] - around[2], inside[0] - around[0],
        g[edge].mean() if edge.any() else 0.0,
        g[inner].mean() if inner.any() else 0.0,
    ]


def frame_gradient(hsv):
    """Gradient magnitude of the V (brightness) channel."""
    v = hsv[..., 2]
    return cv2.magnitude(cv2.Sobel(v, cv2.CV_32F, 1, 0), cv2.Sobel(v, cv2.CV_32F, 0, 1))


class BallScorer:
    """Scores candidates with the trained model (probability it's the ball)."""

    def __init__(self, path=MODEL_PATH):
        self.model = joblib.load(path)

    def filter(self, frame_bgr, candidates, cores, min_score):
        """Drop candidates/cores the model is confident aren't the ball."""
        if not candidates and not cores:
            return candidates, cores
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        both = candidates + cores
        scores = self.model.predict_proba(
            np.array([candidate_features(hsv, c) for c in both]))[:, 1]
        keep = [c for c, p in zip(both, scores) if p >= min_score]
        n = len(candidates)
        kept_ids = {id(c) for c in keep}
        return [c for c in both[:n] if id(c) in kept_ids], [c for c in both[n:] if id(c) in kept_ids]


def load_scorer():
    return BallScorer() if MODEL_PATH.exists() else None


def build_training_set(cfg):
    X, y, groups = [], [], []
    for session_dir in sorted(Path("recordings").glob("*/")):
        tracks_path = session_dir / "ball_tracks.jsonl"
        if not tracks_path.exists():
            continue
        tracks = [json.loads(line) for line in open(tracks_path) if line.strip()]
        prev = None
        n_pos = n_neg = 0
        for t in tracks:
            if t["ball_x"] is None:
                prev = None
                continue
            moved = None if prev is None else \
                ((t["ball_x"] - prev[0]) ** 2 + (t["ball_y"] - prev[1]) ** 2) ** 0.5
            prev = (t["ball_x"], t["ball_y"])
            if moved is None or not MIN_MOVE_PX <= moved <= MAX_MOVE_PX:
                continue
            frame = cv2.imread(str(session_dir / "frames" / t["frame"]))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            candidates, _ = track_ball.find_ball_candidates(frame, cfg)
            for c in candidates:
                dist = ((c[1] - t["ball_x"]) ** 2 + (c[2] - t["ball_y"]) ** 2) ** 0.5
                if dist <= 3:
                    label = 1
                elif dist >= NEGATIVE_MIN_DIST_PX:
                    label = 0
                else:
                    continue
                X.append(candidate_features(hsv, c))
                y.append(label)
                groups.append(session_dir.name)
                n_pos += label
                n_neg += 1 - label
        print(f"{session_dir.name}: {n_pos} ball / {n_neg} not-ball examples")
    return np.array(X), np.array(y), np.array(groups)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report

    cfg = track_ball.load_config()
    X, y, groups = build_training_set(cfg)
    print(f"\n{len(y)} examples: {int(y.sum())} ball, {int(len(y) - y.sum())} not-ball")

    # Hold out whole sessions (different maps/lighting), not random frames --
    # the point is to generalize to places it hasn't seen.
    sessions = sorted(set(groups))
    test_sessions = set(sessions[::4])
    test = np.array([g in test_sessions for g in groups])
    model = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0)
    model.fit(X[~test], y[~test])
    proba = model.predict_proba(X[test])[:, 1]
    print(f"\nHeld-out sessions: {sorted(test_sessions)}")
    for threshold in (0.1, 0.2, 0.3, 0.5):
        print(f"--- threshold {threshold} ---")
        print(classification_report(y[test], proba >= threshold,
                                    target_names=["not ball", "ball"], digits=3))

    model.fit(X, y)  # final model uses every session
    joblib.dump(model, MODEL_PATH)
    print(f"Saved {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
