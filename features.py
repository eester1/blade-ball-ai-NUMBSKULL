"""
Per-frame feature computation shared by build_dataset.py (training) and
play_live.py (live play), so the model sees exactly the same feature
definitions in both places. Feed detections in time order to a
FeatureTracker and it returns one feature row per detection.

Screen position alone can't tell how close the ball is -- a ball near
screen center can be right next to you or across the arena. The ball's
apparent size can: it grows as the ball approaches. So besides position
and screen velocity, this tracks the ball's radius, how fast the radius
is growing, and the resulting time-to-contact estimate (radius divided
by its growth rate -- the classic "looming" cue), which is what actually
tells you when to block.

Whether the ball is after *you* is read from your own character's red
"targeted" tint (self_red, see track_ball.self_highlight_score) as well as
the ball's color: in the recordings the ball was only picked up as red on
about half the frames where you were actually targeted.
"""

FEATURE_COLUMNS = [
    "ball_rel_x", "ball_rel_y", "ball_vel_x", "ball_vel_y",
    "ball_distance", "ball_closing_speed", "state_targeting",
    "ball_radius", "ball_radius_rate", "ball_ttc", "self_red",
]

# Below this dt, elapsed time is treated as zero (timestamp glitch) rather
# than dividing by a near-zero number and producing a huge bogus spike.
MIN_DT = 1e-3
# A gap longer than this since the previous detection means motion history
# is stale (ball was lost, round restarted, camera turned) -- velocity and
# radius growth restart from zero instead of spanning the gap.
MAX_GAP_S = 1.0
# Contour area jitters frame to frame, so the radius is smoothed before
# its growth rate is taken. Weight given to the newest measurement.
RADIUS_SMOOTHING = 0.5
# Time-to-contact reported when the ball isn't approaching (or is
# approaching too slowly to matter), and the upper clip otherwise.
TTC_CAP_S = 3.0


class FeatureTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        """Forget motion history. Call when the ball is lost or when screen
        motion is known to be fake (e.g. the camera just turned, which
        sweeps everything across the screen)."""
        self._prev = None  # (x, y, smoothed_radius, t)

    def update(self, x, y, state, radius, self_red, t, center_x, center_y):
        prev = self._prev
        dt = None if prev is None else t - prev[3]
        fresh = prev is None or dt < MIN_DT or dt > MAX_GAP_S

        if fresh:
            r_smooth = radius
            vel_x = vel_y = r_rate = 0.0
        else:
            r_smooth = RADIUS_SMOOTHING * radius + (1 - RADIUS_SMOOTHING) * prev[2]
            vel_x = (x - prev[0]) / dt
            vel_y = (y - prev[1]) / dt
            r_rate = (r_smooth - prev[2]) / dt
        self._prev = (x, y, r_smooth, t)

        rel_x, rel_y = x - center_x, y - center_y
        distance = (rel_x ** 2 + rel_y ** 2) ** 0.5
        # Positive = ball closing in on screen center, negative = moving away.
        closing_speed = 0.0 if distance < 1e-6 else -(rel_x * vel_x + rel_y * vel_y) / distance
        ttc = min(r_smooth / r_rate, TTC_CAP_S) if r_rate > 1e-6 else TTC_CAP_S

        return {
            "ball_rel_x": rel_x,
            "ball_rel_y": rel_y,
            "ball_vel_x": vel_x,
            "ball_vel_y": vel_y,
            "ball_distance": distance,
            "ball_closing_speed": closing_speed,
            "state_targeting": 1 if state == "targeting" else 0,
            "ball_radius": r_smooth,
            "ball_radius_rate": r_rate,
            "ball_ttc": ttc,
            "self_red": self_red,
        }
