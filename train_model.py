"""
Train a small neural network to predict which keys/buttons you'd hold,
given the ball's position, velocity, and state (idle/targeting).

SETUP (run once):
    pip install scikit-learn pandas joblib

USAGE:
    python train_model.py dataset.csv -o model.joblib

    Re-run this any time you've recorded more sessions and rebuilt a
    bigger dataset.csv -- it always trains fresh from scratch on
    whatever's in the CSV.
"""

import argparse
import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

FEATURE_COLUMNS = [
    "ball_rel_x", "ball_rel_y", "ball_vel_x", "ball_vel_y",
    "ball_distance", "ball_closing_speed", "state_targeting",
]
LABEL_COLUMNS = [
    "held_w", "held_a", "held_s", "held_d",
    "held_mouse_left", "held_mouse_right",
]


def load_dataset(path):
    df = pd.read_csv(path)
    df["state_targeting"] = (df["ball_state"] == "targeting").astype(int)
    df["ball_speed"] = (df["ball_vel_x"] ** 2 + df["ball_vel_y"] ** 2) ** 0.5
    # Sort by session then time, so the time-based split below is meaningful
    df = df.sort_values(["session", "t"]).reset_index(drop=True)
    return df


def time_based_split(df, test_frac=0.2):
    """Holds out the last test_frac of EACH session (in time order) as test data,
    rather than a random split, so the model isn't tested on near-duplicate
    frames it basically already saw in training."""
    train_parts, test_parts = [], []
    for _, group in df.groupby("session"):
        cutoff = int(len(group) * (1 - test_frac))
        train_parts.append(group.iloc[:cutoff])
        test_parts.append(group.iloc[cutoff:])
    return pd.concat(train_parts), pd.concat(test_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to dataset.csv from build_dataset.py")
    parser.add_argument("-o", "--output", default="model.joblib")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[32, 16],
                         help="Hidden layer sizes, e.g. --hidden-sizes 64 32")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    print(f"Loaded {len(df)} rows from {args.dataset}")

    train_df, test_df = time_based_split(df)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows (last 20% of each session)")

    # Blocking/ability are rare compared to movement, so the model can
    # score well by just always predicting "not held." Oversample those
    # rare-but-important rows in the TRAINING set only (never touch the
    # test set, or we'd be grading on an easier, unrealistic test).
    RARE_LABELS = ["held_mouse_left", "held_mouse_right"]
    OVERSAMPLE_FACTOR = 5
    rare_rows = train_df[train_df[RARE_LABELS].any(axis=1)]
    if len(rare_rows) > 0:
        extra = pd.concat([rare_rows] * (OVERSAMPLE_FACTOR - 1), ignore_index=True)
        train_df = pd.concat([train_df, extra], ignore_index=True)
        print(f"Oversampled {len(rare_rows)} rare-action rows x{OVERSAMPLE_FACTOR} "
              f"-> train set now {len(train_df)} rows")

    # Fast-ball frames are just as rare as blocking is, for the same reason:
    # most of a match is slower early-round play, so the few frames where
    # the ball is genuinely fast get drowned out and the model barely
    # learns from them. Oversample the fastest slice of TRAINING rows only
    # (threshold is relative to this dataset's own speed distribution, so
    # it stays meaningful as you add faster recordings later).
    FAST_SPEED_QUANTILE = 0.85
    FAST_OVERSAMPLE_FACTOR = 3
    speed_threshold = train_df["ball_speed"].quantile(FAST_SPEED_QUANTILE)
    fast_rows = train_df[train_df["ball_speed"] >= speed_threshold]
    if len(fast_rows) > 0:
        extra = pd.concat([fast_rows] * (FAST_OVERSAMPLE_FACTOR - 1), ignore_index=True)
        train_df = pd.concat([train_df, extra], ignore_index=True)
        print(f"Oversampled {len(fast_rows)} fast-ball rows "
              f"(speed >= {speed_threshold:.0f} px/s) x{FAST_OVERSAMPLE_FACTOR} "
              f"-> train set now {len(train_df)} rows")

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[LABEL_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[LABEL_COLUMNS].values

    # Neural nets train much better when inputs are on similar scales --
    # pixel positions/velocities can be in the hundreds, so we normalize.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining MLP with hidden layers {args.hidden_sizes} ...")
    model = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_sizes),
        max_iter=2000,
        random_state=0,
    )
    model.fit(X_train_scaled, y_train)

    print("\n--- Test set performance (per action) ---")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=LABEL_COLUMNS, zero_division=0))

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
    }, args.output)
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()