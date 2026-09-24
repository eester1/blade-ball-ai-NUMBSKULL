"""
Model wrapper saved inside model.joblib. It lives in its own module (not in
train_model.py) because pickle records where a class is defined: a class
defined in a script run directly is saved as "__main__.<name>", which then
can't be found when play_live.py loads the model.
"""

import numpy as np


class SeedEnsemble:
    """Averages several identically configured MLPs that differ only in their
    random initialization. On this little data a single MLP's results swing
    noticeably with its seed (block F1 0.29-0.33 across seeds); averaging
    smooths that out instead of making model quality a matter of luck."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
