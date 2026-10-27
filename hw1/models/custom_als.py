import numpy as np

from models.als import ExplicitALS


class RecencyALS:
    def __init__(self, n_factors=20, reg=1.0, n_epochs=5, decay=0.5, seed=42):
        self.n_factors = n_factors
        self.reg = reg
        self.n_epochs = n_epochs
        self.decay = decay
        self.seed = seed

    def fit(self, interactions):
        reference_time = interactions["timestamp"].max()
        age_in_years = (
            (reference_time - interactions["timestamp"]).dt.total_seconds()
            / (365 * 24 * 60 * 60)
        )
        target = (
            np.maximum(interactions["overall"].to_numpy(dtype=float) - 3, 0)
            / 2
            * np.exp(-self.decay * age_in_years)
        )
        prepared = interactions.copy()
        prepared["overall"] = target
        self.model = ExplicitALS(
            n_factors=self.n_factors,
            reg=self.reg,
            n_epochs=self.n_epochs,
            seed=self.seed,
        ).fit(prepared)
        return self

    def predict(self, user_id, item_id):
        return self.model.predict(user_id, item_id)

    def recommend(self, user_id, seen_items, candidate_items, k=10):
        return self.model.recommend(user_id, seen_items, candidate_items, k)

    def recommend_many(self, user_ids, seen_items, candidate_items, k=10):
        return self.model.recommend_many(user_ids, seen_items, candidate_items, k)
