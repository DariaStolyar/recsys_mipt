import numpy as np
import pandas as pd
from scipy import sparse


class ExplicitALS:
    def __init__(self, n_factors=20, reg=0.1, n_epochs=5, seed=42):
        self.n_factors = n_factors
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

    def fit(self, interactions):
        user_codes, self.user_ids = pd.factorize(interactions["reviewerID"], sort=True)
        item_codes, self.item_ids = pd.factorize(interactions["asin"], sort=True)
        values = interactions["overall"].to_numpy(dtype=float)

        self.matrix = sparse.csr_matrix(
            (values, (user_codes, item_codes)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )
        self.matrix_csc = self.matrix.tocsc()
        self.user_to_index = {
            user_id: index for index, user_id in enumerate(self.user_ids)
        }
        self.item_to_index = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }
        self.global_mean = values.mean()

        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(
            0,
            0.1,
            size=(len(self.user_ids), self.n_factors),
        )
        self.item_factors = rng.normal(
            0,
            0.1,
            size=(len(self.item_ids), self.n_factors),
        )
        identity = np.eye(self.n_factors)

        for _ in range(self.n_epochs):
            for user_index in range(len(self.user_ids)):
                start = self.matrix.indptr[user_index]
                end = self.matrix.indptr[user_index + 1]
                item_indices = self.matrix.indices[start:end]
                ratings = self.matrix.data[start:end]
                factors = self.item_factors[item_indices]
                left = factors.T @ factors + self.reg * identity
                right = factors.T @ ratings
                self.user_factors[user_index] = np.linalg.solve(left, right)

            for item_index in range(len(self.item_ids)):
                start = self.matrix_csc.indptr[item_index]
                end = self.matrix_csc.indptr[item_index + 1]
                user_indices = self.matrix_csc.indices[start:end]
                ratings = self.matrix_csc.data[start:end]
                factors = self.user_factors[user_indices]
                left = factors.T @ factors + self.reg * identity
                right = factors.T @ ratings
                self.item_factors[item_index] = np.linalg.solve(left, right)

        return self

    def _scores(self, user_id):
        if user_id not in self.user_to_index:
            return np.full(len(self.item_ids), self.global_mean)
        user_index = self.user_to_index[user_id]
        return self.user_factors[user_index] @ self.item_factors.T

    def predict(self, user_id, item_id):
        if item_id not in self.item_to_index:
            return self.global_mean
        return self._scores(user_id)[self.item_to_index[item_id]]

    def _candidate_indices(self, candidate_items):
        return np.array([
            index
            for index, item_id in enumerate(self.item_ids)
            if item_id in candidate_items
        ], dtype=int)

    def _recommend(self, user_id, seen_items, candidate_indices, k=10):
        scores = self._scores(user_id)
        available_scores = np.full(scores.shape, -np.inf)
        available_scores[candidate_indices] = scores[candidate_indices]
        scores = available_scores
        seen_indices = [
            self.item_to_index[item_id]
            for item_id in seen_items
            if item_id in self.item_to_index
        ]
        scores[seen_indices] = -np.inf

        n_recommendations = min(k, np.isfinite(scores).sum())
        if n_recommendations == 0:
            return []

        top_indices = np.argpartition(scores, -n_recommendations)[-n_recommendations:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return self.item_ids[top_indices].tolist()

    def recommend(self, user_id, seen_items, candidate_items, k=10):
        candidate_indices = self._candidate_indices(candidate_items)
        return self._recommend(user_id, seen_items, candidate_indices, k)

    def recommend_many(self, user_ids, seen_items, candidate_items, k=10):
        candidate_indices = self._candidate_indices(candidate_items)
        return {
            user_id: self._recommend(
                user_id,
                seen_items.get(user_id, set()),
                candidate_indices,
                k,
            )
            for user_id in user_ids
        }
