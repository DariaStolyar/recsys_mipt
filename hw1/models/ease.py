import numpy as np
import pandas as pd
from scipy import sparse


class EASE:
    def __init__(self, reg=100.0):
        self.reg = reg

    def fit(self, interactions):
        user_codes, self.user_ids = pd.factorize(interactions["reviewerID"], sort=True)
        item_codes, self.item_ids = pd.factorize(interactions["asin"], sort=True)
        values = np.ones(len(interactions), dtype=float)

        self.matrix = sparse.csr_matrix(
            (values, (user_codes, item_codes)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )
        self.user_to_index = {
            user_id: index for index, user_id in enumerate(self.user_ids)
        }
        self.item_to_index = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }

        gram = (self.matrix.T @ self.matrix).toarray()
        gram.flat[:: gram.shape[0] + 1] += self.reg
        precision = np.linalg.inv(gram)
        self.coefficients = -precision / np.diag(precision)
        np.fill_diagonal(self.coefficients, 0)
        return self

    def _scores(self, user_id):
        if user_id not in self.user_to_index:
            return np.zeros(len(self.item_ids))
        profile = self.matrix.getrow(self.user_to_index[user_id])
        return np.asarray(profile @ self.coefficients).ravel()

    def predict(self, user_id, item_id):
        if item_id not in self.item_to_index:
            return 0.0
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
