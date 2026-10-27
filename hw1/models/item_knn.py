import numpy as np
import pandas as pd
from scipy import sparse


class ItemKNN:
    def __init__(self, n_neighbors=50):
        self.n_neighbors = n_neighbors

    def fit(self, interactions):
        user_codes, self.user_ids = pd.factorize(interactions["reviewerID"], sort=True)
        item_codes, self.item_ids = pd.factorize(interactions["asin"], sort=True)
        values = interactions["overall"].to_numpy(dtype=float)

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
        self.global_mean = values.mean()

        item_norms = np.sqrt(self.matrix.power(2).sum(axis=0)).A1
        similarity = (self.matrix.T @ self.matrix).toarray()
        denominator = np.outer(item_norms, item_norms)
        similarity = np.divide(
            similarity,
            denominator,
            out=np.zeros_like(similarity),
            where=denominator != 0,
        )
        np.fill_diagonal(similarity, 0)

        if self.n_neighbors < similarity.shape[0]:
            for index in range(similarity.shape[0]):
                removed = np.argpartition(similarity[index], -self.n_neighbors)[
                    :-self.n_neighbors
                ]
                similarity[index, removed] = 0

        self.similarity = similarity
        rating_sum = np.asarray(self.matrix.sum(axis=0)).ravel()
        rating_count = np.diff(self.matrix.tocsc().indptr)
        self.item_mean = rating_sum / rating_count
        return self

    def _scores(self, user_id):
        if user_id not in self.user_to_index:
            return self.item_mean.copy()

        profile = self.matrix.getrow(self.user_to_index[user_id])
        scores = np.asarray(profile @ self.similarity).ravel()
        weights = np.asarray(profile.sign() @ self.similarity).ravel()
        return np.divide(
            scores,
            weights,
            out=self.item_mean.copy(),
            where=weights != 0,
        )

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
        seen_indices = [
            self.item_to_index[item_id]
            for item_id in seen_items
            if item_id in self.item_to_index
        ]
        available_scores = np.full(scores.shape, -np.inf)
        available_scores[candidate_indices] = scores[candidate_indices]
        scores = available_scores
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
