import numpy as np
import pandas as pd
import torch
from torch import nn


class _FMNetwork(nn.Module):
    def __init__(self, field_sizes, n_factors):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.ModuleList([nn.Embedding(size, 1) for size in field_sizes])
        self.factors = nn.ModuleList(
            [nn.Embedding(size, n_factors) for size in field_sizes]
        )
        for embedding in self.factors:
            nn.init.normal_(embedding.weight, std=0.01)

    def forward(self, features):
        linear_part = sum(
            embedding(features[:, field]).squeeze(1)
            for field, embedding in enumerate(self.linear)
        )
        factors = torch.stack([
            embedding(features[:, field])
            for field, embedding in enumerate(self.factors)
        ], dim=1)
        interaction_part = 0.5 * (
            factors.sum(dim=1).square().sum(dim=1)
            - factors.square().sum(dim=(1, 2))
        )
        return self.bias + linear_part + interaction_part


class FactorizationMachine:
    def __init__(
        self,
        n_factors=8,
        learning_rate=0.01,
        n_epochs=10,
        batch_size=2048,
        seed=42,
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, interactions, item_metadata):
        torch.manual_seed(self.seed)
        user_codes, self.user_ids = pd.factorize(interactions["reviewerID"], sort=True)
        item_codes, self.item_ids = pd.factorize(interactions["asin"], sort=True)
        self.user_to_index = {
            user_id: index for index, user_id in enumerate(self.user_ids)
        }
        self.item_to_index = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }

        metadata = item_metadata.drop_duplicates("asin").set_index("asin").reindex(
            self.item_ids
        )
        brands = metadata["brand"].fillna("unknown").astype(str)
        categories = metadata["categories"].map(self._category).fillna("unknown")
        brand_codes, self.brand_values = pd.factorize(brands, sort=True)
        category_codes, self.category_values = pd.factorize(categories, sort=True)
        self.item_features = np.column_stack([
            np.arange(len(self.item_ids)), brand_codes, category_codes
        ])
        features = np.column_stack(
            [user_codes, item_codes, brand_codes[item_codes], category_codes[item_codes]]
        )
        targets = interactions["overall"].to_numpy(dtype=np.float32)
        self.global_mean = float(targets.mean())
        self.network = _FMNetwork([
            len(self.user_ids),
            len(self.item_ids),
            len(self.brand_values),
            len(self.category_values)
        ], self.n_factors)
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        feature_tensor = torch.tensor(features, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        self.network.train()
        for _ in range(self.n_epochs):
            order = torch.randperm(len(target_tensor))
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                prediction = self.network(feature_tensor[indices])
                loss = (prediction - target_tensor[indices]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.item_feature_tensor = torch.tensor(
            self.item_features, dtype=torch.long
        )
        self.network.eval()
        return self

    def _category(self, value):
        while isinstance(value, (list, tuple)) and len(value):
            value = value[-1]
        if isinstance(value, (list, tuple)):
            return "unknown"
        if pd.isna(value):
            return "unknown"
        return str(value)

    def _scores(self, user_id):
        if user_id not in self.user_to_index:
            return np.full(len(self.item_ids), self.global_mean)
        item_features = self.item_feature_tensor
        features = torch.column_stack([
            torch.full(
                (len(item_features),),
                self.user_to_index[user_id],
                dtype=torch.long
            ),
            item_features
        ])
        with torch.no_grad():
            return self.network(features).numpy()

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
        seen_indices = [
            self.item_to_index[item_id]
            for item_id in seen_items
            if item_id in self.item_to_index
        ]
        available_scores[seen_indices] = -np.inf
        n_recommendations = min(k, np.isfinite(available_scores).sum())
        if n_recommendations == 0:
            return []
        top_indices = np.argpartition(available_scores, -n_recommendations)[-n_recommendations:]
        top_indices = top_indices[np.argsort(available_scores[top_indices])[::-1]]
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
