from collections import Counter

import torch


class BaseMetric:
    pass


class StatefullMetric(BaseMetric):
    def reduce(self):
        raise NotImplementedError


class StaticMetric:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __call__(self, inputs):
        inputs[self._name] = self._value

        return inputs


class CompositeMetric:
    def __init__(self, metrics):
        self._metrics = metrics

    @classmethod
    def create_from_config(cls, config):
        return cls(
            metrics=[BaseMetric.create_from_config(cfg) for cfg in config["metrics"]]
        )

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


class NDCGMetric:
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs):
        predictions = inputs["logits"][
            :, : self._k
        ].float()  # (batch_size, top_k_indices)
        labels = inputs["labels.ids"].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(
            predictions, labels[..., None]
        ).float()  # (batch_size, top_k_indices)
        discount_factor = 1 / torch.log2(
            torch.arange(1, self._k + 1, 1).float() + 1.0
        ).to(
            hits.device
        )  # (k)
        dcg = hits @ discount_factor  # (batch_size)

        return dcg.cpu().tolist()


class RecallMetric:
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs):
        predictions = inputs["logits"][
            :, : self._k
        ].float()  # (batch_size, top_k_indices)
        labels = inputs["labels.ids"].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(
            predictions, labels[..., None]
        ).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class CoverageMetric(StatefullMetric):
    def __init__(self, k, num_items):
        self._k = k
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(k=config["k"], num_items=kwargs["num_items"])

    def __call__(self, inputs):
        predictions = inputs["logits"][:, : self._k]  # (batch_size, top_k_indices)
        return predictions.reshape(-1).cpu().detach().tolist()  # (batch_size * k)

    def reduce(self, values):
        return len(set(values)) / self._num_items


class NoveltyMetric(StatefullMetric):
    """
    Novelty(i) = 1 - (#users recommended i) / (#users that have NOT interacted with i)
    """

    def __init__(self, k, item2num_iteractions, num_users):
        self._k = k
        self._item2num_iteractions = item2num_iteractions
        self._num_users = num_users

    def __call__(self, inputs):
        predictions = inputs["logits"][:, : self._k]  # (batch_size, top_k_indices)
        return predictions.reshape(-1).cpu().detach().tolist()  # (batch_size * k)

    def reduce(self, recommended_items):
        """
        recommended_items: flat list of item ids for ALL users in epoch/batch
        returns: average novelty over all items seen in recommendations
        """

        novelty_values = []
        rec_count = Counter(recommended_items)

        for item, recommended_times in rec_count.items():
            # number of users who have NOT interacted with this item
            no_interacted = self._num_users - self._item2num_iteractions[item]

            if no_interacted == 0:
                continue  # skip items impossible to recommend

            novelty = 1.0 - (recommended_times / no_interacted)
            novelty_values.append(novelty)

        if len(novelty_values) == 0:
            return 0.0

        return sum(novelty_values) / len(novelty_values)


class ILDMetric:
    """
    Vectorized ILD:
    ILD(u) = mean over i<j of dist(item_i, item_j)
    """

    def __init__(self, k, dist_matrix):
        self._k = k
        self._dist = dist_matrix  # (num_items, num_items)
        self._triu = torch.triu_indices(k, k, offset=1)  # (2, K(K - 1)/2)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            k=config["k"],
            dist_matrix=kwargs["dist_matrix"],
        )

    def __call__(self, inputs):
        """
        inputs["logits"]: (B, K) item ids
        Returns: tensor (B,) — ILD per user
        """
        topk = inputs["logits"][:, : self._k]  # (B, K)

        sub = self._dist[topk[:, None], topk[:, :, None]]  # (B, (K, K))

        pairwise = sub[:, self._triu[0], self._triu[1]]  # (B, K(K - 1)/2)

        ild_per_user = pairwise.mean(dim=1)  # (B,)

        return ild_per_user.tolist()
