import torch
import torch.nn as nn


class LocalObjective:

    def __init__(self, labels_prefix):
        self._labels_prefix = labels_prefix

        # TODO: add scalable softmax
        self._loss = nn.CrossEntropyLoss()

    def __call__(self, inputs):
        all_logits = inputs["local_prediction"]  # (all_items, num_classes)
        all_labels = inputs["{}.ids".format(self._labels_prefix)]  # (all_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)  # (1)

        return loss


class GlobalObjective:

    def __init__(
        self,
        positive_prefix,
        negative_prefix,
    ):
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix

    def __call__(self, inputs):
        pos_scores = inputs[self._positive_prefix]  # (all_batch_items)
        neg_scores = inputs[self._negative_prefix]  # (all_batch_items)

        loss = -(pos_scores - neg_scores).sigmoid().log().mean()  # (1)

        return loss


class FusionObjective:

    def __init__(
        self,
        positive_prefix,
        negative_prefix,
    ):
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix

    def __call__(self, inputs):
        pos_scores = inputs[self._positive_prefix]  # (all_batch_items)
        neg_scores = inputs[self._negative_prefix]  # (all_batch_items)

        loss = -(pos_scores - neg_scores).sigmoid().log().mean()  # (1)

        return loss


class ContrastiveObjective:

    def __init__(
        self,
        fst_embeddings_prefix,
        snd_embeddings_prefix,
        tau=1.0,
        normalize_embeddings=False,
        use_mean=True,
    ):
        self._fst_embeddings_prefix = fst_embeddings_prefix
        self._snd_embeddings_prefix = snd_embeddings_prefix
        self._tau = tau
        self._loss_function = nn.CrossEntropyLoss(
            reduction="mean" if use_mean else "sum"
        )
        self._normalize_embeddings = normalize_embeddings

    def __call__(self, inputs):
        fst_embeddings = inputs[self._fst_embeddings_prefix]  # (x, embedding_dim)
        snd_embeddings = inputs[self._snd_embeddings_prefix]  # (x, embedding_dim)

        batch_size = fst_embeddings.shape[0]

        combined_embeddings = torch.cat(
            (fst_embeddings, snd_embeddings), dim=0
        )  # (2 * x, embedding_dim)

        if self._normalize_embeddings:
            combined_embeddings = torch.nn.functional.normalize(
                combined_embeddings, p=2, dim=-1, eps=1e-6
            )  # (2 * x, embedding_dim)

        similarity_scores = (
            torch.mm(combined_embeddings, combined_embeddings.T) / self._tau
        )  # (2 * x, 2 * x)

        positive_samples = torch.cat(
            (
                torch.diag(similarity_scores, batch_size),
                torch.diag(similarity_scores, -batch_size),
            ),
            dim=0,
        ).reshape(
            2 * batch_size, 1
        )  # (2 * x, 1)
        if not torch.allclose(
            torch.diag(similarity_scores, batch_size),
            torch.diag(similarity_scores, -batch_size),
        ):
            print(
                torch.diag(similarity_scores, batch_size),
                torch.diag(similarity_scores, -batch_size),
            )
            assert False

        mask = torch.ones(
            2 * batch_size, 2 * batch_size, dtype=torch.bool
        )  # (2 * x, 2 * x)
        mask = mask.fill_diagonal_(0)  # Remove equal embeddings scores
        for i in range(batch_size):  # Remove positives
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        negative_samples = similarity_scores[mask].reshape(
            2 * batch_size, -1
        )  # (2 * x, 2 * x - 2)

        labels = (
            torch.zeros(2 * batch_size).to(positive_samples.device).long()
        )  # (2 * x)
        logits = torch.cat(
            (positive_samples, negative_samples), dim=1
        )  # (2 * x, 2 * x - 1)

        loss = self._loss_function(logits, labels) / 2  # (1)

        return loss


class MRGSRecLoss:

    def __init__(
        self,
        local_objective,
        global_objective,
        fusion_objective,
        contrastive_objective,
    ):
        self._local_objective = local_objective
        self._global_objective = global_objective
        self._fusion_objective = fusion_objective
        self._contrastive_objective = contrastive_objective

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            local_objective=LocalObjective(
                labels_prefix=config["local"]["labels_prefix"],
            ),
            global_objective=GlobalObjective(
                positive_prefix=config["global"]["positive_prefix"],
                negative_prefix=config["global"]["negative_prefix"],
            ),
            fusion_objective=FusionObjective(
                positive_prefix=config["fusion"]["positive_prefix"],
                negative_prefix=config["fusion"]["negative_prefix"],
            ),
            contrastive_objective=ContrastiveObjective(
                fst_embeddings_prefix=config["contrastive"]["fst_embeddings_prefix"],
                snd_embeddings_prefix=config["contrastive"]["snd_embeddings_prefix"],
                tau=config["contrastive"].get("tau", 1.0),
                normalize_embeddings=config["contrastive"].get(
                    "normalize_embeddings", True
                ),
                use_mean=config["contrastive"].get("use_mean", True),
            ),
        )

    def __call__(self, inputs):
        return (
            1 * self._local_objective(inputs)
            + 0.2 * self._global_objective(inputs)
            + 0.4 * self._fusion_objective(inputs)
            + 0.05 * self._contrastive_objective(inputs)
        )
