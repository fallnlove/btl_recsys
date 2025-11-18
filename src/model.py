import torch
import torch.nn as nn

from .graph import GraphEncoder
from .sequence import SequentialEncoder
from .utils import create_masked_tensor, get_activation_function


class MRGSRecModel(nn.Module):
    def __init__(
        self,
        cfg,
        num_items,
        num_users,
        max_sequence_length,
        graph,
    ):
        super().__init__()
        # dataset meta
        self._num_items = num_items
        self._num_users = num_users
        self._max_sequence_length = max_sequence_length
        # model params
        self._embedding_dim = cfg["embedding_dim"]
        self._num_heads = cfg["num_heads"]
        self._num_layers = cfg["num_layers"]
        self._dim_feedforward = cfg["dim_feedforward"]
        self._dropout = cfg["dropout"]
        self._activation = get_activation_function(cfg["activation"])
        self._layer_norm_eps = cfg["layer_norm_eps"]
        self._initializer_range = cfg["initializer_range"]
        self._num_hops = cfg["num_hops"]
        self._eta = cfg["eta"]
        self._topk_k = cfg["topk_k"]
        # layers
        self._graph_encoder = GraphEncoder(
            graph, self._dropout, num_users, num_items, self._num_hops
        )

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 2,
            embedding_dim=self._embedding_dim,
            padding_idx=0,
        )
        self._newuser_embeddings = nn.Embedding(
            num_embeddings=1, embedding_dim=self._embedding_dim
        )
        self._item_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2,
            embedding_dim=self._embedding_dim,
            padding_idx=0,
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length
            + 1,  # in order to include `max_sequence_length` value
            embedding_dim=self._embedding_dim,
        )

        self._sequential_encoder = SequentialEncoder(
            self._embedding_dim,
            self._num_heads,
            self._dim_feedforward,
            self._dropout,
            self._activation,
            self._layer_norm_eps,
            self._num_layers,
            self._num_items,
            self._position_embeddings,
            self._item_embeddings,
        )

        self._fusion_part = nn.Sequential(
            nn.Linear(2 * self._embedding_dim, self._dim_feedforward),
            self._activation,
            nn.Linear(self._dim_feedforward, self._embedding_dim),
        )

        self._init_weights(self._initializer_range)
        dummy = torch.empty(0)
        self.register_buffer("_device_anchor", dummy, persistent=False)

    @property
    def device(self):
        return self._device_anchor.device

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            num_items=kwargs["num_items"],
            num_users=kwargs["num_users"],
            max_sequence_length=kwargs["max_sequence_length"],
            embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            graph=kwargs["graph"],
            num_hops=config["num_hops"],
            dropout=config["dropout"],
            activation=config["activation"],
            layer_norm_eps=config["layer_norm_eps"],
            initializer_range=config["initializer_range"],
        )

    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if "weight" in key:
                if "norm" in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range,
                    )
            elif "bias" in key:
                nn.init.zeros_(value.data)
            else:
                raise ValueError(f"Unknown transformer weight: {key}")

    @staticmethod
    def _get_last_embedding(embeddings, mask):
        lengths = torch.sum(mask, dim=-1)  # (batch_size)
        lengths = lengths - 1  # (batch_size)
        assert torch.all(torch.gt(lengths, 0))
        last_masks = mask.gather(dim=1, index=lengths[:, None])  # (batch_size, 1)
        lengths = torch.tile(
            lengths[:, None, None], (1, 1, embeddings.shape[-1])
        )  # (batch_size, 1, emb_dim)
        last_embeddings = embeddings.gather(
            dim=1, index=lengths
        )  # (batch_size, 1, emb_dim)
        last_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)
        if not torch.allclose(embeddings[mask][-1], last_embeddings[-1]):
            print(embeddings)
            print(lengths, lengths.max(), lengths.min())
            print(embeddings[mask][-1])
            print(last_embeddings[-1])
            assert False
        return last_embeddings

    def _get_embeddings(self, ids, lengths, ego_embeddings, final_embeddings):
        final_embeddings = final_embeddings[ids]  # (all_batch_events, embedding_dim)
        ego_embeddings = ego_embeddings(ids)  # (all_batch_events, embedding_dim)

        padded_embeddings, mask = create_masked_tensor(
            final_embeddings, lengths, device=self.device
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        padded_ego_embeddings, ego_mask = create_masked_tensor(
            ego_embeddings, lengths, device=self.device
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        assert torch.all(mask == ego_mask)

        return padded_embeddings, padded_ego_embeddings, mask

    def forward(self, inputs, ind=None):
        sequence_user_embeddings, sequence_embeddings = self._sequential_part(inputs)

        # All embeddings after graph part
        (
            all_graph_enriched_user_embeddings,
            all_graph_enriched_item_embeddings,
        ) = self._graph_encoder(
            self._user_embeddings, self._item_embeddings, ind
        )  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)

        # Enriched embeddings of users from batch
        batch_graph_enriched_user_embeddings = all_graph_enriched_user_embeddings[
            inputs["user.ids"]
        ]

        # Fusion part
        fusion_user_embeddings = self._fuse_embeddings(
            sequence_user_embeddings, batch_graph_enriched_user_embeddings
        )

        if self.training:  # training mode
            return self._compute_training_outputs(
                inputs,
                sequence_embeddings,
                batch_graph_enriched_user_embeddings,
                fusion_user_embeddings,
                all_graph_enriched_item_embeddings,
            )
        else:
            return self._compute_inference_outputs(fusion_user_embeddings)

    def _sequential_part(self, inputs):
        all_sample_lengths = inputs[f"item.length"]
        all_sample_events = inputs[f"item.ids"]  # (all_batch_events)
        user_ids = inputs[f"user.ids"]

        padded_sequence, mask = self._sequential_encoder._pad_sequence(
            all_sample_events, all_sample_lengths
        )

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        inputs["mask"] = mask

        sequence_embeddings = self._item_embeddings(
            padded_sequence
        )  # (batch_size, seq_len, embedding_dim)

        sequence_embeddings = self._sequential_encoder._prepare_sequence(
            seq_len, mask, batch_size, all_sample_lengths, sequence_embeddings
        )

        sequence_user_embeddings = self._user_embeddings(user_ids).unsqueeze(
            1
        )  # (batch_size, 1, embedding_dim)

        sequence_embeddings = torch.cat(
            [sequence_user_embeddings, sequence_embeddings], dim=1
        )  # (batch_size, seq_len + 1, embedding_dim)

        sequence_embeddings = self._sequential_encoder._encode_sequence(
            seq_len, batch_size, mask, sequence_embeddings, True
        )

        sequence_user_embeddings, sequence_embeddings = (
            sequence_embeddings[:, 0, :],
            sequence_embeddings[:, 1:, :],
        )  # (batch_size, embedding_dim), (batch_size, seq_len, embedding_dim)

        return (
            sequence_user_embeddings,
            sequence_embeddings,
        )

    def _fuse_embeddings(self, sequence_user_embeddings, graph_user_embeddings):
        return self._fusion_part(
            torch.cat(
                [
                    sequence_user_embeddings,
                    (1 - self._eta) * sequence_user_embeddings
                    + self._eta * graph_user_embeddings,
                ],
                dim=1,
            )
        )  # (batch_size, embedding_dim)

    def _compute_inference_outputs(self, fusion_user_embeddings):
        # b - batch_size, n - num_candidates, d - embedding_dim
        candidate_scores = fusion_user_embeddings @ self._item_embeddings.weight.T
        # (batch_size, num_items + 2)

        candidate_scores[:, 0] = -torch.inf
        candidate_scores[:, self._num_items + 1 :] = -torch.inf

        _, indices = torch.topk(
            candidate_scores, k=self._topk_k, dim=-1, largest=True
        )  # (batch_size, _topk_k)

        return indices

    def _compute_training_outputs(
        self,
        inputs,
        sequence_embeddings,
        graph_user_embeddings,
        fusion_user_embeddings,
        all_final_item_embeddings,
    ):
        mask = inputs["mask"]
        batch_size = mask.shape[0]
        max_sequence_length = mask.shape[1]

        all_positive_sample_events = inputs[f"labels.ids"]  # (all_batch_events)
        all_positive_sample_lengths = inputs[f"labels.length"]  # (batch_size)

        bpr_positive_user_ids = self._get_bpr_positive_user_ids(
            max_sequence_length, batch_size, all_positive_sample_lengths
        )

        # Sequential part
        all_sample_sequence_embeddings = sequence_embeddings[
            mask
        ]  # (all_batch_events, embedding_dim)

        sequence_scores = all_sample_sequence_embeddings @ all_final_item_embeddings.T

        # (all_batch_events, num_items + 2)

        (
            graph_positive_scores,
            graph_scores,
            graph_user_embeddings,
        ) = self._get_graph_scores(
            graph_user_embeddings,
            bpr_positive_user_ids,
            all_final_item_embeddings,
            all_positive_sample_events,
        )

        fusion_positive_scores, fusion_scores = self._get_fusion_scores(
            fusion_user_embeddings, bpr_positive_user_ids, all_positive_sample_events
        )

        return {
            "local_prediction": sequence_scores,
            "global_positive": graph_positive_scores,
            "global_negative": graph_scores,
            "contrastive_fst_embeddings": all_sample_sequence_embeddings,
            "contrastive_snd_embeddings": graph_user_embeddings,
            "fusion_positive": fusion_positive_scores,
            "fusion_negative": fusion_scores,
        }

    def _get_bpr_positive_user_ids(
        self, max_sequence_length, batch_size, all_positive_sample_lengths
    ):
        bpr_mask = (
            torch.arange(end=max_sequence_length, device=self.device)[None].tile(
                [batch_size, 1]
            )
            < all_positive_sample_lengths[:, None]
        )  # (batch_size, max_seq_len)
        bpr_positive_user_ids = (
            torch.arange(end=batch_size, device=self.device)[None]
            .tile([max_sequence_length, 1])
            .T
        )  # (batch_size, max_seq_len)
        bpr_positive_user_ids = bpr_positive_user_ids[bpr_mask]  # (all_batch_events)

        return bpr_positive_user_ids

    def _get_graph_scores(
        self,
        graph_user_embeddings,
        bpr_positive_user_ids,
        all_final_item_embeddings,
        all_positive_sample_events,
    ):
        graph_user_embeddings = graph_user_embeddings[
            bpr_positive_user_ids
        ]  # (all_batch_events, embedding_dim)
        graph_scores = graph_user_embeddings @ all_final_item_embeddings.T
        # (all_batch_events, num_items + 2)
        graph_positive_scores = torch.gather(
            input=graph_scores, dim=1, index=all_positive_sample_events[..., None]
        )  # (all_batch_events, 1)

        return graph_positive_scores, graph_scores, graph_user_embeddings

    def _get_fusion_scores(
        self, fusion_user_embeddings, bpr_positive_user_ids, all_positive_sample_events
    ):
        fusion_user_embeddings = fusion_user_embeddings[
            bpr_positive_user_ids
        ]  # (all_batch_events, embedding_dim)
        fusion_scores = fusion_user_embeddings @ self._item_embeddings.weight.T
        # (all_batch_events, num_items + 2)
        fusion_positive_scores = torch.gather(
            input=fusion_scores, dim=1, index=all_positive_sample_events[..., None]
        )  # (all_batch_events, 1)

        return fusion_positive_scores, fusion_scores
