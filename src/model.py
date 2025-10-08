import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from .utils import DEVICE, create_masked_tensor, get_activation_function


class MRGSRecModel(nn.Module):

    def __init__(
        self,
        num_items,
        num_users,
        max_sequence_length,
        embedding_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        graph,
        dropout=0.0,
        activation="relu",
        layer_norm_eps=1e-9,
        initializer_range=0.02,
    ):
        super().__init__()
        self._num_items = num_items
        self._num_users = num_users
        self._max_sequence_length = max_sequence_length
        self._embedding_dim = embedding_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._graph = graph
        self._dropout_rate = dropout
        self._activation = get_activation_function(activation)

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 2, embedding_dim=self._embedding_dim
        )
        self._newuser_embeddings = nn.Embedding(
            num_embeddings=1, embedding_dim=self._embedding_dim
        )
        self._item_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2, embedding_dim=self._embedding_dim
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length
            + 1,  # in order to include `max_sequence_length` value
            embedding_dim=embedding_dim,
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=self._activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self._fusion_part = nn.Sequential(
            nn.Linear(2 * embedding_dim, dim_feedforward),
            self._activation,
            nn.Linear(dim_feedforward, embedding_dim),
        )

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            num_items=kwargs["num_items"],
            num_users=kwargs["num_users"],
            max_sequence_length=kwargs["max_sequence_length"],
            embedding_dim=config["embedding_dim"],
            num_heads=config.get("num_heads", int(config["embedding_dim"] // 64)),
            num_layers=config["num_layers"],
            dim_feedforward=config.get("dim_feedforward", 4 * config["embedding_dim"]),
            graph=kwargs["graph"],
            dropout=config.get("dropout", 0.0),
            activation=config.get("activation", "relu"),
            layer_norm_eps=config.get("layer_norm_eps", 1e-9),
            initializer_range=config.get("initializer_range", 0.02),
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
            final_embeddings, lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        padded_ego_embeddings, ego_mask = create_masked_tensor(
            ego_embeddings, lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        assert torch.all(mask == ego_mask)

        return padded_embeddings, padded_ego_embeddings, mask

    def forward(self, inputs, ind=None):
        all_sample_events = inputs[f"item.ids"]  # (all_batch_events)
        all_sample_lengths = inputs[f"item.length"]  # (batch_size)

        sequence_embeddings = self._item_embeddings(
            all_sample_events
        )  # (all_batch_events, embedding_dim)
        sequence_embeddings, mask = create_masked_tensor(
            data=sequence_embeddings, lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        inputs["sequence_embeddings"] = sequence_embeddings
        inputs["mask"] = mask

        sequence_user_embeddings, sequence_embeddings = self._sequential_part(inputs)

        graph_enriched_user_embeddings, graph_enriched_item_embeddings = (
            self._apply_graph_encoder(ind)
        )  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)

        graph_user_embeddings = graph_enriched_user_embeddings[inputs["user.ids"]]

        # Fusion part
        fusion_user_embeddings = self._fuse_embeddings(
            sequence_user_embeddings, graph_user_embeddings
        )

        # print(f"{fusion_user_embeddings.shape=}")
        # print(f"{graph_enriched_item_embeddings.shape=}")
        # print(f"{graph_user_embeddings.shape=}")
        # print(f"{sequence_user_embeddings.shape=}")
        # print(f"{sequence_embeddings.shape=}")

        if self.training:  # training mode
            return self._compute_training_outputs(
                inputs,
                sequence_embeddings,
                graph_user_embeddings,
                fusion_user_embeddings,
                graph_enriched_item_embeddings,
            )
        else:
            return self._compute_inference_outputs(fusion_user_embeddings)

    def _apply_graph_encoder(self, ind=None):
        ego_embeddings = torch.cat(
            (self._user_embeddings.weight, self._item_embeddings.weight), dim=0
        )
        if ind is not None:
            ego_embeddings = torch.cat(
                (
                    ego_embeddings[:ind],
                    self._newuser_embeddings.weight,
                    ego_embeddings[ind + 1 :],
                ),
                dim=0,
            )
        all_embeddings = [ego_embeddings]

        if self._dropout_rate > 0:  # drop some edges
            if self.training:  # training_mode
                size = self._graph.size()
                index = self._graph.indices().t()
                values = self._graph.values()
                random_index = torch.rand(len(values)) + (1 - self._dropout_rate)
                random_index = random_index.int().bool()
                index = index[random_index]
                values = values[random_index] / (1 - self._dropout_rate)
                graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
            else:  # eval mode
                graph_dropped = self._graph
        else:
            graph_dropped = self._graph

        for i in range(1):
            ego_embeddings = torch.sparse.mm(graph_dropped, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.mean(torch.stack(all_embeddings, dim=-1), dim=-1)
        user_final_embeddings, item_final_embeddings = torch.split(
            all_embeddings, [self._num_users + 2, self._num_items + 2]
        )

        return user_final_embeddings, item_final_embeddings

    def _sequential_part(self, inputs):
        mask = inputs["mask"]
        batch_size = mask.shape[0]
        seq_len = mask.shape[1]
        all_sample_lengths = inputs[f"item.length"]
        user_ids = inputs[f"user.ids"]
        sequence_embeddings = inputs["sequence_embeddings"]

        # START:Sequential part
        sequence_embeddings = self._prepare_sequence_embeddings(
            seq_len, mask, batch_size, all_sample_lengths, user_ids, sequence_embeddings
        )

        sequence_embeddings = self._encode_sequence(
            seq_len, batch_size, mask, sequence_embeddings
        )

        sequence_user_embeddings, sequence_embeddings = (
            sequence_embeddings[:, 0, :],
            sequence_embeddings[:, 1:, :],
        )  # (batch_size, embedding_dim), (batch_size, seq_len, embedding_dim)
        # END:Sequential part

        return (
            sequence_user_embeddings,
            sequence_embeddings,
        )

    def _encode_sequence(self, seq_len, batch_size, mask, sequence_embeddings):
        causal_mask = (
            torch.tril(torch.ones(seq_len, seq_len)).bool().to(DEVICE)
        )  # (seq_len, seq_len)
        advanced_mask = torch.ones(
            seq_len + 1, seq_len + 1, dtype=torch.bool, device=DEVICE
        )  # (seq_len + 1, seq_len + 1)
        advanced_mask[1:, 1:] = causal_mask
        advanced_src_key_padding_mask = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=DEVICE), mask], dim=1
        )  # (batch_size, seq_len + 1)
        sequence_embeddings = self._encoder(
            src=sequence_embeddings,
            mask=~advanced_mask,
            src_key_padding_mask=~advanced_src_key_padding_mask,
        )  # (batch_size, seq_len + 1, embedding_dim)

        return sequence_embeddings

    def _prepare_sequence_embeddings(
        self,
        seq_len,
        mask,
        batch_size,
        all_sample_lengths,
        user_ids,
        sequence_embeddings,
    ):
        positions = (
            torch.arange(start=seq_len - 1, end=-1, step=-1, device=mask.device)[None]
            .tile([batch_size, 1])
            .long()
        )  # (batch_size, seq_len)
        positions_mask = (
            positions < all_sample_lengths[:, None]
        )  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)
        position_embeddings = self._position_embeddings(
            positions
        )  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings, lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)
        assert torch.allclose(position_embeddings[~mask], sequence_embeddings[~mask])

        sequence_embeddings = (
            sequence_embeddings + position_embeddings
        )  # (batch_size, seq_len, embedding_dim)
        sequence_embeddings = self._layernorm(
            sequence_embeddings
        )  # (batch_size, seq_len, embedding_dim)
        sequence_embeddings = self._dropout(
            sequence_embeddings
        )  # (batch_size, seq_len, embedding_dim)
        sequence_embeddings[~mask] = 0

        sequence_user_embeddings = self._user_embeddings(user_ids).unsqueeze(
            1
        )  # (batch_size, 1, embedding_dim)

        sequence_embeddings = torch.cat(
            [sequence_user_embeddings, sequence_embeddings], dim=1
        )  # (batch_size, seq_len + 1, embedding_dim)

        return sequence_embeddings

    def _fuse_embeddings(self, sequence_user_embeddings, graph_user_embeddings):
        return self._fusion_part(
            torch.cat(
                [
                    sequence_user_embeddings,
                    0.1 * sequence_user_embeddings + 0.9 * graph_user_embeddings,
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
            candidate_scores, k=20, dim=-1, largest=True
        )  # (batch_size, 20)

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

        all_positive_sample_events = inputs[f"positive.ids"]  # (all_batch_events)
        all_positive_sample_lengths = inputs[f"positive.length"]  # (batch_size)

        bpr_positive_user_ids = self._get_bpr_positive_user_ids(
            max_sequence_length, batch_size, all_positive_sample_lengths
        )

        # Sequential part
        all_sample_sequence_embeddings = sequence_embeddings[
            mask
        ]  # (all_batch_events, embedding_dim)

        # TODO: test with original and tune L_c coef
        sequence_scores = all_sample_sequence_embeddings @ all_final_item_embeddings.T
        # (all_batch_events, num_items + 2)

        graph_positive_scores, graph_scores, graph_user_embeddings = (
            self._get_graph_scores(
                graph_user_embeddings,
                bpr_positive_user_ids,
                all_final_item_embeddings,
                all_positive_sample_events,
            )
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
            torch.arange(end=max_sequence_length, device=DEVICE)[None].tile(
                [batch_size, 1]
            )
            < all_positive_sample_lengths[:, None]
        )  # (batch_size, max_seq_len)
        bpr_positive_user_ids = (
            torch.arange(end=batch_size, device=DEVICE)[None]
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
