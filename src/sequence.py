from torch import nn
import torch

from .utils import DEVICE, create_masked_tensor


class SequentialEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dim_feedforward,
        dropout,
        activation,
        layer_norm_eps,
        num_layers,
        position_embeddings,
    ):
        super().__init__()
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)
        self._position_embeddings = position_embeddings
        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

    def _pad_sequence(self, all_sample_events, all_sample_lengths):
        batch_size = all_sample_lengths.shape[0]
        max_sequence_length = all_sample_lengths.max().item()

        padded_sequence = torch.zeros(
            batch_size,
            max_sequence_length,
            dtype=torch.long,
            device=DEVICE,
        )

        mask = (
            torch.arange(end=max_sequence_length, device=DEVICE).tile(batch_size, 1)
            < all_sample_lengths[:, None]
        )  # (batch_size, max_seq_len)
        padded_sequence[mask] = all_sample_events

        return padded_sequence, mask

    def _prepare_sequence(
        self,
        seq_len,
        mask,
        batch_size,
        all_sample_lengths,
        sequence_embeddings,
    ):
        """
        add positions embeddings
        """
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
        # assert torch.allclose(position_embeddings[~mask], sequence_embeddings[~mask])

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

        return sequence_embeddings

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

    # def forward(self, inputs):
    #     all_sample_events = inputs[f"item.ids"]  # (all_batch_events)
    #     all_sample_lengths = inputs[f"item.length"]  # (batch_size)

    #     batch_size = all_sample_lengths.shape[0]
    #     max_sequence_length = all_sample_lengths.max().item()

    #     padded_sequence = torch.zeros(
    #         batch_size,
    #         max_sequence_length,
    #         dtype=torch.long,
    #         device=DEVICE,
    #     )

    #     mask = (
    #         torch.arange(end=max_sequence_length, device=DEVICE).tile(batch_size, 1)
    #         < all_sample_lengths[:, None]
    #     )  # (batch_size, max_seq_len)
    #     # print(f"{max_sequence_length=}")
    #     # print(f"{mask.shape=}")
    #     padded_sequence[mask] = all_sample_events

    #     sequence_embeddings = self._item_embeddings(
    #         padded_sequence
    #     )  # (all_batch_events, embedding_dim)
