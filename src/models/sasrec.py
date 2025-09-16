import torch
import torch.nn as nn

from models import BaseModel

from utils import (
    DEVICE,
    create_masked_tensor,
    get_activation_function,
)


class SASRec(BaseModel):
    def __init__(
            self,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-9,
            initializer_range=0.02
    ):
        super().__init__()
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length
        self._embedding_dim = embedding_dim

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 2,
            embedding_dim=embedding_dim
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length + 1,
            embedding_dim=embedding_dim
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self._init_weights(initializer_range)

    def forward(self, inputs):
        all_sample_events = inputs['item.ids']  # (all_batch_events)
        all_sample_lengths = inputs['item.length']  # (batch_size)

        embeddings = self._item_embeddings(all_sample_events)  # (all_batch_events, embedding_dim)
        embeddings, mask = create_masked_tensor(
            data=embeddings, lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = (
            torch.arange(start=seq_len - 1, end=-1, step=-1, device=mask.device)[None]
            .tile([batch_size, 1]).long()
        )  # (batch_size, seq_len)
        positions = positions[torch.flip(mask, dims=[-1])]  # (all_batch_events)
        position_embeddings = self._position_embeddings(positions)  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)

        embeddings = embeddings + position_embeddings # (batch_size, seq_len, embedding_dim)
        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings[~mask] = 0

        causal_mask = (
            torch.tril(torch.ones(seq_len, seq_len)).bool().to(DEVICE)
        )  # (seq_len, seq_len)
        embeddings = self._encoder(
            src=embeddings,
            mask=~causal_mask,
            src_key_padding_mask=~mask,
        )  # (batch_size, seq_len, embedding_dim)

        if self.training:  # training mode
            all_positive_sample_events = inputs['positive.ids']  # (all_batch_events)

            all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)

            all_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)

            # a -- all_batch_events, n -- num_items, d -- embedding_dim
            all_scores = torch.einsum(
                'ad,nd->an',
                all_sample_embeddings,
                all_embeddings
            )  # (all_batch_events, num_items)

            positive_scores = torch.gather(
                input=all_scores,
                dim=1,
                index=all_positive_sample_events[..., None]
            )[:, 0]  # (all_batch_items)

            negative_scores = torch.gather(
                input=all_scores,
                dim=1,
                index=torch.randint(low=0, high=all_scores.shape[1], size=all_positive_sample_events.shape, device=all_positive_sample_events.device)[..., None]
            )[:, 0]  # (all_batch_items)

            return {
                'positive_scores': positive_scores,
                'negative_scores': negative_scores
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)
            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items)

            _, indices = torch.topk(
                candidate_scores,
                k=50, dim=-1, largest=True
            )  # (batch_size, 50)

            return indices
