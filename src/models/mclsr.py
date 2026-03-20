import torch
import torch.nn as nn

from utils import create_masked_tensor


from .base import BaseModel


class MCLSR(BaseModel):
    def __init__(
        self,
        num_users,
        num_items,
        max_sequence_length,
        embedding_dim,
        num_graph_layers,
        common_graph,
        user_graph,
        item_graph,
        dropout=0.0,
        layer_norm_eps=1e-5,
        graph_dropout=0.0,
        alpha=0.5,
        initializer_range=0.02,
    ):
        super().__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._num_graph_layers = num_graph_layers
        self._graph_dropout = graph_dropout
        self._alpha = alpha
        self._graph = common_graph
        self._user_graph = user_graph
        self._item_graph = item_graph

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 2,
            embedding_dim=embedding_dim,
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length + 1,
            embedding_dim=embedding_dim,
        )
        self._user_embeddings = nn.Embedding(
            num_embeddings=num_users + 2,
            embedding_dim=embedding_dim,
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        # Current interest learning
        self._current_interest_learning_encoder = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=4 * embedding_dim,
                bias=False,
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=4 * embedding_dim,
                out_features=1,
                bias=False,
            ),
        )

        # General interest learning
        self._general_interest_learning_encoder = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=False,
            ),
            nn.Tanh(),
        )

        # Cross-view contrastive learning
        self._sequential_projector = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
        )
        self._graph_projector = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
        )

        self._user_projection = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
        )

        self._item_projection = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
        )

        self._init_weights(initializer_range)

    def _apply_graph_encoder(self, embeddings, graph, use_mean=False):
        assert self.training
        # Here we use graph only in training_mode

        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        dropout_mask = torch.rand(len(values)) + self._graph_dropout
        dropout_mask = dropout_mask.int().bool()
        index = index[~dropout_mask]
        values = values[~dropout_mask] / (1.0 - self._graph_dropout)
        graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)

        all_embeddings = [embeddings]
        for _ in range(self._num_graph_layers):
            new_embeddings = torch.sparse.mm(graph_dropped, all_embeddings[-1])
            all_embeddings.append(new_embeddings)

        if use_mean:
            all_embeddings = torch.stack(all_embeddings, dim=1)
            return torch.mean(all_embeddings, dim=1)
        else:
            return all_embeddings[-1]

    def forward(self, inputs):
        all_sample_events = inputs['item.ids']  # (all_batch_events)
        all_sample_lengths = inputs['item.length']  # (batch_size)
        user_ids = inputs['user.ids']  # (batch_size)

        embeddings = self._item_embeddings(
            all_sample_events,
        )  # (all_batch_events, embedding_dim)
        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=all_sample_lengths,
        )  # (batch_size, seq_len, embedding_dim)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        # Current interest learning
        # 1) get embeddings with positions
        positions = (
            torch.arange(
                start=seq_len - 1,
                end=-1,
                step=-1,
                device=mask.device
            )[None]
            .tile([batch_size, 1])
            .long()
        )  # (batch_size, seq_len)
        # (batch_size, max_seq_len)

        positions_mask = positions < all_sample_lengths[:, None]

        positions = positions[positions_mask]  # (all_batch_events)
        position_embeddings = self._position_embeddings(
            positions
        )  # (all_batch_events, embedding_dim)

        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)

        assert torch.allclose(position_embeddings[~mask], embeddings[~mask])

        positioned_embeddings = embeddings + position_embeddings
        positioned_embeddings = self._layernorm(
            positioned_embeddings
        )  # (batch_size, seq_len, embedding_dim)

        positioned_embeddings = self._dropout(
            positioned_embeddings
        )  # (batch_size, seq_len, embedding_dim)
        positioned_embeddings[~mask] = 0

        # formula 2
        sequential_attention_matrix = self._current_interest_learning_encoder(
            positioned_embeddings,  # E_u,p
        ).squeeze()  # (batch_size, seq_len)

        sequential_attention_matrix[~mask] = -torch.inf
        sequential_attention_matrix = torch.softmax(
            sequential_attention_matrix,
            dim=1,
        )  # (batch_size, seq_len)

        # formula 3
        sequential_representation = torch.einsum(
            'bs,bsd->bd',
            sequential_attention_matrix,  # A^s
            embeddings,
        )  # (batch_size, embedding_dim)

        if self.training:
            # general interest
            # formula 4
            all_init_embeddings = torch.cat([self._user_embeddings.weight,
                                             self._item_embeddings.weight],
                                            dim=0)
            all_graph_embeddings = self._apply_graph_encoder(embeddings=all_init_embeddings,
                                                             graph=self._graph)

            common_graph_user_embs_all, common_graph_item_embs_all = torch.split(
                all_graph_embeddings, [
                    self._num_users + 2, self._num_items + 2]
            )
            common_graph_user_embs_batch = common_graph_user_embs_all[user_ids]
            common_graph_item_embs_batch, _ = create_masked_tensor(
                data=common_graph_item_embs_all[all_sample_events],
                lengths=all_sample_lengths
            )

            # formula 5: A_c = softmax(tanh(W_3 * h_u,uv) * (E_u,uv)^T)
            graph_attention_matrix = torch.einsum('bd,bsd->bs',
                                                  self._general_interest_learning_encoder
                                                  (common_graph_user_embs_batch),
                                                  common_graph_item_embs_batch)
            graph_attention_matrix[~mask] = -torch.inf
            graph_attention_matrix = torch.softmax(
                graph_attention_matrix, dim=1)

            # formula 6: I_c = A_c * E_u,uv
            original_graph_representation = torch.einsum('bs,bsd->bd',
                                                         graph_attention_matrix,
                                                         common_graph_item_embs_batch)
            original_sequential_representation = sequential_representation

            # formula 13: I_comb = alpha * I_s + (1 - alpha) * I_c
            # L_P (Downstream Loss)
            combined_representation = (self._alpha * original_sequential_representation +
                                       (1 - self._alpha) * original_graph_representation)
            labels = inputs['labels.ids']
            labels_embeddings = self._item_embeddings(labels)

            # formula 7
            # L_IL (Interest-level CL)
            sequential_representation_proj = self._sequential_projector(
                original_sequential_representation)
            graph_representation_proj = self._graph_projector(
                original_graph_representation)

            # formula 9: H_u,uu = GraphEncoder(H_u, G_uu)
            # L_UC (User-level CL)
            user_graph_user_embs_all = self._apply_graph_encoder(embeddings=self._user_embeddings.weight,
                                                                 graph=self._user_graph)
            user_graph_user_embs_batch = user_graph_user_embs_all[user_ids]

            # formula 10
            # T_f,uu = MLP(H_u,uu) и T_f,uv = MLP(H_u,uv)
            user_graph_user_embeddings_proj = self._user_projection(
                user_graph_user_embs_batch)
            common_graph_user_embeddings_proj = self._user_projection(
                common_graph_user_embs_batch)

            # item level CL
            common_graph_items_flat = common_graph_item_embs_batch[mask]

            item_graph_items_all = self._apply_graph_encoder(embeddings=self._item_embeddings.weight,
                                                             graph=self._item_graph)
            item_graph_items_flat = item_graph_items_all[all_sample_events]

            unique_item_ids, inverse_indices = torch.unique(all_sample_events,
                                                            return_inverse=True)

            def scatter_mean(src, index, dim=0, dim_size=None):
                out_size = dim_size if dim_size is not None else index.max() + 1
                out = torch.zeros((out_size, src.size(1)),
                                  dtype=src.dtype, device=src.device)
                counts = torch.bincount(
                    index, minlength=out_size).unsqueeze(-1).clamp(min=1)
                return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src) / counts

            num_unique_items = unique_item_ids.shape[0]

            unique_common_graph_items = scatter_mean(common_graph_items_flat,
                                                     inverse_indices, dim=0,
                                                     dim_size=num_unique_items)

            unique_item_graph_items = scatter_mean(item_graph_items_flat,
                                                   inverse_indices, dim=0,
                                                   dim_size=num_unique_items)

            # projection for Item-level Feature CL
            unique_common_graph_items_proj = self._item_projection(
                unique_common_graph_items)
            unique_item_graph_items_proj = self._item_projection(
                unique_item_graph_items)

            # (batch_size, num_negatives)
            negative_ids = inputs['negatives.ids']
            # (batch_size, num_negatives, embedding_dim)
            negative_embeddings = self._item_embeddings(negative_ids)

            return {
                # L_P (formula 14)
                'combined_representation': combined_representation,
                'label_representation': labels_embeddings,

                'negative_representation': negative_embeddings,

                # for L_IL (formula 8)

                'sequential_representation': sequential_representation_proj,
                'graph_representation': graph_representation_proj,

                # for L_UC (formula 11)
                'user_graph_user_embeddings': user_graph_user_embeddings_proj,
                'common_graph_user_embeddings': common_graph_user_embeddings_proj,

                # for L_IC
                'item_graph_item_embeddings': unique_item_graph_items_proj,
                'common_graph_item_embeddings': unique_common_graph_items_proj,
            }
        else:  # eval mode
            # formula 16: R(u,N) = Top-N((I_s)^T * h_o)
            candidate_embeddings = (
                self._item_embeddings.weight
            )  # (num_items + 2, embedding_dim)
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                sequential_representation,  # I_s
                candidate_embeddings,  # all h_v
            )  # (batch_size, num_items + 2)

            # Mask padding (index 0) and special tokens (index num_items+1 onwards)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            _, indices = torch.topk(
                candidate_scores,
                k=50, dim=-1, largest=True
            )  # (batch_size, 50)

            return indices