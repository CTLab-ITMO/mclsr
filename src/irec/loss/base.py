import copy

from irec.utils import (
    MetaParent,
    maybe_to_list,
)

import torch
import torch.nn as nn


class BaseLoss(metaclass=MetaParent):
    pass


class TorchLoss(BaseLoss, nn.Module):
    pass


class IdentityLoss(BaseLoss, config_name='identity'):
    def __call__(self, inputs):
        return inputs


class CompositeLoss(TorchLoss, config_name='composite'):
    def __init__(self, losses, weights=None, output_prefix=None):
        super().__init__()
        self._losses = losses
        self._weights = weights or [1.0] * len(losses)
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        losses = []
        weights = []

        for loss_cfg in copy.deepcopy(config)['losses']:
            weight = loss_cfg.pop('weight') if 'weight' in loss_cfg else 1.0
            loss_function = BaseLoss.create_from_config(loss_cfg)

            weights.append(weight)
            losses.append(loss_function)

        return cls(
            losses=losses,
            weights=weights,
            output_prefix=config.get('output_prefix'),
        )

    def forward(self, inputs):
        total_loss = 0.0
        for loss, weight in zip(self._losses, self._weights):
            total_loss += weight * loss(inputs)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = total_loss.cpu().item()

        return total_loss


class FpsLoss(TorchLoss, config_name='fps'):
    def __init__(
        self,
        fst_embeddings_prefix,
        snd_embeddings_prefix,
        tau,
        normalize_embeddings=False,
        use_mean=True,
        output_prefix=None,
    ):
        super().__init__()
        self._fst_embeddings_prefix = fst_embeddings_prefix
        self._snd_embeddings_prefix = snd_embeddings_prefix
        self._tau = tau
        self._loss_function = nn.CrossEntropyLoss(
            reduction='mean' if use_mean else 'sum',
        )
        self._normalize_embeddings = normalize_embeddings
        self._output_prefix = output_prefix
        print(self._tau)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            fst_embeddings_prefix=config['fst_embeddings_prefix'],
            snd_embeddings_prefix=config['snd_embeddings_prefix'],
            tau=config.get('temperature', 1.0), 
            normalize_embeddings=config.get('normalize_embeddings', False),
            use_mean=config.get('use_mean', True),
            output_prefix=config.get('output_prefix')
        )

    def forward(self, inputs):
        fst_embeddings = inputs[
            self._fst_embeddings_prefix
        ]  # (x, embedding_dim)
        snd_embeddings = inputs[
            self._snd_embeddings_prefix
        ]  # (x, embedding_dim)

        batch_size = fst_embeddings.shape[0]

        combined_embeddings = torch.cat(
            (fst_embeddings, snd_embeddings),
            dim=0,
        )  # (2 * x, embedding_dim)

        if self._normalize_embeddings:
            combined_embeddings = torch.nn.functional.normalize(
                combined_embeddings,
                p=2,
                dim=-1,
                eps=1e-6,
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
        ).reshape(2 * batch_size, 1)  # (2 * x, 1)
        assert torch.allclose(
            torch.diag(similarity_scores, batch_size),
            torch.diag(similarity_scores, -batch_size),
        )

        mask = torch.ones(
            2 * batch_size,
            2 * batch_size,
            dtype=torch.bool,
        )  # (2 * x, 2 * x)
        mask = mask.fill_diagonal_(0)  # Remove equal embeddings scores
        for i in range(batch_size):  # Remove positives
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        negative_samples = similarity_scores[mask].reshape(
            2 * batch_size,
            -1,
        )  # (2 * x, 2 * x - 2)

        labels = (
            torch.zeros(2 * batch_size).to(positive_samples.device).long()
        )  # (2 * x)
        logits = torch.cat(
            (positive_samples, negative_samples),
            dim=1,
        )  # (2 * x, 2 * x - 1)

        loss = self._loss_function(logits, labels) / 2  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class SASRecLoss(TorchLoss, config_name='sasrec'):

    def __init__(
            self,
            positive_prefix,
            negative_prefix,
            output_prefix=None
    ):
        super().__init__()
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        positive_scores = inputs[self._positive_prefix]  # (x)
        negative_scores = inputs[self._negative_prefix]  # (x)
        assert positive_scores.shape[0] == negative_scores.shape[0]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_scores, torch.ones_like(positive_scores)
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            negative_scores, torch.zeros_like(negative_scores)
        )

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class SamplesSoftmaxLoss(TorchLoss, config_name='sampled_softmax'):
    def __init__(
        self,
        queries_prefix,
        positive_prefix,
        negative_prefix,
        output_prefix=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        queries_embeddings = inputs[
            self._queries_prefix
        ]  # (batch_size, embedding_dim)
        positive_embeddings = inputs[
            self._positive_prefix
        ]  # (batch_size, embedding_dim)
        negative_embeddings = inputs[
            self._negative_prefix
        ]  # (num_negatives, embedding_dim) or (batch_size, num_negatives, embedding_dim)

        # b -- batch_size, d -- embedding_dim
        positive_scores = torch.einsum(
            'bd,bd->b',
            queries_embeddings,
            positive_embeddings,
        ).unsqueeze(-1)  # (batch_size, 1)

        if negative_embeddings.dim() == 2:  # (num_negatives, embedding_dim)
            # b -- batch_size, n -- num_negatives, d -- embedding_dim
            negative_scores = torch.einsum(
                'bd,nd->bn',
                queries_embeddings,
                negative_embeddings,
            )  # (batch_size, num_negatives)
        else:
            assert (
                negative_embeddings.dim() == 3
            )  # (batch_size, num_negatives, embedding_dim)
            # b -- batch_size, n -- num_negatives, d -- embedding_dim
            negative_scores = torch.einsum(
                'bd,bnd->bn',
                queries_embeddings,
                negative_embeddings,
            )  # (batch_size, num_negatives)
        all_scores = torch.cat(
            [positive_scores, negative_scores],
            dim=1,
        )  # (batch_size, 1 + num_negatives)

        logits = torch.log_softmax(
            all_scores,
            dim=1,
        )  # (batch_size, 1 + num_negatives)
        loss = (-logits)[:, 0]  # (batch_size)
        loss = loss.mean()  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class MCLSRLoss(TorchLoss, config_name='mclsr'):
    def __init__(
        self,
        all_scores_prefix,
        mask_prefix,
        normalize_embeddings=False,
        tau=1.0,
        output_prefix=None,
    ):
        super().__init__()
        self._all_scores_prefix = all_scores_prefix
        self._mask_prefix = mask_prefix
        self._normalize_embeddings = normalize_embeddings
        self._output_prefix = output_prefix
        self._tau = tau

    def forward(self, inputs):
        all_scores = inputs[
            self._all_scores_prefix
        ]  # (batch_size, batch_size, seq_len)
        mask = inputs[self._mask_prefix]  # (batch_size)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positive_mask = torch.eye(batch_size, device=mask.device).bool()

        positive_scores = all_scores[positive_mask]  # (batch_size, seq_len)
        negative_scores = torch.reshape(
            all_scores[~positive_mask],
            shape=(batch_size, batch_size - 1, seq_len),
        )  # (batch_size, batch_size - 1, seq_len)
        assert torch.allclose(all_scores[0, 1], negative_scores[0, 0])
        assert torch.allclose(all_scores[-1, -2], negative_scores[-1, -1])
        assert torch.allclose(all_scores[0, 0], positive_scores[0])
        assert torch.allclose(all_scores[-1, -1], positive_scores[-1])

        # Maybe try mean over sequence TODO
        loss = torch.sum(
            torch.log(
                torch.sigmoid(positive_scores.unsqueeze(1) - negative_scores),
            ),
        )  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss
