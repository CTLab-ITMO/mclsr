import copy

from irec.utils import (
    MetaParent,
    maybe_to_list,
)

import torch
import torch.nn as nn


import pickle
import os
import logging

logger = logging.getLogger(__name__)

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


class BatchLogSoftmaxLoss(TorchLoss, config_name='batch_logsoftmax'):
    def __init__(self, predictions_prefix, candidates_prefix):
        super().__init__()
        self._predictions_prefix = predictions_prefix
        self._candidates_prefix = candidates_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            predictions_prefix=config.get('predictions_prefix'),
            candidates_prefix=config.get('candidates_prefix'),
        )

    def forward(self, inputs):  # use log soft max
        predictions = inputs[self._predictions_prefix]
        candidates = inputs[self._candidates_prefix]

        dot_product_matrix = predictions @ candidates.T

        row_log_softmax = nn.LogSoftmax(dim=1)
        softmax_matrix = -row_log_softmax(dot_product_matrix)

        diagonal_elements = torch.diag(softmax_matrix)

        loss = diagonal_elements.mean()

        return loss


class CrossEntropyLoss(TorchLoss, config_name='ce'):
    def __init__(self, predictions_prefix, labels_prefix, output_prefix=None):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix
        self._output_prefix = output_prefix

        self._loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix]  # (all_items, num_classes)
        all_labels = inputs[
            '{}.ids'.format(self._labels_prefix)
        ]  # (all_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)  # (1)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class BinaryCrossEntropyLoss(TorchLoss, config_name='bce'):
    def __init__(
        self,
        predictions_prefix,
        labels_prefix,
        with_logits=True,
        output_prefix=None,
    ):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix
        self._output_prefix = output_prefix

        if with_logits:
            self._loss = nn.BCEWithLogitsLoss()
        else:
            self._loss = nn.BCELoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix].float()  # (all_batch_items)
        all_labels = inputs[self._labels_prefix].float()  # (all_batch_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)  # (1)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class BPRLoss(TorchLoss, config_name='bpr'):
    def __init__(self, positive_prefix, negative_prefix, output_prefix=None):
        super().__init__()
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        pos_scores = inputs[self._positive_prefix]  # (all_batch_items)
        neg_scores = inputs[self._negative_prefix]  # (all_batch_items)
        loss = -torch.log(
            (pos_scores - neg_scores).sigmoid() + 1e-9,
        ).mean()  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class RegularizationLoss(TorchLoss, config_name='regularization'):
    def __init__(self, prefix, output_prefix=None):
        super().__init__()
        self._prefix = maybe_to_list(prefix)
        self._output_prefix = output_prefix

    def forward(self, inputs):
        loss = 0.0
        for prefix in self._prefix:
            loss += (1 / 2) * inputs[prefix].pow(2).mean()

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class FpsLoss(TorchLoss, config_name='fps'):
    def __init__(
        self,
        fst_embeddings_prefix,
        snd_embeddings_prefix,
        tau,
        normalize_embeddings=False,
        use_mean=True,
        output_prefix=None,
        use_logq_correction=False,
        logq_prefix=None,
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
        self._use_logq_correction = use_logq_correction
        self._logq_prefix = logq_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            fst_embeddings_prefix=config['fst_embeddings_prefix'],
            snd_embeddings_prefix=config['snd_embeddings_prefix'],
            tau=config.get('temperature', 1.0), 
            normalize_embeddings=config.get('normalize_embeddings', False),
            use_mean=config.get('use_mean', True),
            output_prefix=config.get('output_prefix'),
            use_logq_correction=config.get('use_logq_correction', False),
            logq_prefix=config.get('logq_prefix', None),
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

        if self._use_logq_correction and self._logq_prefix is not None:
            log_q = inputs[self._logq_prefix]
            log_q_combined = torch.cat((log_q, log_q), dim=0)
            
            log_q_matrix = log_q_combined.unsqueeze(0).expand(2 * batch_size, -1)  # (2B, 2B)
            negative_log_q = log_q_matrix[mask].reshape(2 * batch_size, -1)  # (2B, 2B-2)

            negative_samples = negative_samples - negative_log_q

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

    
# sasrec logq debug

class LogqSamplesSoftmaxLoss(TorchLoss, config_name='logq_sampled_softmax'):
    def __init__(
        self,
        queries_prefix,
        positive_prefix,
        negative_prefix,
        positive_ids_prefix=None,
        negative_ids_prefix=None,
        output_prefix=None,
        use_logq_correction=False,
        logq_prefix=None,
        log_counts=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix

        self._positive_ids_prefix = positive_ids_prefix 
        self._negative_ids_prefix = negative_ids_prefix

        self._output_prefix = output_prefix
        self._use_logq = use_logq_correction
        self._logq_prefix = logq_prefix
        self._log_counts = log_counts

    @classmethod
    def create_from_config(cls, config, **kwargs):
        log_counts = None
        path_to_counts = config.get('path_to_item_counts')
        
        if path_to_counts and config.get('use_logq_correction'):
            import pickle
            with open(path_to_counts, 'rb') as f:
                counts = pickle.load(f)
            
            counts_tensor = torch.tensor(counts, dtype=torch.float32)
            # Normalize in probability and use logarithm (Google Eq. 3)
            probs = torch.clamp(counts_tensor / counts_tensor.sum(), min=1e-10)
            log_counts = torch.log(probs)
            logger.info(f"Loaded item counts from {path_to_counts} for LogQ correction")

        return cls(
            queries_prefix=config['queries_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            positive_ids_prefix=config.get('positive_ids_prefix'),
            negative_ids_prefix=config.get('negative_ids_prefix'),
            output_prefix=config.get('output_prefix'),
            use_logq_correction=config.get('use_logq_correction', False),
            logq_prefix=config.get('logq_prefix'),
            log_counts=log_counts  # <-- ПЕРЕДАЕМ В КОНСТРУКТОР
        )

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

        # --- FALSE NEGATIVE MASKING (Critical for In-Batch Negatives) ---
        # If we have item IDs, we must ensure that a positive item for a user 
        # is not treated as a negative for that same user if it appears 
        # elsewhere in the batch.
        if self._positive_ids_prefix and self._negative_ids_prefix:
            pos_ids = inputs[self._positive_ids_prefix]  # (BatchSize,)
            neg_ids = inputs[self._negative_ids_prefix]  # (NumNegatives,)

            # Create a boolean mask of shape (BatchSize, NumNegatives)
            # where True indicates that pos_ids[i] == neg_ids[j]
            false_negative_mask = (pos_ids.unsqueeze(1) == neg_ids.unsqueeze(0))
            
            # Mask out these scores by setting them to a very large negative value
            # This prevents the model from receiving contradictory signals 
            # (trying to both increase and decrease the score of the same item).
            negative_scores = negative_scores.masked_fill(false_negative_mask, -1e12)

        # --- 2. UNBIASED LOGQ CORRECTION ---
        # Applying correction to EACH logit per Google Paper (Eq. 3)
        if self._use_logq:
            # Source of truth: our pre-loaded self._log_counts from the pickle
            if self._log_counts is not None:
                if self._log_counts.device != positive_scores.device:
                    self._log_counts = self._log_counts.to(positive_scores.device)
                
                # We need IDs to fetch the correct frequencies for items in this batch
                pos_ids = inputs[self._positive_ids_prefix]
                neg_ids = inputs[self._negative_ids_prefix]
                
                log_q_pos = self._log_counts[pos_ids].unsqueeze(-1)  # (B, 1)
                log_q_neg = self._log_counts[neg_ids]               # (N,) or (B, N)
                
                # --- LOGQ CORRECTION COMMENTS ---
                # According to "Sampling-Bias-Corrected Neural Modeling..." (Google, 2019):
                # "we correct EACH logit s(x_i, y_j) by the following equation: 
                # s_c(x_i, y_j) = s(x_i, y_j) - log(p_j)"
                # This ensures the estimator remains unbiased by penalizing popular 
                # items equally when they are targets and when they are negatives.
                
                positive_scores = positive_scores - log_q_pos
                negative_scores = negative_scores - log_q_neg
            
            # (Optional) If frequencies were passed directly in inputs, not via pickle:
            elif self._logq_prefix in inputs:
                log_q = inputs[self._logq_prefix]
                positive_scores = positive_scores - log_q[:, :1]
                negative_scores = negative_scores - log_q[:, 1:]

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

class MCLSRLogqLoss(TorchLoss, config_name='mclsr_logq_special'):
    """
    Specialized Loss for MCLSR model implementing Sampling-Bias Correction (LogQ) 
    with a tunable lambda coefficient for correction strength.
    
    Theory:
    Following Yi et al. (2019), we correct the sampling bias introduced by non-uniform 
    negative sampling (e.g., popularity-based or in-batch sampling).
    
    Mathematical Formula (Eq. 3 in the paper):
        s_c(x, y) = s(x, y) - lambda * log(p_j)
    where p_j is the sampling probability of item j and lambda is the correction strength.
    """
    def __init__(
        self,
        queries_prefix,
        positive_prefix,
        negative_prefix,
        positive_ids_prefix,
        negative_ids_prefix,
        path_to_item_counts,
        logq_lambda=1.0,  # Strength of the LogQ correction (usually 0.1 to 1.0)
        output_prefix=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._positive_ids_prefix = positive_ids_prefix
        self._negative_ids_prefix = negative_ids_prefix
        self._output_prefix = output_prefix
        self._logq_lambda = logq_lambda

        if not os.path.exists(path_to_item_counts):
            raise FileNotFoundError(f"Item counts file not found at {path_to_item_counts}")

        with open(path_to_item_counts, 'rb') as f:
            counts = pickle.load(f)
        
        counts_tensor = torch.tensor(counts, dtype=torch.float32)
        
        # Calculate log probabilities log(p_j). 
        # Using clamp(min=1e-10) for numerical stability to avoid log(0) -> NaN.
        probs = torch.clamp(counts_tensor / counts_tensor.sum(), min=1e-10)
        log_q = torch.log(probs)
        
        # register_buffer ensures the lookup table moves to the correct device (GPU/CPU) 
        # along with the model parameters.
        self.register_buffer('_log_q_table', log_q)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        """Standard framework factory method to initialize the loss from a JSON config."""
        return cls(
            queries_prefix=config['queries_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            positive_ids_prefix=config['positive_ids_prefix'],
            negative_ids_prefix=config['negative_ids_prefix'],
            path_to_item_counts=config['path_to_item_counts'],
            logq_lambda=config.get('logq_lambda', 1.0),
            output_prefix=config.get('output_prefix')
        )

    def forward(self, inputs):
        # Retrieve Embeddings
        queries = inputs[self._queries_prefix]           # (Batch, Dim)
        pos_embs = inputs[self._positive_prefix]         # (Batch, Dim)
        neg_embs = inputs[self._negative_prefix]         # (B, N, D) or (N, D)
        
        # Retrieve Item IDs for correction and collision masking
        pos_ids = inputs[self._positive_ids_prefix]      # (Batch)
        neg_ids = inputs[self._negative_ids_prefix]      # (Batch, N) or (N)

        # --- DEVICE SYNCHRONIZATION ---
        # Explicitly ensure the LogQ lookup table is on the same device as the inputs.
        # This prevents "indices should be on the same device" RuntimeError on CUDA.
        device = queries.device
        if self._log_q_table.device != device:
            self._log_q_table = self._log_q_table.to(device)

        # --- STEP 1: Score Calculation (Inner Product) ---
        # Positive scores: (Batch, 1)
        pos_scores = torch.einsum('bd,bd->b', queries, pos_embs).unsqueeze(-1)
        
        # Negative scores with dimension check (handles both shared and per-user negatives)
        if neg_embs.dim() == 2:
            # Case: Shared negatives across the batch (N, D) -> Result: (B, N)
            neg_scores = torch.einsum('bd,nd->bn', queries, neg_embs)
        else:
            # Case: Individual negatives per user (B, N, D) -> Result: (B, N)
            neg_scores = torch.einsum('bd,bnd->bn', queries, neg_embs)

        # --- STEP 2: False Negative Masking ---
        # Prevent contradictory gradients by masking negative samples that match 
        # the ground truth item ID for a given user.
        # logic: pos_ids.unsqueeze(1) creates (B, 1) for broadcasting against (B, N) or (1, N)
        neg_id_comparison = neg_ids.unsqueeze(0) if neg_ids.dim() == 1 else neg_ids
        false_negative_mask = (pos_ids.unsqueeze(1) == neg_id_comparison)
        
        # Mask out scores of false negatives by setting them to a large negative value
        neg_scores = neg_scores.masked_fill(false_negative_mask, -1e12)

        # --- STEP 3: Tunable LogQ Correction ---
        # Applying the bias correction as per Google Paper Eq. 3, scaled by lambda.
        # s_c = s - lambda * log(Q)
        log_q_pos = self._log_q_table[pos_ids].unsqueeze(-1) # (B, 1)
        
        if neg_ids.dim() == 1:
            # Case: Shared negative IDs
            log_q_neg = self._log_q_table[neg_ids].unsqueeze(0) # (1, N)
        else:
            # Case: Individual negative IDs
            log_q_neg = self._log_q_table[neg_ids]              # (B, N)

        # Apply final correction
        pos_scores = pos_scores - (self._logq_lambda * log_q_pos)
        neg_scores = neg_scores - (self._logq_lambda * log_q_neg)

        # --- STEP 4: Softmax Loss Calculation ---
        # Concatenate positive and negative scores: (B, 1 + N)
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        
        # Cross-Entropy using log_softmax for numerical stability
        loss = -torch.log_softmax(all_scores, dim=1)[:, 0]
        
        final_loss = loss.mean()
        if self._output_prefix:
            inputs[self._output_prefix] = final_loss.cpu().item()
            
        return final_loss

class S3RecPretrainLoss(TorchLoss, config_name='s3rec_pretrain'):
    def __init__(
        self,
        positive_prefix,
        negative_prefix,
        representation_prefix,
        output_prefix=None,
    ):
        super().__init__()
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._representation_prefix = representation_prefix
        self._criterion = nn.BCEWithLogitsLoss(reduction='none')
        self._output_prefix = output_prefix

    def forward(self, inputs):
        positive_embeddings = inputs[
            self._positive_prefix
        ]  # (x, embedding_dim)
        negative_embeddings = inputs[
            self._negative_prefix
        ]  # (x, embedding_dim)
        current_embeddings = inputs[
            self._representation_prefix
        ]  # (x, embedding_dim)
        assert (
            positive_embeddings.shape[0]
            == negative_embeddings.shape[0]
            == current_embeddings.shape[0]
        )

        positive_scores = torch.einsum(
            'bd,bd->b',
            positive_embeddings,
            current_embeddings,
        )  # (x)

        negative_scores = torch.einsum(
            'bd,bd->b',
            negative_embeddings,
            current_embeddings,
        )  # (x)

        distance = torch.sigmoid(positive_scores) - torch.sigmoid(
            negative_scores,
        )  # (x)
        loss = torch.sum(
            self._criterion(
                distance,
                torch.ones_like(distance, dtype=torch.float32),
            ),
        )  # (1)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class Cl4sRecLoss(TorchLoss, config_name='cl4srec'):
    def __init__(
        self,
        current_representation,
        all_items_representation,
        tau=1.0,
        output_prefix=None,
    ):
        super().__init__()
        self._current_representation = current_representation
        self._all_items_representation = all_items_representation
        self._loss_function = nn.CrossEntropyLoss()
        self._tau = tau
        self._output_prefix = output_prefix

    def forward(self, inputs):
        current_representation = inputs[
            self._current_representation
        ]  # (batch_size, embedding_dim)
        all_items_representation = inputs[
            self._all_items_representation
        ]  # (batch_size, num_negatives + 1, embedding_dim)

        batch_size = current_representation.shape[0]

        logits = torch.einsum(
            'bnd,bd->bn',
            all_items_representation,
            current_representation,
        )  # (batch_size, num_negatives + 1)
        labels = logits.new_zeros(batch_size)  # (batch_size)

        loss = self._loss_function(logits, labels)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class DuorecSSLLoss(TorchLoss, config_name='duorec_ssl'):
    def __init__(
        self,
        original_embedding_prefix,
        dropout_embedding_prefix,
        similar_embedding_prefix,
        normalize_embeddings=False,
        tau=1.0,
        output_prefix=None,
    ):
        super().__init__()
        self._original_embedding_prefix = original_embedding_prefix
        self._dropout_embedding_prefix = dropout_embedding_prefix
        self._similar_embedding_prefix = similar_embedding_prefix
        self._normalize_embeddings = normalize_embeddings
        self._output_prefix = output_prefix
        self._tau = tau
        self._loss_function = nn.CrossEntropyLoss(reduction='mean')

    def _compute_partial_loss(self, fst_embeddings, snd_embeddings):
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
            )

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

        # TODO optimize
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

        return loss

    def forward(self, inputs):
        original_embeddings = inputs[
            self._original_embedding_prefix
        ]  # (x, embedding_dim)
        dropout_embeddings = inputs[
            self._dropout_embedding_prefix
        ]  # (x, embedding_dim)
        similar_embeddings = inputs[
            self._similar_embedding_prefix
        ]  # (x, embedding_dim)

        dropout_loss = self._compute_partial_loss(
            original_embeddings,
            dropout_embeddings,
        )
        ssl_loss = self._compute_partial_loss(
            original_embeddings,
            similar_embeddings,
        )

        loss = dropout_loss + ssl_loss

        if self._output_prefix is not None:
            inputs[f'{self._output_prefix}_dropout'] = (
                dropout_loss.cpu().item()
            )
            inputs[f'{self._output_prefix}_ssl'] = ssl_loss.cpu().item()
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
