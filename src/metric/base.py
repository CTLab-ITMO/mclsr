import torch

from utils import create_masked_tensor


class BaseMetric:
    pass


class StatefullMetric(BaseMetric):
    def reduce(self):
        raise NotImplementedError


class CompositeMetric(BaseMetric):
    def __init__(self, metrics):
        self._metrics = metrics

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


class CoverageMetric(StatefullMetric):
    def __init__(self, k, num_items):
        self._k = k
        self._num_items = num_items

    def __call__(self, inputs):
        predictions = inputs['predicted_ids'][:, :self._k].float()  # (batch_size, k)
        return predictions.flatten().long().tolist() # (batch_size * k)

    def reduce(self, values):
        return len(set(values)) / self._num_items


class NDCGMetric(BaseMetric):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs):
        predictions = inputs['predicted_ids'][:, :self._k].float()  # (batch_size, k)
        labels_flat = inputs['target.ids']  # (batch_size)
        labels_lengths = inputs['target.length']  # (batch_size)
        assert predictions.shape[0] == labels_lengths.shape[0]

        batch_size = predictions.shape[0]

        padded_labels, labels_mask = create_masked_tensor(data=labels_flat, lengths=labels_lengths)
        padded_labels[~labels_mask] = -1

        positions = torch.arange(2, self._k + 2, device=predictions.device)
        weights = 1. / torch.log2(positions.float())

        is_hit = (predictions[:, :, None] == padded_labels[:, None, :]).sum(dim=-1)  # (batch_size, k)

        num_ideal_hits = torch.minimum(labels_lengths, torch.as_tensor(self._k, device=labels_lengths.device, dtype=labels_lengths.dtype))  # (batch_size)
        ideal_mask = (torch.arange(self._k, device=is_hit.device, dtype=weights.dtype)[None, :].tile(dims=[batch_size, 1]) < num_ideal_hits[:, None])  # (batch_size, k)
        
        dcg = (is_hit.float() * weights).sum(dim=-1)  # (batch_size)
        idcg = (ideal_mask.float() * weights).sum(dim=-1)  # (batch_size)

        ndcg = dcg / idcg

        return ndcg.tolist()


class RecallMetric(BaseMetric):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs):
        predictions = inputs['predicted_ids'][:, :self._k] # (batch_size, k)
        labels_flat = inputs['target.ids']  # (batch_size)
        labels_lengths = inputs['target.length'] # (batch_size)

        assert predictions.shape[0] == labels_lengths.shape[0]

        padded_labels, labels_mask = create_masked_tensor(data=labels_flat, lengths=labels_lengths)
        padded_labels[~labels_mask] = -1

        is_hit = (predictions[:, :, None] == padded_labels[:, None, :]).sum(dim=-1).float()  # (batch_size, k)
        recall = is_hit.sum(dim=-1) / torch.minimum(labels_lengths, torch.as_tensor(self._k, device=labels_lengths.device, dtype=labels_lengths.dtype))  # (batch_size)
        
        return recall.tolist()


class HitRateMetric(BaseMetric):
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs):
        predictions = inputs['predicted_ids'][:, :self._k] # (batch_size, k)
        labels_flat = inputs['target.ids']  # (all_targets)
        labels_lengths = inputs['target.length'] # (batch_size)

        assert predictions.shape[0] == labels_lengths.shape[0]

        padded_labels, labels_mask = create_masked_tensor(data=labels_flat, lengths=labels_lengths)
        padded_labels[~labels_mask] = -1
        
        hit_rate = (predictions[:, :, None] == padded_labels[:, None, :]).sum(dim=-1).max(dim=-1).values.float()  # (batch_size)
        
        return hit_rate.tolist()