import numpy as np
from irec.dataset.negative_samplers.base import BaseNegativeSampler
from collections import Counter


class PopularNegativeSampler(BaseNegativeSampler, config_name='popular'):
    def __init__(self, dataset, num_users, num_items):
        super().__init__(
            dataset=dataset,
            num_users=num_users,
            num_items=num_items,
        )

        # --- OLD DETERMINISTIC LOGIC ---
        # self._popular_items = self._items_by_popularity()
        
        # --- NEW STOCHASTIC LOGIC FOR LogQ COMPATIBILITY ---
        # Pre-calculate item probabilities based on global frequency
        self._item_ids, self._probs = self._calculate_item_probabilities()

    @classmethod
    def create_from_config(cls, _, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
        )

    # --- OLD METHOD: Deterministic sorting ---
    # def _items_by_popularity(self):
    #     popularity = Counter()
    #     for sample in self._dataset:
    #         for item_id in sample['item.ids']:
    #             popularity[item_id] += 1
    #     popular_items = sorted(popularity, key=popularity.get, reverse=True)
    #     return popular_items

    def _calculate_item_probabilities(self):
        """
        Calculates sampling probabilities proportional to item popularity.
        This distribution is required to provide non-zero p_j values for LogQ correction.
        """
        counts = Counter()
        for sample in self._dataset:
            for item_id in sample['item.ids']:
                counts[item_id] += 1
        
        items = np.array(list(counts.keys()))
        freqs = np.array(list(counts.values()), dtype=np.float32)
        probabilities = freqs / freqs.sum()
        
        return items, probabilities

    # --- OLD METHOD: Picking Top-K items sequentially (Deterministic) ---
    # def generate_negative_samples(self, sample, num_negatives):
    #     user_id = sample['user.ids'][0]
    #     popularity_idx = 0
    #     negatives = []
    #     while len(negatives) < num_negatives:
    #         negative_idx = self._popular_items[popularity_idx]
    #         if negative_idx not in self._seen_items[user_id]:
    #             negatives.append(negative_idx)
    #         popularity_idx += 1
    #     return negatives

    def generate_negative_samples(self, sample, num_negatives):
        """
        Stochastic sampling proportional to popularity.
        
        Justification:
        The original implementation always picked the same Top-K popular items.
        For LogQ correction (Yi et al., Google 2019), we need a stochastic 
        sampling process where p_j > 0 for all items in the distribution. 
        This allows the model to see a diverse set of negatives across epochs 
        while penalizing popular items correctly via the log(p_j) term.
        """
        user_id = sample['user.ids'][0]
        seen = self._seen_items[user_id]
        
        negatives = set()
        while len(negatives) < num_negatives:
            # Sample items based on the pre-calculated frequency distribution
            sampled_ids = np.random.choice(
                self._item_ids, 
                size=num_negatives - len(negatives), 
                p=self._probs,
                replace=True
            )
            
            # Filter out items already seen by the user (False Negatives)
            for idx in sampled_ids:
                if idx not in seen:
                    negatives.add(idx)
                    if len(negatives) == num_negatives:
                        break
                        
        return list(negatives)
