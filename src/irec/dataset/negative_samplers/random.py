from irec.dataset.negative_samplers.base import BaseNegativeSampler

import numpy as np


class RandomNegativeSampler(BaseNegativeSampler, config_name='random'):
    @classmethod
    def create_from_config(cls, _, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
        )

    # def generate_negative_samples(self, sample, num_negatives):
    #     user_id = sample['user.ids'][0]
    #     all_items = list(range(1, self._num_items + 1))
    #     np.random.shuffle(all_items)

    #     negatives = []
    #     running_idx = 0
    #     while len(negatives) < num_negatives and running_idx < len(all_items):
    #         negative_idx = all_items[running_idx]
    #         if negative_idx not in self._seen_items[user_id]:
    #             negatives.append(negative_idx)
    #         running_idx += 1

    #     return negatives

    def generate_negative_samples(self, sample, num_negatives):
        """
        Optimized via Rejection Sampling (O(k) complexity).
        
        Mathematical Proof of Equivalence:
        Let V be the set of all items and H be the user's history. 
        We need a uniform random sample S ⊂ (V \ H) such that |S| = k.
        
        1. Shuffle Approach (Previous): Generates a random permutation of V, 
           then filters H. Complexity: O(|V|).
        2. Rejection Sampling (Current): Independently draws i ~ Uniform(V) 
           and accepts i if i ∉ H and i ∉ S. Complexity: O(k * 1/p), 
           where p = (|V| - |H|) / |V|.
        
        Since |H| << |V|, the probability p ≈ 1, making the expected complexity 
        effectively O(k). Both methods yield an identical uniform distribution 
        over the valid item space.
        """
        user_id = sample['user.ids'][0]
        seen = self._seen_items[user_id]
        
        negatives = set()
        while len(negatives) < num_negatives:
            # Drawing a random index is O(1)
            negative_idx = np.random.randint(1, self._num_items + 1)
            if negative_idx not in seen:
                negatives.add(negative_idx)

        return list(negatives)
