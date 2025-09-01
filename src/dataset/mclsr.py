import os

from dataset.base import TrainSampler, EvalSampler, BaseSequenceDataset

from collections import defaultdict
import random


class MCLSRTrainSampler(TrainSampler):
    def __init__(self, dataset, num_users, num_items):
        super().__init__(dataset)
        self._num_users = num_users
        self._num_items = num_items
        # self._num_negatives = num_negatives
        # self._all_items_set = set(range(1, num_items + 1))
        # self._user_to_all_seen_items = user_to_all_seen_items

    def __getitem__(self, index):
        sample = self._dataset[index]

        item_sequence = sample['item.ids'][:-1]
        positive_item = sample['item.ids'][-1]

        # user_seen = self._user_to_all_seen_items[user_id]
        # unseen_items = list(self._all_items_set - user_seen)
        # negatives = random.sample(unseen_items, self._num_negatives)
        
        negatives = [random.randint(1, self._num_items) for _ in range(self._num_negatives)]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'labels.ids': [positive_item],
            'labels.length': 1,

            'negatives.ids': negatives,
            'negatives.length': len(negatives),
        }


class MCLSREvalSampler(EvalSampler):    
    pass


class MCLSRDataset(BaseSequenceDataset):
    def get_samplers(self):
        train_sampler = MCLSRTrainSampler(
            dataset=self._train_dataset, 
            num_users=self._num_users, 
            num_items=self._num_items
        )
        valid_sampler = MCLSREvalSampler(
            dataset=self._valid_dataset, 
            num_users=self._num_users, 
            num_items=self._num_items
        )
        test_sampler = MCLSREvalSampler(
            dataset=self._test_dataset, 
            num_users=self._num_users, 
            num_items=self._num_items
        )
        return train_sampler, valid_sampler, test_sampler
    