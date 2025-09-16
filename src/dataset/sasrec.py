import copy
import os

from dataset.base import TrainSampler, EvalSampler
from dataset.base import BaseSequenceDataset


class SASRecTrainSampler(TrainSampler):
    def __init__(
        self,
        dataset,
        num_users,
        num_items,
    ):
        super().__init__(dataset=dataset)
        self._num_users = num_users
        self._num_items = num_items

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        next_item_sequence = sample['item.ids'][1:]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'positive.ids': next_item_sequence,
            'positive.length': len(next_item_sequence),
        }


class SASRecEvalSampler(EvalSampler):
    pass


class SASRecDataset(BaseSequenceDataset):
    def get_samplers(self):
        train_sampler = SASRecTrainSampler(
            dataset=self._train_dataset, 
            num_users=self._num_users, 
            num_items=self._num_items
        )
        valid_sampler = SASRecEvalSampler(
            dataset=self._valid_dataset,
            num_users=self._num_users,
            num_items=self._num_items
        )
        test_sampler = SASRecEvalSampler(
            dataset=self._test_dataset,
            num_users=self._num_users,
            num_items=self._num_items
        )
        return train_sampler, valid_sampler, test_sampler

        
