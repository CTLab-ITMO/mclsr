import copy
import numpy as np

from dataset.base import TrainSampler, EvalSampler, BaseSequenceDataset


class MCLSRTrainSampler(TrainSampler):
    def __init__(self, dataset, num_users, num_items, num_negatives):
        super().__init__(dataset)
        self._num_users = num_users
        self._num_items = num_items
        self._num_negatives = num_negatives

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        positive_item = sample['item.ids'][-1]

        negatives = np.random.randint(0, self._num_items + 1, (self._num_negatives,)).tolist()

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
    def __init__(self, data_dir_path, num_negatives):
        super().__init__(data_dir_path=data_dir_path, train_extended=True)
        self._num_negatives = num_negatives
    
    def get_samplers(self):
        train_sampler = MCLSRTrainSampler(
            dataset=self._train_dataset, 
            num_users=self._num_users, 
            num_items=self._num_items,
            num_negatives=self._num_negatives
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
    