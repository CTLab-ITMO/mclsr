import os

from .base import BaseSequenceDataset, SequenceDataset, MCLSRDataset
from .samplers import TrainSampler, EvalSampler

class SASRecDataset(BaseSequenceDataset, config_name='sasrec_comparison'):
    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir = os.path.join(config['path_to_data_dir'], config['name'])
        max_seq_len = config.get('max_sequence_length')

        train_dataset, u1, i1, _ = SequenceDataset._create_dataset(
            dir_path=data_dir,
            part='train_sasrec',
            max_sequence_length=max_seq_len
        )

        validation_dataset, test_dataset, u_eval, i_eval = \
            MCLSRDataset._create_evaluation_sets(data_dir, max_seq_len)

        num_users = max(u1, u_eval)
        num_items = max(i1, i_eval)
        
        train_sampler = TrainSampler.create_from_config(
            config['train_sampler'],
            dataset=train_dataset, num_users=num_users, num_items=num_items
        )
        validation_sampler = EvalSampler.create_from_config(
            config['eval_sampler'],
            dataset=validation_dataset, num_users=num_users, num_items=num_items
        )
        test_sampler = EvalSampler.create_from_config(
            config['eval_sampler'],
            dataset=test_dataset, num_users=num_users, num_items=num_items
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=num_users,
            num_items=num_items,
            max_sequence_length=max_seq_len
        )
