import torch


def batch_processor(batch):
    processed_batch = {}

    for key in batch[0].keys():
        if key.endswith('.ids'):
            prefix = key.split('.')[0]
            assert '{}.length'.format(prefix) in batch[0]

            processed_batch[f'{prefix}.ids'] = []
            processed_batch[f'{prefix}.length'] = []

            for sample in batch:
                processed_batch[f'{prefix}.ids'].extend(
                    sample[f'{prefix}.ids'],
                )
                processed_batch[f'{prefix}.length'].append(
                    sample[f'{prefix}.length'],
                )

    for part, values in processed_batch.items():
        processed_batch[part] = torch.tensor(values, dtype=torch.long)

    return processed_batch