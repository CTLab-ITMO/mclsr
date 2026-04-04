import torch
import itertools
from irec.utils import MetaParent


class BaseBatchProcessor(metaclass=MetaParent):
    def __call__(self, batch):
        raise NotImplementedError


class IdentityBatchProcessor(BaseBatchProcessor, config_name="identity"):
    def __call__(self, batch):
        return torch.tensor(batch)


class BasicBatchProcessor(BaseBatchProcessor, config_name="basic"):
    def __call__(self, batch):
        processed_batch = {}

        for key in batch[0].keys():
            if key.endswith(".ids"):
                prefix = key.split(".")[0]
                length_key = f"{prefix}.length"
                assert length_key in batch[0]

                # --- OLD SLOW IMPLEMENTATION (Python loop with manual .extend) ---
                # processed_batch[f'{prefix}.ids'] = []
                # processed_batch[f'{prefix}.length'] = []
                # for sample in batch:
                #     processed_batch[f'{prefix}.ids'].extend(
                #         sample[f'{prefix}.ids'],
                #     )
                #     processed_batch[f'{prefix}.length'].append(
                #         sample[f'{prefix}.length'],
                #     )

                # --- NEW OPTIMIZED IMPLEMENTATION (Books-Scale Ready) ---
                """
                Optimization Strategy: C-level Flattening via itertools.
                
                Justification for Amazon Books scale:
                1. Avoiding Reallocations: Python's list.extend() repeatedly triggers 
                   memory reallocation as the list grows. For large batches on a 
                   9-million-interaction dataset, this creates significant overhead.
                2. itertools.chain.from_iterable: This is implemented in C. It creates 
                   a flat iterator over the sequence slices without creating 
                   intermediate Python list objects, which is much faster.
                3. List Comprehension: Collecting lengths via a comprehension is 
                   consistently faster than manual .append() calls in a for-loop.
                """
                # Efficiently flatten all sequence IDs into one long list
                ids_iter = itertools.chain.from_iterable(s[key] for s in batch)
                processed_batch[key] = torch.tensor(list(ids_iter), dtype=torch.long)

                # Efficiently collect all lengths into a tensor
                lengths_list = [s[length_key] for s in batch]
                processed_batch[length_key] = torch.tensor(
                    lengths_list, dtype=torch.long
                )

        # Final conversion for any keys that might have missed the .ids check
        for part, values in processed_batch.items():
            if not isinstance(processed_batch[part], torch.Tensor):
                processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch
