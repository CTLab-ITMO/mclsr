import torch
import torch.nn as nn


class BaseModel(nn.Module):
    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range
                    )
            elif 'bias' in key:
                nn.init.zeros_(value.data)
            else:
                raise ValueError(f'Unknown transformer weight: {key}')

    @staticmethod
    def _get_last_embedding(embeddings, mask):
        lengths = torch.sum(mask, dim=-1)
        last_item_offsets = torch.cumsum(lengths, dim=-1) - 1  # (batch_size)
        flatten_embeddings = embeddings[mask]  # (total_num_items, ...)
        last_embeddings = flatten_embeddings[last_item_offsets]  # (batch_size, ...)
        return last_embeddings

