import torch
from torch.utils.data import DataLoader

from callbacks import CompositeCallback, InferenceCallback, MetricCallback
from dataloader import batch_processor
from dataset import SASRecDataset
from loss import SASRecLoss
from metric import HitRateMetric, NDCGMetric, RecallMetric
from models import SASRec
from optimizer import Optimizer

import utils
from utils import (
    create_logger,
    DEVICE,
    fix_random_seed,
    ensure_checkpoints_dir,
)

from train_base import train


LOGGER = create_logger(name=__name__)
SEED_VAL = 42
EXPERIMENT_NAME = 'sasrec_clothing_all_data_their'

TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
EMBEDDING_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.3
OPTIMIZER = 'adam'
CLIP_GRAD_THRESHOLD = 1.0
TRAIN_EPOCH_NUM = 100


def main():
    fix_random_seed(SEED_VAL)

    tensorboard_writer = utils.tensorboards.TensorboardWriter(EXPERIMENT_NAME)
    utils.tensorboards.GLOBAL_TENSORBOARD_WRITER = tensorboard_writer
    
    dataset = SASRecDataset(data_dir_path='./data/Clothing')
    train_sampler, valid_sampler, test_sampler = dataset.get_samplers()

    train_dataloader = DataLoader(
        dataset=train_sampler,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=batch_processor,
    )
    valid_dataloader = DataLoader(
        dataset=valid_sampler,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=batch_processor,
    )
    test_dataloader = DataLoader(
        dataset=test_sampler,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=batch_processor,
    )

    model = SASRec(
        num_items=dataset.num_items + 2,
        max_sequence_length=20,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        activation='gelu',
        layer_norm_eps=1e-9
    ).to(DEVICE)

    loss = SASRecLoss(output_prefix='loss')

    optimizer = Optimizer(
        model=model,
        optimizer=OPTIMIZER,
        clip_grad_threshold=CLIP_GRAD_THRESHOLD,
        lr=0.001
    )
    
    callback = CompositeCallback(
        callbacks=[
            MetricCallback(
                on_step=1,
                value_prefix='loss'
            ),
            InferenceCallback(
                model=model,
                dataloader=valid_dataloader,
                on_step=64,
                metrics={
                    f'{metric_name}@{k}': metric_cls(k=k)
                    for metric_cls, metric_name in [
                        (HitRateMetric, 'hit'), (NDCGMetric, 'ndcg'), (RecallMetric, 'recall')
                    ] for k in [5, 10, 20, 50]
                },
                metric_prefix='validation'
            ),
            InferenceCallback(
                model=model,
                dataloader=test_dataloader,
                on_step=256,
                metrics={
                    f'{metric_name}@{k}': metric_cls(k=k)
                    for metric_cls, metric_name in [
                        (HitRateMetric, 'hit'), (NDCGMetric, 'ndcg'), (RecallMetric, 'recall')
                    ] for k in [5, 10, 20, 50]
                },
                metric_prefix='eval'
            )
        ]
    )

    LOGGER.debug('Everything is ready for training process!')

    # Train process
    best_model = train(
        dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss,
        callback=callback,
    )

    LOGGER.debug('Saving model...')
    ensure_checkpoints_dir()
    checkpoint_path = f'./checkpoints/{EXPERIMENT_NAME}_best_state.pth'
    torch.save(best_model.state_dict(), checkpoint_path)
    LOGGER.debug(f'Model saved as {checkpoint_path}')


if __name__ == '__main__':
    main()
