import utils
from utils import (
    create_logger,
    DEVICE,
    fix_random_seed,
    ensure_checkpoints_dir,
)

from callbacks import CompositeCallback, InferenceCallback, MetricCallback
from dataset import MCLSRDataset, GraphDataset
from metric import HitRateMetric, NDCGMetric, RecallMetric
from models import MCLSR
from optimizer import Optimizer
from loss import CompositeLoss, SamplesSoftmaxLoss, FpsLoss

import copy
import json
import os
import torch


from torch.utils.data import DataLoader

from dataloader import batch_processor
from train_base import train

LOGGER = create_logger(name=__name__)
SEED_VAL = 42
EXPERIMENT_NAME = 'mclsr_Clothing'

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
    
    dataset = MCLSRDataset(data_dir_path='./data/Clothing', num_negatives=1280)
    graph_dataset = GraphDataset(
        dataset=dataset,
        graph_dir_path='./data/Clothing',
        use_user_graph=True,
        use_item_graph=True,
        neighborhood_size=50
    )
    train_sampler, valid_sampler, test_sampler = graph_dataset.get_samplers()

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
    model = MCLSR(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        max_sequence_length=20,
        embedding_dim=EMBEDDING_DIM,
        num_graph_layers=NUM_LAYERS,
        common_graph=graph_dataset.graph,
        user_graph=graph_dataset.user_graph,
        item_graph=graph_dataset.item_graph,
        dropout=DROPOUT,
        layer_norm_eps=1e-9,
        graph_dropout=DROPOUT,
    ).to(DEVICE)

    loss = CompositeLoss(
        losses=[
            SamplesSoftmaxLoss(
                queries_prefix="combined_representation",
                positive_prefix="label_representation",
                negative_prefix="negative_representation",
                output_prefix="downstream_loss",
            ),

            FpsLoss(
                fst_embeddings_prefix="sequential_representation",
                snd_embeddings_prefix="graph_representation",
                tau=0.5,
                normalize_embeddings=True,
                use_mean=True,
                output_prefix="contrastive_interest_loss",
            ),

            FpsLoss(
                fst_embeddings_prefix="user_graph_user_embeddings",
                snd_embeddings_prefix="common_graph_user_embeddings",
                tau=0.5,
                normalize_embeddings=True,
                use_mean=True,
                output_prefix="contrastive_user_feature_loss",
            ),

            FpsLoss(
                fst_embeddings_prefix="item_graph_item_embeddings",
                snd_embeddings_prefix="common_graph_item_embeddings",
                tau=0.5,
                normalize_embeddings=True,
                use_mean=True,
                output_prefix="contrastive_item_feature_loss",
            ),
        ],
    weights=[1.0, 1.0, 0.05, 0.05],
    output_prefix="loss",
)


    optimizer = Optimizer(
        model=model,
        optimizer=OPTIMIZER,
        clip_grad_threshold=CLIP_GRAD_THRESHOLD,
        lr=0.001
    )
    
    callback = CompositeCallback(
        callbacks=[
            MetricCallback(on_step=1, value_prefix='loss'),
            MetricCallback(on_step=1, value_prefix='downstream_loss'),
            MetricCallback(on_step=1, value_prefix='contrastive_interest_loss'),
            MetricCallback(on_step=1, value_prefix='contrastive_user_feature_loss'),
            MetricCallback(on_step=1, value_prefix='contrastive_item_feature_loss'),
            InferenceCallback(
                model=model,
                dataloader=valid_dataloader,
                on_step=64,
                metrics={
                    f'{metric_name}@{k}': metric_cls(k=k)
                    for metric_cls, metric_name in [(HitRateMetric, 'hit'), (NDCGMetric, 'ndcg'), (RecallMetric, 'recall')] 
                    for k in [5, 10, 20, 50]
                },
                metric_prefix='validation'
            ),
            InferenceCallback(
                model=model,
                dataloader=test_dataloader,
                on_step=256,
                metrics={
                    f'{metric_name}@{k}': metric_cls(k=k)
                    for metric_cls, metric_name in [(HitRateMetric, 'hit'), (NDCGMetric, 'ndcg'), (RecallMetric, 'recall')] 
                    for k in [5, 10, 20, 50]
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
