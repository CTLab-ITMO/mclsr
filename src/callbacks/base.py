import numpy as np
import torch

from metric import StatefullMetric

import utils
from utils import DEVICE, create_logger


logger = create_logger(name=__name__)


class BaseCallback:
    def __call__(self, inputs, step_num):
        raise NotImplementedError


class MetricCallback(BaseCallback):
    def __init__(
        self,
        on_step,
        value_prefix,
    ):
        self._on_step = on_step
        self._value_prefix = value_prefix

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                f'train/{self._value_prefix}',
                inputs[self._value_prefix],
                step_num,
            )
            utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.flush()


class InferenceCallback(BaseCallback):
    def __init__(
        self,
        model,
        dataloader,
        on_step,
        metrics,
        metric_prefix,
    ):
        self._on_step = on_step
        self._model = model
        self._dataloader = dataloader
        self._metrics = metrics
        self._metric_prefix = metric_prefix

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            logger.debug(f'Running {self._metric_prefix} on step {step_num}...')

            running_metric_values = {}
            for metric_name, metric_function in self._metrics.items():
                running_metric_values[metric_name] = []

            self._model.eval()
            with torch.no_grad():
                for batch in self._dataloader:
                    for key, value in batch.items():
                        batch[key] = value.to(DEVICE)

                    batch['predicted_ids'] = self._model(batch)
                    for metric_name, metric_function in self._metrics.items():
                        running_metric_values[metric_name].extend(metric_function(inputs=batch))

            for label, value in running_metric_values.items():
                inputs[f'{self._metric_prefix}/{label}'] = np.mean(value)
                utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    f'{self._metric_prefix}/{label}',
                    inputs[f'{self._metric_prefix}/{label}'],
                    step_num,
                )

            utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.flush()
            logger.debug(f'Running {self._metric_prefix} on step {step_num} is done!')


class CompositeCallback(BaseCallback):
    def __init__(self, callbacks):
        self._callbacks = callbacks

    def __call__(self, inputs, step_num):
        for callback in self._callbacks:
            callback(inputs, step_num)
