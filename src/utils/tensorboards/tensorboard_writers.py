import os
import time
import datetime

from torch.utils.tensorboard import SummaryWriter

LOGS_DIR = './tensorboard_logs'
GLOBAL_TENSORBOARD_WRITER = None


class TensorboardWriter(SummaryWriter):
    def __init__(self, experiment_name, use_time=True):
        self._experiment_name = experiment_name
        super().__init__(
            log_dir=os.path.join(
                LOGS_DIR,
                f'{experiment_name}_{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M" if use_time else "")}',
            ),
        )

    def add_scalar(self, *args, **kwargs):
        super().add_scalar(*args, **kwargs)
