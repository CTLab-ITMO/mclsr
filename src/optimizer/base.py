import torch


OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}


class Optimizer:
    def __init__(
        self,
        model,
        optimizer,
        clip_grad_threshold=None,
        **optimizer_params
    ):
        self._model = model
        self._optimizer = OPTIMIZERS[optimizer](model.parameters(), **optimizer_params)
        self._clip_grad_threshold = clip_grad_threshold

    def step(self, loss):
        self._optimizer.zero_grad()
        loss.backward()

        if self._clip_grad_threshold is not None:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self._clip_grad_threshold,
            )

        self._optimizer.step()
