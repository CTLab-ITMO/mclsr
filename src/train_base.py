import copy

from utils import create_logger, DEVICE

LOGGER = create_logger(name=__name__)


def train(
    dataloader,
    model,
    optimizer,
    loss_function,
    callback,
    epoch_cnt=None,
    step_cnt=None,
    epochs_threshold=40
):
    step_num = 0
    epoch_num = 0
    current_metric = 0

    best_epoch = 0
    best_checkpoint = None

    best_metric = 'validation/ndcg@20'

    LOGGER.debug('Start training...')

    while (epoch_cnt is None or epoch_num < epoch_cnt) and (
        step_cnt is None or step_num < step_cnt
    ):
        if best_epoch + epochs_threshold < epoch_num:
            LOGGER.debug(f'There is no progress during {epochs_threshold} epochs. Finish training')
            break

        LOGGER.debug(f'Start epoch {epoch_num}')
        for _, batch in enumerate(dataloader):
            batch = copy.deepcopy(batch)

            model.train()
            for key, _ in batch.items():
                batch[key] = batch[key].to(DEVICE)

            batch.update(model(batch))
            loss = loss_function(batch)

            optimizer.step(loss)
            callback(batch, step_num)
            step_num += 1

            if (
                best_checkpoint is None
                or best_metric in batch
                and current_metric <= batch[best_metric]
            ):
                # If it is the first checkpoint, or it is the best checkpoint
                current_metric = batch[best_metric]
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = epoch_num

        epoch_num += 1
    
    LOGGER.debug('Training procedure has been finished!')
    return best_checkpoint
