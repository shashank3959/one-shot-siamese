import torch

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from data_loader import *


def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    # instantiate data loaders
    if config.is_train:
        data_loader = get_train_valid_loader()
    else:
        data_loader = get_test_loader()

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
