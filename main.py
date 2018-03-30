import torch
import random

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config, load_config
from data_loader import get_train_valid_loader, get_test_loader


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
        data_loader = get_train_valid_loader(
            config.data_dir, config.batch_size,
            config.augment, **kwargs
        )
    else:
        data_loader = get_test_loader(
            config.data_dir, config.batch_size, **kwargs
        )

    # sample 3 layer wise hyperparams if firs time training
    if config.is_train and not config.resume:
        layer_hyperparams = {
            'layer_init_lrs': [],
            'layer_end_momentums': [],
            'layer_l2_regs': []
        }
        for i in range(6):
            # sample
            lr = random.uniform(1e-4, 1e-1)
            mom = random.uniform(0, 1)
            reg = random.uniform(0, 0.1)

            # store
            layer_hyperparams['layer_init_lrs'].append(lr)
            layer_hyperparams['layer_end_momentums'].append(mom)
            layer_hyperparams['layer_l2_regs'].append(reg)

        # save
        save_config(config, layer_hyperparams)
    # else load it from config file
    else:
        print("[*] Loaded layer hyperparameters")
        layer_hyperparams = load_config(config)

    # instantiate trainer
    trainer = Trainer(config, data_loader, layer_hyperparams)

    # either train
    if config.is_train:
        pass
        # trainer.train()
    # or load a pretrained model and test
    else:
        pass
        # trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
