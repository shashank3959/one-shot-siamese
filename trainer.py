import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.optim import SGD

import os
import pickle

from tqdm import tqdm
from utils import AverageMeter
from model import SiameseNet


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for training
    the Siamese Network model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator.
        """
        self.config = config

        # misc params
        self.use_gpu = config.use_gpu
        self.dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        # optimization params
        self.epochs = config.epochs
        self.lr = config.init_lr
        self.momentum = config.momentum

        # instantiate the model
        self.model = SiameseNet()
        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # initialize optimizer
        self.optimizer = SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum,
        )

    def train(self):
        pass

    def train_one_epoch(self, epoch):
        pass

    def save_checkpoint(self, state):
        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self):
        pass
