import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import os
import time
import pickle

from tqdm import tqdm
from model import SiameseNet
from utils import AverageMeter, get_num_model


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for training
    the Siamese Network model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader, layer_hyperparams):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator.
        - layer_hyperparams: dict containing layer-wise hyperparameters
          such as the initial learning rate, the end momentum, and the l2
          regularization strength.
        """
        self.config = config
        self.layer_hyperparams = layer_hyperparams

        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[0]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)

        self.model = SiameseNet()
        if config.use_gpu:
            self.model.cuda()

        # model params
        self.num_params = sum(
            [p.data.nelement() for p in self.model.parameters()]
        )
        self.num_model = get_num_model(config)
        self.num_layers = len(list(self.model.children()))

        print('[*] Number of model parameters: {:,}'.format(self.num_params))

        # path params
        self.ckpt_dir = os.path.join(config.ckpt_dir, self.num_model)
        self.logs_dir = os.path.join(config.logs_dir, self.num_model)

        # misc params
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.use_gpu = config.use_gpu
        self.dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        # optimization params
        self.best_valid_acc = 0.
        self.epochs = config.epochs
        self.start_epoch = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience

        # grab layer-wise hyperparams
        self.init_lrs = self.layer_hyperparams['layer_init_lrs']
        self.init_momentums = [config.init_momentum]*self.num_layers
        self.end_momentums = self.layer_hyperparams['layer_end_momentums']
        self.l2_regs = self.layer_hyperparams['layer_l2_regs']

        # compute temper rate for momentum
        if self.epochs == 1:
            f = lambda max, min: min
        else:
            f = lambda max, min: (max - min) / (self.epochs-1)
        self.momentum_temper_rates = [
            f(x, y) for x,y in zip(self.end_momentums, self.init_momentums)
        ]

        # set global learning rates and momentums
        self.lrs = self.init_lrs
        self.momentums = self.init_momentums

        # initialize optimizer
        optim_dict = []
        for i, layer in enumerate(self.model.children()):
            group = {}
            group['params'] = layer.parameters()
            group['lr'] = self.lrs[i]
            group['momentum'] = self.momentums[i]
            group['weight_decay'] = self.l2_regs[i]
            optim_dict.append(group)
        self.optimizer = optim.SGD(optim_dict)

        # learning rate scheduler
        self.scheduler = StepLR(
            self.optimizer, step_size=self.lr_patience, gamma=0.99,
        )

    def train(self):
        if self.resume:
            self.load_checkpoint(best=False)

        # switch to train mode
        self.model.train()

        # create train and validation log files
        optim_file = open(os.path.join(self.logs_dir, 'optim.csv'), 'w')
        train_file = open(os.path.join(self.logs_dir, 'train.csv'), 'w')
        valid_file = open(os.path.join(self.logs_dir, 'valid.csv'), 'w')

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):
            self.decay_lr()
            self.temper_momentum(epoch)

            # log lrs and momentums
            optim_file.write('{},{},{}'.format(
                epoch, *self.momentums, *self.lrs)
            )

            print('\nEpoch: {}/{}'.format(epoch+1, self.epochs))

            train_loss = self.train_one_epoch(epoch, train_file)
            valid_acc = self.validate(epoch, valid_file)

            # check for improvement
            is_best = valid_acc > self.best_valid_acc
            msg = "train loss: {:.3f} - train acc: {:.3f} "
            msg += "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                msg += " [*]"
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            # checkpoint the model
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optim.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                }, is_best
            )

        # release resources
        optim_file.close()
        train_file.close()
        valid_file.close()

    def train_one_epoch(self, epoch, file):
        train_batch_time = AverageMeter()
        train_losses = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()
                x, y, = Variable(x), Variable(y)

                # split input pairs along the batch dimension
                batch_size = x.shape[0]
                x1, x2 = x[:, 0], x[:, 1]

                out = self.model(x1, x2)
                loss = F.binary_cross_entropy_with_logits(out, y)

                # compute gradients and update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # store batch statistics
                toc = time.time()
                train_losses.update(loss.data[0], batch_size)
                train_batch_time.update(toc-tic)
                tic = time.time()

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f}".format(
                            train_batch_time.val,
                            train_losses.val,
                        )
                    )
                )
                pbar.update(batch_size)

                # log loss and acc
                iter = (epoch * len(self.train_loader)) + i
                file.write('{},{}\n'.format(
                    iter, train_losses.val)
                )

            return train_losses.avg

    def validate(self, epoch, file):
        valid_batch_time = AverageMeter()
        valid_accs = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        tic = time.time()
        with tqdm(total=self.num_valid) as pbar:
            correct = 0
            for i, (x, y) in enumerate(self.valid_loader):
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x, volatile=True), Variable(y, volatile=True)

                batch_size = x.shape[0]
                x = x.squeeze(dim=0)
                y = y.squeeze(dim=0)

                # split input pairs along the batch dimension
                x1, x2 = x[:, 0], x[:, 1]

                # compute log probabilities
                out = self.model(x1, x2)
                log_probas = F.sigmoid(out)

                # get index of max log prob
                pred = log_probas.data.max(0)[1]
                correct += pred.eq(y.long().data).long().cpu()
                acc = (100. * correct) / (i+1)

                # store batch statistics
                toc = time.time()
                valid_accs.update(acc)
                valid_batch_time.update(toc-tic)
                tic = time.time()

                pbar.set_description(
                    (
                        "{:.1f}s - acc: {:.3f}".format(
                            valid_batch_time.val,
                            valid_accs.val,
                        )
                    )
                )
                pbar.update(batch_size)

                # log loss and acc
                iter = (epoch * len(self.valid_loader)) + i
                file.write('{},{}\n'.format(
                    iter, valid_accs.val)
                )

            return valid_accs.avg

    def test(self):
        pass

    def temper_momentum(self, epoch):
        """
        This function linearly increases the per-layer momentum to
        a predefined ceiling over a set amount of epochs.
        """
        if epoch == 0:
            return
        self.momentums = [
            x + y for x,y in zip(self.momentums, self.momentum_temper_rates)
        ]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['momentum'] = self.momentums[i]

    def decay_lr(self):
        """
        This function linearly decays the per-layer lr over a set
        amount of epochs.
        """
        self.scheduler.step()
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.lrs[i] = param_group['lr']

    def save_checkpoint(self, state, is_best):
        filename = 'model_ckpt.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = 'best_model_ckpt.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = 'model_ckpt.tar'
        if best:
            filename = 'best_model_ckpt.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch']+1, ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch']+1)
            )
