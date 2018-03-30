import torch

from utils import pickle_load
from torch.utils.data.dataset import Dataset


class Omniglot(Dataset):
    def __init__(self, data_dir, mode='train', augment=False):
        self.mode = mode
        processed = data_dir + 'processed/'

        if mode is 'train':
            x_name = "X_train"
            y_name = "y_train"
            if augment:
                x_name += '_aug'
                y_name += '_aug'
            x_name += '.p'
            y_name += '.p'
        elif mode is 'valid':
            x_name = "X_valid.p"
            y_name = "y_valid.p"
        else:
            x_name = "X_test.p"
            y_name = "y_test.p"

        self.X = pickle_load(processed + x_name)
        self.y = pickle_load(processed + y_name)

    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index]).float()
        return (img, label)

    def __len__(self):
        if self.mode is 'train':
            return 2 * len(self.X)
        elif self.mode is 'valid':
            return 2 * 14 * len(self.X)
        else:
            return 2 * 20 * len(self.X)


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid multi-process
    iterators over the Omniglot dataset.

    If using CUDA, num_workers should be set to `1` and pin_memory to `True`.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to load the augmented version of the train dataset.
    - num_workers: number of subprocesses to use when loading the dataset. Set
      to `1` if using GPU.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      `True` if using GPU.
    """
    # create train loader
    train_dataset = Omniglot(data_dir, mode='train', augment=augment)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # create valid loader
    valid_dataset = Omniglot(data_dir, mode='valid')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process iterator
    over the Omniglot test dataset.

    If using CUDA, num_workers should be set to `1` and pin_memory to `True`.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset. Set
      to `1` if using GPU.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      `True` if using GPU.
    """
    dataset = Omniglot(data_dir, mode='test')
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return loader
