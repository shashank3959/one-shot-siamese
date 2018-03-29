import torch

from utils import pickle_load
from torch.utils.data.dataset import Dataset


class Omniglot(Dataset):
    def __init__(self, data_dir, train=True, augment=False):
        processed = data_dir + 'processed/'

        x_name = "X_train.p" if train else "X_valid.p"
        y_name = "y_train.p" if train else "y_valid.p"

        if augment:
            x_name = "X_train_aug.p"
            y_name = "y_train_aug.p"

        self.X = pickle_load(processed + x_name)
        self.y = pickle_load(processed + y_name)

    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index])
        return (img, label)

    def __len__(self):
        return 2 * len(self.X)


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           num_workers,
                           pin_memory):
    """
    Utility function for loading and returning train and valid multi-process
    iterator over the Omniglot dataset.

    If using CUDA, num_workers should be set to `1` and pin_memory to `True`.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to load the augmented Omniglot dataset.
    - num_workers: number of subprocesses to use when loading the dataset. Set
      to `1` if using GPU.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      `True` if using GPU.
    """
    # create train loader
    train_dataset = Omniglot(data_dir, True, augment)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # create valid loader
    valid_dataset = Omniglot(data_dir, False, not augment)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)
