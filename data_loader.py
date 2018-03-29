import torch
import pickle

from torchvision import transforms
from torch.utils.data.dataset import Dataset


class Omniglot(Dataset):
    def __init__(self, data_dir, train=True, transform=False):
        self.train = train
        self.transform = transform

        # dir and file names
        processed = data_dir + 'processed/'
        x_name = "X_train.p" if train else "X_valid.p"
        y_name = "y_train.p" if train else "y_valid.p"

        # unpickle imgs and gd truth labels
        self.X = pickle.load(open(processed + x_name, "rb"))
        self.y = pickle.load(open(processed + y_name, "rb"))

    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index])
        return (img, label)

    def __len__(self):
        return 2 * len(self.X)
