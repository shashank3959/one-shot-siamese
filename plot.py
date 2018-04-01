import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logs_dir = './logs/exp_1/'
plot_dir = './plots/'


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def main():
    filename = logs_dir + 'train.csv'
    df = pd.read_csv(filename, header=None, names=['iter', 'loss'])
    losses = df['loss'].data

    fig, ax = plt.subplots(figsize=(15, 8))
    rolling_mean = np.mean(rolling_window(losses, 50), 1)
    rolling_std = np.std(rolling_window(losses, 50), 1)
    plt.plot(range(len(rolling_mean)), rolling_mean, alpha=0.98, linewidth=0.9)
    plt.fill_between(
        range(len(rolling_std)),
        rolling_mean-rolling_std,
        rolling_mean+rolling_std,
        alpha=0.5
    )
    plt.title('Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_dir + 'train_loss_exp1.png', format='png', dpi=100)


if __name__ == '__main__':
    main()
