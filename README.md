# Siamese Networks for One-Shot Learning

<p align="center">
 <img src="./plots/loss.png" alt="Drawing", width=80%>
</p>

## Paper Modifications

I've done some slight modifications to the paper to eliminate variables while I debug my code. Specifically, validation and test accuracy currently suck so I'm checking if there's a bug either in the dataset generation or trainer code.

- I'm using `Relu -> Maxpool` rather than `Maxpool - Relu`.
- I'm using batch norm between the conv layers.
- I'm using He et. al. initialization.
- I'm using a global learning rate, l2 reg factor, and momentum rather than per-layer parameters.

## Omniglot Dataset

<p align="center">
 <img src="./plots/omniglot.png" alt="Drawing", width=60%>
</p>

Execute the following commands

* Download the data using `run.sh`
    * `chmod +x run.sh`
    * `./run.sh`
* Process the data using `data_prep.ipynb`

Then, you can load the dataset using:

```python
from data_loader import *

# batch size of 32 with data augmentation
train_loader, valid_loader = get_train_valid_loader(data_dir, 32, True)

for idx, (x, y) in enumerate(train_loader):
    # do something
```

Checkout [Playground.ipynb](https://github.com/kevinzakka/siamese-network/blob/master/Playground.ipynb) for a minimal working example.
