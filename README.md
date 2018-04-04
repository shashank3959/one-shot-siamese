# Siamese Networks for One-Shot Learning

**WORK IN PROGRESS**

- [ ] on-the-fly data augmentation
- [ ] currently, dataloader is inefficient. It loads the pickle dump and iterates over it. I can instead store path parameters and at each iteration, read from the folders, convert to tensors and return.
- [x] read the csv log files and plot corresponding variables.

## Paper Modifications

I've done some slight modifications to the paper to eliminate variables while I debug my code. Specifically, validation and test accuracy currently suck so I'm checking if there's a bug either in the dataset generation or trainer code.

- I'm using `Relu -> Maxpool` rather than `Maxpool - Relu`.
- I'm using batch norm between the conv layers.
- I'm using He et. al. initialization.
- I'm using a global learning rate, l2 reg factor, and momentum rather than per-layer parameters.


## Log

- I'm using `Relu -> Maxpool` rather than `Maxpool - Relu`.
- I'm trying to get decent results with a single learning rate, l2 reg factor, and momentum before using per-layer parameters like in the paper.
- Batch norm seems to have a significant effect in decreasing the loss and increasing validation accuracy right from the get-go (epoch 1)
- I've noticed the model is extremely sensitive to the learning rate: less than 1e-4 and the loss never decreases, more than 1e-3 and it explodes.
- Validation accuracy will not increase past 69%. Test accuracy is very poor (in the 45% range) which means it's not able to generalize even when using validation one-shot trials as a stopping criterion.

## Omniglot Dataset

<p align="center">
 <img src="./plots/omniglot.png" alt="Drawing", width=40%>
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
