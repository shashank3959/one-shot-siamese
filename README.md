# siamese-network

**WORK IN PROGRESS**

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
