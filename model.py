import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    """
    "Siamese Neural Networks for One-shot Image Recognition" [1].

    Siamese networts learn image representations via a supervised metric-based
    approach. Once tuned, their learned features can be leveraged for one-shot
    learning without any retraining.

    References
    ----------
    - Koch et. al., https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    """
    def __init__(self):
        super(SiameseNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 1)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, 0, 1e-2)
                nn.init.normal(m.bias, 0.5, 1e-2)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 2e-1)
                nn.init.normal(m.weight, 0, 1e-2)

    def sub_forward(self, x):
        """
        Forward pass the input image through 1 subnetwork.

        Args
        ----
        - x: a Variable of size (B, C, H, W). Contains either the first or
        second image pair across the input batch.

        Returns
        -------
        - out: a Variable of size (B, 4096). An encoded representation of
        """
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = F.relu(F.max_pool2d(self.conv3(out), 2))
        out = F.relu(self.conv4(out), 2)
        out = out.view(out.shape[0], -1)
        out = F.sigmoid(self.fc1(out))
        return out

    def forward(self, x1, x2):
        """
        Forward pass the input image through both subtwins.

        Concretely, we take the encoded vector representations of both subnets
        and compute their L1 component-wise distance. We then feed the
        difference to a last fc layer followed by a sigmoid activation.
        """
        # encode image pairs
        h1 = self.forward_twin(x1)
        h2 = self.forward_twin(x2)

        # compute l1 distance
        diff = torch.abs(h1 - h2)

        # score the similarity between the 2 encodings
        out = F.sigmoid(self.fc2(diff))

        return out
