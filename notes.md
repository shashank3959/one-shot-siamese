one shot learning: learning a class from a single labeled example.

disadvantages of typical (ConvNet + Softmax) approach:

* if database increases (add a new class), must retrain the network
* extremely data hungry:
    * need crap ton of data of every class to work well
    * collecting and labelling data is expensive
    * potential overfit if dataset is not large enough

## Losses

We can use different types of losses to train networks for one-shot learning:

- contrastive loss: decreases the energy of like pairs and increases the energy of unlike pairs.
- triplet loss: minimizes the distance between an anchor and a positive, both of which have the same identity, and maximizes the distance between the anchor and a negative of a different identity.

The triplet loss optimizes the embedding space such that data points with the same identity are closer to each other than those with different identities. We use it as follows:
    - use the Euclidean distance to measure similarity of extracted features from 2 images
    - replace the Euclidean distance with a learned metric. For example, we can use a fc-layer with a 1D output to learn this metric thanks to backprop. Note that we need to make sure that the output is constrained to the range [0, 1]. We can simply threshold this, or play around with the output of the fc layer and maybe add a softmax layer to its output.
We then plug in one of the above metrics in the formula of the triplet loss.

The contrastive loss function works on input pairs. We compute a parametrized distance function of the inputs (i.e. they are fed through the siamese net), specifically, we take the Euclidean distance of of the parametrized outputs. We then feed the scalar result to the contrastive loss which pulls similar neighbors together and pushes non-neighbors apart. Given a set of high-dimensional inputs with a neighborhood relationship (some are similar, some are dissimilar), the goal of the contrastive loss is to learn a parametrize mapping that maps the high-dimensional inputs to a lower-dimensional manifold such that the Euclidean distance between the points on the manifold approximates the semantic similarity of inputs in the input space.

The problem of mapping a set of high-dimensional points onto a low dimensional manifold has a long history. The classical method is PCA (principal component analysis) which involves the projection of inputs to a low dimensional subspace that maximizes the variance. PCA, unfortunately, is a linear embedding.

## Some Ideas

- insert STN to increase effectiveness of convnet?
- use distillation to decrease footprint of finalized network
    - FitNet
    - adversarial distillation

## Data Splitting

Omniglot consists in 1623 different handwritten characters (from 50 alphabets). Each character is drawn 20 times and stored as a 105x105 grayscale image. We are given a background set and an evaluation set.

- background set: contains 40 alphabets. We use this set, with normal train, valid, test splits to tune the model for verification.
- evaluation set: contains 10 alphabets. We use this set to measure the one-shot classification performance.

**TODO: add K-shot, N-way classification explanation**

The authors consider all 50 alphabets together.

- train: 30/50 (60%) of the alphabets, 12 of the 20 drawers. Uniform number of training examples per alphabet so that each alphabet receives equal representation. Hallucinate 8 affine transformations for each training example. 3 datasets of sizes 30k, 90k, 150k. With the hallucinations, these sizes get multiplied by 9: 270k, 810k, 1,350k.
- validation: 10k example pairs from 10/50 (20%) alphabets and 4 drawers. Generate a set of 320 one-shot recognition trials and use it as a termination criterion.
- testing: 10/50 (20%) alphabets, 4 drawers.

## Variations on the Paper

- relu placed after maxpool to reduce the # of computation by 75%
- weight init strategy possibly changed to he et al since it uses relu activation
