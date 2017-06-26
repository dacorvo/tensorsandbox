# Alex-like variants of the Tensorflow CIFAR10 tutorial

In this directory, we test several modified versions of the tutorial model.

These models retain the same overall structure inspired by Alexei Krizhevsky
initial model: a few conv nets and pooling, then a few FC nets.

All these models also share some common features that have been tested on the
tutorial model itself as improving training and/or accuracy:

- pooling kernels have a size of 3, but a stride of 2,
- biases for the first and last layers are initialized with zeroes,
- biases for the other layers are initialized with 0.1 values,
- weight decay is applied only on FC nets

The goal is to reach at least the same accuracy as the tutorial model:

- 81,0% after 10,000 iterations,
- 85,5% after 50,000 iterations,
- 86,0% after 100,000 iterations.

All models are trained using data augmentation and variables moving averages.

# Alex 0

Same model, but with a reduced Kernel size and more filters in the middle
layer.

Results:

~~~~
@10,000:  82,2%
@50,000:  85,7%
@100,000: 86,6%
~~~~

More parameters but less processing cost.

~~~~
Size : 1.46 Millions of parameters
Flops: 25.24 Millions of operations
~~~~

# Alex 1

Same model, but replace first FC layer with a 5x5x64x64 conv layer.

Results:

~~~~
@10,000:  81,8%
@50,000:  85,1%
@100,000: 85,9%
~~~~

Much less parameters but a slightly higher processing cost.

~~~~
Size : 0.32 Millions of parameters
Flops: 43.51 Millions of operations
~~~~

# Alex 2

Same model, but without local response normalization.

Results:

~~~~
@10,000:  81,4%
@10,000:  85,3%
@100,000: 86,1%
~~~~

Huge decrease in training time despite equivalent computation cost.
This may be related to the lrn implementation (or my bogus calculations).

~~~~
Size : 1.07 Millions of parameters
Flops: 37.08 Millions of operations
~~~~

# Alex 3

Same model, but with one FC layer removed.

Results:

~~~~
@10,000:  81,2%
@50,000:  84,8%
@100,000: 85,5%
~~~~

same processing time.

~~~~
Size : 1.00 Millions of parameters
Flops: 37.60 Millions of operations
~~~~

# Alex 4

We combine here the two best variants, ie Alex0 + Alex2 (modified middle
convnet topology, without lrn normalization)
Since we don't use normalization, we can further increase the number of
filters without inducing too much processing.

Results:

~~~~
@10,000:  81,2%
@50,000:  86,0%
@100,000: 86,4%
@300,000: 87,5%
~~~~

Again, we see a lower training time for an equivalent processing cost, which
is odd. Also, inference time seems a bit higher.

~~~~
Size : 1.49 Millions of parameters
Flops: 35.20 Millions of operations
~~~~
