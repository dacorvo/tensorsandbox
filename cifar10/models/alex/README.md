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
- 81% qfter 10,000 iterations.

All models are trained using data augmentation and variables moving averages.

# Alex 0

Same model, but with a reduced Kernel size and mode filters in the middle
layer.

Result: 82,15%, less parameters and processing time
