# Experiment with SqueezeNets

[SqueezeNet](https://arxiv.org/abs/1602.07360) is a compact convolutional net
that stacks layers of 'fire module'. Each fire module combines a 1x1 convolution
to 'squeeze' the input and mixed 1x1 and 3x3 convolutions to expand the output.

The model is trained with data augmentation and moving average of variables.

## Squeeze 0

conv [3x3x3x64]
pool x2
fire 16/64/64
fire 16/64/64
pool x2
fire 32/128/128
fire 32/128/128
conv [1,1,128*2,10]
average pool

Results:

@10,000:  63,9%
@50,000:  82,3%
@100,000: 84,7%

The network is tiny, and the processing cost is much lower than the tutorial
version, but the accuracy is a bit lower.

Size : 0.12 Millions of parameters
Flops: 16.04 Millions of operations

## Squeeze 1

conv [3x3x3x64]
pool x2
fire 32/64/64
fire 32/64/64
pool x2
fire 32/128/128
fire 32/128/128
conv [1,1,128*2,10]
average pool

Results:

@10,000  : 66,0%
@50,000  : 83,6%
@100,000 : 85,7%
@300,000 : 87,8%

Following a hint in the paper, we reduce the squeeze ratio of the first layer.
Accuracy is improved a bit, but processing time rises.

Size : 0.15 Millions of parameters
Flops: 22.84 Millions of operations

