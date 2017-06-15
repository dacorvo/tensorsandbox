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
@10,000:  55,8%
@50,000:  71,8%
@100,000: 77,5%
@150,000: 78,9%

The network is tiny, and the processing time is equivalent to the tutorial
version, but the accuracy is not as good.

Size : 0.12 Millions of parameters
Flops: 37.13 Millions of operations
