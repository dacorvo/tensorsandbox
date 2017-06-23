#!/bin/bash
#
# This script just uses the label_image binary from the Tensorflow examples to
# perform an inference on one of our pre-trained CIFAR10 classification model
#
# Note that the results may not be as accurate as those obtained with the
# python eval script, as the label_image binary standardizes all images using
# the same global mean and variance values, but the models were trained using
# images normalized independently.
# As a workaround, medium values of 128 are used for global mean and variance.
#

SCRIPTPATH=$(readlink -f ${0})
SCRIPTDIR=$(dirname ${SCRIPTPATH})

model=${1:-'tuto'}
image=${2:-"${SCRIPTDIR}/samples/cars/251.png"}

TF_BUILD_PATH=/tensorflow/bazel-bin
LABEL_IMAGE_BIN=${TF_BUILD_PATH}/tensorflow/examples/label_image/label_image
IMAGE_HEIGHT=24
IMAGE_WIDTH=24

GRAPH=${SCRIPTDIR}/trained_models/${model}.pb

${LABEL_IMAGE_BIN} \
    --input_width=${IMAGE_WIDTH} \
    --input_height=${IMAGE_HEIGHT} \
    --input_mean=128 \
    --input_std=128 \
    --graph=${GRAPH} \
    --input_layer=inputs \
    --output_layer=predictions \
    --image=${image} \
    --labels=${SCRIPTDIR}/labels.txt
