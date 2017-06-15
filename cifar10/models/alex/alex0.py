# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models.model as model

WEIGHT_DECAY = 5e2

class Alex0(model.Model):

    def __init__(self, wd=WEIGHT_DECAY):

        super(Alex0, self).__init__(wd)

    def inference(self, images):

        # conv1
        conv1 = self.conv_layer(images,
                                size=5,
                                filters=64,
                                stride=1,
                                decay=False,
                                name='conv1')

        # pool1 & norm1
        pool1 = self.pool_layer(conv1,
                                size=3,
                                stride=2,
                                name='pool1')
        norm1 = self.lrn_layer(pool1, name='norm1')

        # conv2
        conv2 = self.conv_layer(norm1,
                                size=3,
                                filters=96,
                                stride=1,
                                decay=False,
                                name='conv2')

        # norm2 & pool2 (order inverted as compared to conv1)
        norm2 = self.lrn_layer(conv2, name='norm2')
        pool2 = self.pool_layer(norm2,
                                size=3,
                                stride=2,
                                name='pool2')

        # local3
        fc2 = self.fc_layer(pool2,
                            neurons=384,
                            decay=True,
                            name='fc2')

        # local4
        fc3 = self.fc_layer(fc2,
                            neurons=192,
                            decay=True,
                            name='fc3')

        # Last fully connected layer
        fc4 = self.fc_layer(fc3,
                            neurons=10,
                            decay=False,
                            relu=False,
                            name='fc4')

        return fc4
