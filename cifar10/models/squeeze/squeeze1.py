# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.squeeze.squeezenet import SqueezeNet

WEIGHT_DECAY = 5e2

class Squeeze1(SqueezeNet):

    def __init__(self, wd=WEIGHT_DECAY):

        super(Squeeze1, self).__init__(wd)

    def inference(self, images):

        # conv1
        conv1 = self.conv_layer(images,
                                size=3,
                                filters=64,
                                stride=1,
                                decay=False,
                                name='conv1')

        # pool1
        pool1 = self.pool_layer(conv1,
                                size=3,
                                stride=2,
                                name='pool1')

        # fire2
        fire2 = self.fire_layer(pool1, 32, 64, 64, decay=False, name='fire2')

        # fire3
        fire3 = self.fire_layer(fire2, 32, 64, 64, decay=False, name='fire3')

        # pool2
        pool2 = self.pool_layer(fire3,
                                size=3,
                                stride=2,
                                name='pool2')

        # fire4
        fire4 = self.fire_layer(pool2, 32, 128, 128, decay=False, name='fire4')

        # fire5
        fire5 = self.fire_layer(fire4, 32, 128, 128, decay=False, name='fire5')

        # Final squeeze to get ten classes
        conv2 = self.conv_layer(fire5,
                                size=1,
                                filters=10,
                                stride=1,
                                decay=False,
                                name='squeeze')

        # Average pooling on spatial dimensions
        predictions = self.avg_layer(conv2, name='avg_pool')

        return predictions
