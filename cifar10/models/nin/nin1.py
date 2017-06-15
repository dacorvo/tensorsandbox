# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models.model as model
from models.nin.nin import NiN

WEIGHT_DECAY = 5e2

class Nin1(NiN):

    def __init__(self, wd=WEIGHT_DECAY):

        super(Nin1, self).__init__(wd)

    def inference(self, images):

        # Nin1 = 5x5x64 / 64 / 64
        nin1 = self.nin_layer(images,
                              size=5,
                              filters=64,
                              stride=1,
                              dim1=96,
                              dim2=96,
                              decay=False,
                              name='nin1')

        # pool1
        pool1 = self.pool_layer(nin1,
                                size=3,
                                stride=2,
                                name='pool1')

        # NiN2 = 5x5x192 / 192 / 192
        nin2 = self.nin_layer(pool1,
                              size=5,
                              filters=64,
                              stride=1,
                              dim1=64,
                              dim2=10,
                              decay=False,
                              name='nin2')

        # Average pooling
        predictions = self.avg_layer(nin2, 'avg_pool')
