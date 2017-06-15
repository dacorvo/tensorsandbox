# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models.model as model
from models.nin.nin import NiN

WEIGHT_DECAY = 5e2
DROPOUT = 0.5

class Nin2(NiN):

    def __init__(self, wd=WEIGHT_DECAY, dropout=DROPOUT):

        super(Nin2, self).__init__(wd, dropout)

    def inference(self, images):

        # Nin1 = 5x5x192 / 160 / 96
        nin1 = self.nin_layer(images,
                              size=5,
                              filters=192,
                              stride=1,
                              dim1=160,
                              dim2=96,
                              decay=False,
                              name='nin1')

        # pool1
        pool1 = self.pool_layer(nin1,
                                size=3,
                                stride=2,
                                name='pool1')
        # dropout
        if self.dropout > 0 and self.dropout < 1:
            pool1 = tf.nn.dropout(pool1, (1.0 - self.dropout))

        # NiN2 = 5x5x192 / 192 / 192
        nin2 = self.nin_layer(pool1,
                              size=5,
                              filters=192,
                              stride=1,
                              dim1=192,
                              dim2=192,
                              decay=False,
                              name='nin2')

        # pool2
        pool2 = self.pool_layer(nin2,
                                size=3,
                                stride=2,
                                name='pool2')
        # dropout
        if self.dropout > 0 and self.dropout < 1:
            pool2 = tf.nn.dropout(pool2, (1.0 - self.dropout))

        # NiN3 3x3x192 / 192 / 10
        nin3 = self.nin_layer(pool2,
                              size=3,
                              filters=192,
                              stride=1,
                              dim1=192,
                              dim2=10,
                              decay=False,
                              name='nin3')

        # Average pooling
        predictions = self.avg_layer(nin3, 'avg_pool')

        return predictions
