# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models.model as model

WEIGHT_DECAY = 5e2

class NiN(model.Model):

    def __init__(self, wd=WEIGHT_DECAY, dropout=0.0):

        super(NiN, self).__init__(wd, dropout)

    def nin_layer(self, inputs, size, filters, stride,
                  dim1, dim2, name, decay=False):
        # size x size x filters -> 1 x 1 x dim1 -> 1 x 1 x dim2
        with tf.variable_scope(name) as scope:
            conv = self.conv_layer(inputs,
                                   size=size,
                                   filters=filters,
                                   stride=stride,
                                   decay=decay,
                                   name='conv')

            cccp1 = self.conv_layer(conv,
                                    size=1,
                                    filters=dim1,
                                    stride=1,
                                    decay=decay,
                                    name='cccp1')

            cccp2 = self.conv_layer(cccp1,
                                    size=1,
                                    filters=dim2,
                                    stride=1,
                                    decay=decay,
                                    name='cccp2')
        return cccp2
