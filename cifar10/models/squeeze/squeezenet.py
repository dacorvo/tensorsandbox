# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models.model as model

WEIGHT_DECAY = 1e1

class SqueezeNet(model.Model):

    def __init__(self, wd=WEIGHT_DECAY, dropout=0.0):

        super(SqueezeNet, self).__init__(wd, dropout)

    def fire_layer(self, inputs, s1x1, e1x1, e3x3, name, decay=False):
        with tf.variable_scope(name) as scope:

            # Squeeze sub-layer
            squeezed_inputs = self.conv_layer(inputs,
                                              size=1,
                                              filters=s1x1,
                                              stride=1,
                                              decay=decay,
                                              name='s1x1')

            # Expand 1x1 sub-layer
            e1x1_outputs = self.conv_layer(squeezed_inputs,
                                           size=1,
                                           filters=e1x1,
                                           stride=1,
                                           decay=decay,
                                           name='e1x1')
            
            # Expand 3x3 sub-layer
            e3x3_outputs = self.conv_layer(squeezed_inputs,
                                           size=3,
                                           filters=e3x3,
                                           stride=1,
                                           decay=decay,
                                           name='e3x3')
        
        # Concatenate outputs along the last dimension (channel)
        return tf.concat([e1x1_outputs, e3x3_outputs], 3)
