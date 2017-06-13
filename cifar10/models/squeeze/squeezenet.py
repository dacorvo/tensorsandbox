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

    def fire(self, inputs, s1x1, e1x1, e3x3, decay=False):

        channels = inputs.get_shape()[3]
        # Squeeze sub-layer
        with tf.variable_scope('s1x1') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[1, 1, channels, s1x1],
                                            decay=decay)
            biases = tf.get_variable('biases', 
                                    shape=[s1x1],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(inputs,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            squeezed_inputs = tf.nn.relu(pre_activation, name= scope.name)

        # Expand 1x1 sub-layer
        with tf.variable_scope('e1x1') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[1, 1, s1x1, e1x1],
                                            decay=decay)
            biases = tf.get_variable('biases', 
                                    shape=[e1x1],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(squeezed_inputs,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            e1x1_outputs = tf.nn.relu(pre_activation, name= scope.name)
        
        # Expand 3x3 sub-layer
        with tf.variable_scope('e3x3') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[1, 3, s1x1, e3x3],
                                            decay=decay)
            biases = tf.get_variable('biases', 
                                    shape=[e3x3],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(squeezed_inputs,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            e3x3_outputs = tf.nn.relu(pre_activation, name= scope.name)
        
        # Concatenate outputs along the last dimension (channel)
        return tf.concat([e1x1_outputs, e3x3_outputs], 3)
