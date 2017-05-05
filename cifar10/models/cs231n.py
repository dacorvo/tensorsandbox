# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models.model as model

class Cs231n(model.Model):

    def __init__(self, wd=model.WEIGHT_DECAY):

        super(Cs231n, self).__init__(wd)

    def inference(self, images):

        # First convolutional layer
        with tf.variable_scope('conv1') as scope:
            # Declare a weights variable to which we associate a L2 loss
            weights = self._get_weights_var('weights',
                                            shape=[3, 3, 3, 64],
                                            decay=True)
            biases = tf.get_variable('biases', 
                                    shape=[64],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(images,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name= scope.name)

        # Apply pooling and local response normalization
        with tf.variable_scope('pooling1_lrn') as scope:
            # We only apply pooling on spatial dimensions
            pool1 = tf.nn.max_pool(conv1,
                                   ksize=[1,2,2,1],
                                   strides=[1,2,2,1],
                                   padding='SAME',
                                   name='pooling1')
            norm1 = tf.nn.lrn(pool1,
                              bias=1.0,
                              alpha=0.001/9.0,
                              beta=0.75,
                              name='norm1')

        # First fully connected layer
        with tf.variable_scope('fc2') as scope:
            N = images.get_shape().as_list()[0]
            reshape = tf.reshape(norm1, shape=[N, -1])
            dim = reshape.get_shape()[1].value
            weights = self._get_weights_var('weights',
                                            shape=[dim,500],
                                            decay=True)
            biases = tf.get_variable('biases',
                                    shape=[500],
                                    dtype=tf.float32, 
                                    initializer=tf.constant_initializer(0.0))
            fc2 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                             name=scope.name)

        # Second fully connected layer
        with tf.variable_scope('fc3') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[500,10],
                                            decay=False)
            biases = tf.get_variable('biases',
                                    shape=[10],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            fc3 = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

        return fc3
