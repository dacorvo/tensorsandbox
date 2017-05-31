# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import models.model as model

WEIGHT_DECAY = 5e2

class Nin0(model.Model):

    def __init__(self, wd=WEIGHT_DECAY):

        super(Nin0, self).__init__(wd)

    def inference(self, images):

        # Nin1 = 5x5 -> 1x1 -> 1x1
        with tf.variable_scope('conv1') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[5, 5, 3, 64],
                                            decay=False)
            biases = tf.get_variable('biases', 
                                    shape=[64],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(images,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name= scope.name)

        with tf.variable_scope('cccp1') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[1, 1, 64, 64],
                                            decay=False)
            biases = tf.get_variable('biases', 
                                    shape=[64],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(conv1,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            cccp1 = tf.nn.relu(pre_activation, name= scope.name)
        
        with tf.variable_scope('cccp2') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[1, 1, 64, 64],
                                            decay=False)
            biases = tf.get_variable('biases', 
                                    shape=[64],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(cccp1,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            cccp2 = tf.nn.relu(pre_activation, name= scope.name)

        # pool1
        with tf.variable_scope('pooling1') as scope:
            pool1 = tf.nn.max_pool(cccp2,
                                   ksize=[1,3,3,1],
                                   strides=[1,2,2,1],
                                   padding='SAME',
                                   name='pooling1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[5, 5, 64, 64],
                                            decay=False)
            biases = tf.get_variable('biases', 
                                    shape=[64],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(pool1,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name= scope.name)

        with tf.variable_scope('cccp3') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[1, 1, 64, 64],
                                            decay=False)
            biases = tf.get_variable('biases', 
                                    shape=[64],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(conv2,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            cccp3 = tf.nn.relu(pre_activation, name= scope.name)
        
        with tf.variable_scope('cccp4') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[1, 1, 64, 10],
                                            decay=False)
            biases = tf.get_variable('biases', 
                                    shape=[10],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(cccp3,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            cccp4 = tf.nn.relu(pre_activation, name= scope.name)

        # Average pooling
        with tf.variable_scope('avg_pool') as scope:
            # Use current spatial dimensions as Kernel size to produce a scalar
            N = cccp4.get_shape().as_list()[0]
            w = cccp4.get_shape().as_list()[1]
            h = cccp4.get_shape().as_list()[2]
            avg = tf.nn.avg_pool(cccp4,
                                 ksize=[1,w,h,1],
                                 strides=[1,1,1,1],
                                 padding='VALID',
                                 name='avg_pool')
        # Reshape output
        return tf.reshape(avg, shape=[N,-1])
