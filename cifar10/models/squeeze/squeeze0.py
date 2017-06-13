# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.squeeze.squeezenet import SqueezeNet

WEIGHT_DECAY = 5e2

class Squeeze0(SqueezeNet):

    def __init__(self, wd=WEIGHT_DECAY):

        super(Squeeze0, self).__init__(wd)

    def inference(self, images):

        # conv1
        with tf.variable_scope('conv1') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[3, 3, 3, 64],
                                            decay=False)
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

        # pool1
        with tf.variable_scope('pooling1') as scope:
            pool1 = tf.nn.max_pool(conv1,
                                   ksize=[1,3,3,1],
                                   strides=[1,2,2,1],
                                   padding='SAME',
                                   name=scope.name)

        # fire2
        with tf.variable_scope('fire2') as scope:
            fire2 = self.fire(pool1, 16, 64, 64)

        # fire3
        with tf.variable_scope('fire3') as scope:
            fire3 = self.fire(fire2, 16, 64, 64)
        
        # pool2
        with tf.variable_scope('pooling2') as scope:
            pool2 = tf.nn.max_pool(fire3,
                                   ksize=[1,3,3,1],
                                   strides=[1,2,2,1],
                                   padding='SAME',
                                   name=scope.name)
        # fire4
        with tf.variable_scope('fire4') as scope:
            fire4 = self.fire(pool2, 32, 128, 128)

        # fire5
        with tf.variable_scope('fire5') as scope:
            fire5 = self.fire(fire4, 32, 128, 128)
       
        # Final squeeze to get ten classes
        with tf.variable_scope('squeeze') as scope:
            weights = self._get_weights_var('weights',
                                            shape=[1, 1, 128*2, 10],
                                            decay=False)
            biases = tf.get_variable('biases', 
                                    shape=[10],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(fire5,
                                weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            classes = tf.nn.relu(pre_activation, name= scope.name)

        # Average pooling on spatial dimensions
        with tf.variable_scope('avg_pool') as scope:
            # Use current spatial dimensions as Kernel size to produce a scalar
            N = classes.get_shape().as_list()[0]
            w = classes.get_shape().as_list()[1]
            h = classes.get_shape().as_list()[2]
            avg = tf.nn.avg_pool(classes,
                                 ksize=[1,w,h,1],
                                 strides=[1,1,1,1],
                                 padding='VALID',
                                 name='avg_pool')
        # Reshape output
        return tf.reshape(avg, shape=[N,-1])

