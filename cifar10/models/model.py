# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

WEIGHT_DECAY = 1e2

class Model(object):

    def __init__(self, wd=WEIGHT_DECAY):

        self.wd = wd

    def _get_weights_var(self, name, shape, decay=False):
        """Helper to create an initialized Variable with weight decay.

        The Variable is initialized using a normal distribution whose variance
        is provided by the xavier formula (ie inversely proportional to the number
        of inputs)

        Args:
            name: name of the tensor variable
            shape: the tensor shape
            decay: a boolean indicating if we apply decay to the tensor weights
            using a regularization loss

        Returns:
            Variable Tensor
        """
        # Declare an initializer for this variable
        initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32)
        # Declare variable (it is trainable by default)
        var = tf.get_variable(name=name,
                              shape=shape,
                              initializer=initializer,
                              dtype=tf.float32)
        if decay:
            # We apply a weight decay to this tensor var that is equal to the
            # model weight decay divided by the tensor size
            weight_decay = self.wd
            for x in shape:
                weight_decay /= x
            # Weight loss is L2 loss multiplied by weight decay
            weight_loss = tf.multiply(tf.nn.l2_loss(var),
                                      weight_decay,
                                      name='weight_loss')
            # Add weight loss for this variable to the global losses collection
            tf.add_to_collection('losses', weight_loss)

        return var

    def inference(self, images):
        raise NotImplementedError('Model subclasses must implement this method')

    def loss(self, logits, labels):

        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_loss = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy_loss')
        # We use a global collection to track losses
        tf.add_to_collection('losses', cross_entropy_loss)

        # The total loss is the sum of all losses, including the cross entropy
        # loss and all of the weight losses (see variables declarations)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss
