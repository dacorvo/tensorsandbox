#
# Get info on a CIFAR10 model
#
# Usage: python cifar10/info.py [options]
#
# Type 'python cifar10/info.py --help' for a list of available options.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import models.select as select
import models.data

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    # Instantiate the model
    model = select.by_name(FLAGS.model)
    if FLAGS.data_aug:
        images = tf.zeros((1,24,24,3))
    else:
        images = tf.zeros((1,32,32,3))
    logits = model.inference(images)
    print("Model: %s" % FLAGS.model)
    print("Size : %.2f Millions of parameters" % (model.get_size()/10**6))
    print("Flops: %.2f Millions of operations" % (model.get_flops()/10**6))

if __name__ == '__main__':
    tf.app.run()
