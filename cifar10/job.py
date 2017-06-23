from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_string('log_dir', './log/cifar10',
                               """Directory where to write event logs.""")
