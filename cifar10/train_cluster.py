# Adapted from CIFAR10 Tensorflow tutorial
#
# Train a CIFAR10 model
#
# Usage: python cifar10/train.py [options]
#
# Type 'python cifar10/train.py --help' for a list of available options.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from datetime import datetime
import os

import train
import models.data as data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of ps hosts""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of worker hosts""")
tf.app.flags.DEFINE_string('job_name', 'ps',
                           """Job name: either ps or worker.""")
tf.app.flags.DEFINE_integer('task_index', 0,
                            """Task index in the current job.""")

def main(argv=None):
    data.maybe_download_and_extract(FLAGS.data_dir)
    # Create a cluster from the parameter server and worker hosts.
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        train.train_loop(cluster=cluster,
                         master=server.target,
                         task_index=FLAGS.task_index)

if __name__ == '__main__':
    tf.app.run()
