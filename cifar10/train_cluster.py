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
import json

import train
import models.data as data

FLAGS = tf.app.flags.FLAGS

# Setup flags
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of ps hosts""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of worker hosts""")
# Run flags
tf.app.flags.DEFINE_string('job_name', '',
                           """Job name: either ps or worker.""")
tf.app.flags.DEFINE_integer('task_index', 0,
                            """Task index in the current job.""")

def main(argv=None):
    data.maybe_download_and_extract(FLAGS.data_dir)
    # If cluster configuration flags were provided, save them
    if FLAGS.ps_hosts != '' and FLAGS.worker_hosts != '':
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")
        cluster_config = {"ps": ps_hosts, "worker": worker_hosts}
        # Save cluster configuration
        with open('cluster.json', 'w') as f:
            json.dump(cluster_config, f)
        print('Cluster configuration saved.')
    else:
        try:
            # Read cluster configuration
            with open('cluster.json', 'r') as f:
                cluster_config = json.load(f)
        except (OSError, IOError) as e:
            print("No cluster configuration found: you need to provide at " \
                  "least once the two lists of ps and worker hosts")
            return

    if FLAGS.job_name =='':
        print('Pass this script a job name (ps or worker) to start a ' \
              'training session.')
        return

    # Create a cluster
    cluster = tf.train.ClusterSpec(cluster_config)

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
