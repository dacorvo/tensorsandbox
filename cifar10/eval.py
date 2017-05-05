# Adapted from CIFAR10 Tensorflow tutorial
#
# Evaluate a trained CIFAR10 model
#
# Usage: python cifar10/eval.py [options]
#
# Type 'python cifar10/eval.py --help' for a list of available options.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import models.data as data
from models.cs231n import Cs231n

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                               """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                               """Either 'test' or 'train'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                               """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                                """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                                """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                                """Size of each batch.""")

def evaluate(saver, summary_writer, predictions_op, summary_op):
    """Run an evaluation on FLAGS.num_examples

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        predictions_op: Vector predictions op.
        summary_op: Summary op.

    Returns:
        global_step: the training step corresponding to the evaluation
    """
    with tf.Session() as sess:

        # Try to restore the model parameters from a checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            # /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = \
                    ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return 0

        # Start the queue runners
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([predictions_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        return global_step

def evaluation_loop():
    """Eval model accuracy at regular intervals"""
    with tf.Graph().as_default() as g:
        # Do we evaluate the net on the training data or the test data ?
        test_data = FLAGS.eval_data == 'test'
        # Get images and labels for CIFAR-10
        images, labels = data.inputs(test_data=test_data,
                                     data_dir=FLAGS.data_dir,
                                     batch_size=FLAGS.batch_size)

        # Instantiate the model
        model = Cs231n()
       
        # Build a Graph that computes the logits predictions from the model
        logits = model.inference(images)

        # Calculate predictions (we are only interested in perfect matches, ie k=1)
        predictions_op = tf.nn.in_top_k(logits, labels, 1)

        # Instantiate a saver to restore model variables from checkpoint
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries
        summary_op = tf.summary.merge_all()

        # Since we don't use a session, we need to write summaries ourselves
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        last_step = 0
        while True:
            global_step = evaluate(saver, summary_writer, predictions_op, summary_op)
            if FLAGS.run_once or last_step == global_step:
                break
            last_step = global_step
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    data.maybe_download_and_extract(FLAGS.data_dir)
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluation_loop()


if __name__ == '__main__':
    tf.app.run()
