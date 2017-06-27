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
import os

import numpy as np
import tensorflow as tf

import job
import models.data as data
import models.select as select

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_data', 'test',
                               """Either 'test' or 'train'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                                """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                                """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('profiling', True,
                             """Whether to add profiling info or not.""")

def get_run_dir(log_dir, model_name):
    model_dir = os.path.join(log_dir, model_name)
    if os.path.isdir(model_dir):
        # We do not create new run directories, we reuse the last one that
        # should have been created by the training process
        run = len(os.listdir(model_dir)) - 1
    else:
        run = 0
    return os.path.join(model_dir, '%d' % run)

def evaluate(saver, checkpoint_dir, summary_writer, predictions_op, summary_op):
    """Run an evaluation on FLAGS.num_examples

    Args:
        saver: Saver.
        checkpoint_dir: the directory from which we load checkpoints
        summary_writer: Summary writer.
        predictions_op: Vector predictions op.
        summary_op: Summary op.

    Returns:
        global_step: the training step corresponding to the evaluation
    """
    with tf.Session() as sess:

        # Try to restore the model parameters from a checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
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
            start = time.time()
            while step < num_iter and not coord.should_stop():
                feed_dict = {'training:0': False}
                if FLAGS.profiling and step == (num_iter -1):
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    predictions = sess.run([predictions_op],
                                           feed_dict=feed_dict,
                                           options=run_options,
                                           run_metadata=run_metadata)
                    summary_writer.add_run_metadata(run_metadata,
                                                    'step %d' %
                                                    int(global_step))
                else:
                    predictions = sess.run([predictions_op],
                                           feed_dict=feed_dict)
                true_count += np.sum(predictions)
                step += 1
            end = time.time()
            duration = end - start
            examples_per_sec = num_iter * FLAGS.batch_size / duration
            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: %.1f inferences/sec, %s accuracy = %.3f'
                  % (datetime.now(), examples_per_sec, FLAGS.eval_data, precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='%s accuracy' % FLAGS.eval_data,
                              simple_value=precision)
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
        images, labels = data.eval_inputs(test_data=test_data,
                                          data_dir=FLAGS.data_dir,
                                          batch_size=FLAGS.batch_size)

        # Instantiate the model
        model = select.by_name(FLAGS.model)

        # Force dropout to zero for evaluation
        model.dropout = 0.0

        # Build a Graph that computes the logits predictions from the model
        logits = model.inference(images)

        # Calculate predictions (we are only interested in perfect matches, ie k=1)
        predictions_op = tf.nn.in_top_k(logits, labels, 1)

        # We restore moving averages instead of raw values
        # Note that at evaluation time, the decay parameter is not used
        variables_averages = \
                tf.train.ExponentialMovingAverage(1.0) # 1.0 decay is unused
        variables_to_restore = variables_averages.variables_to_restore()

        # Instantiate a saver to restore model variables from checkpoint
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries
        summary_op = tf.summary.merge_all()

        # Since we don't use a session, we need to write summaries ourselves
        run_dir = get_run_dir(FLAGS.log_dir, FLAGS.model)
        eval_dir = os.path.join(run_dir, 'eval', FLAGS.eval_data)
        tf.gfile.MakeDirs(eval_dir)
        summary_writer = tf.summary.FileWriter(eval_dir, g)

        # We need a checkpoint dir to restore model parameters
        checkpoint_dir = os.path.join(run_dir, 'train')

        last_step = 0
        while True:
            global_step = evaluate(saver,
                                   checkpoint_dir,
                                   summary_writer,
                                   predictions_op,
                                   summary_op)
            if FLAGS.run_once or last_step == global_step:
                break
            last_step = global_step
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    data.maybe_download_and_extract(FLAGS.data_dir)
    evaluation_loop()


if __name__ == '__main__':
    tf.app.run()
