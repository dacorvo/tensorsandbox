# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Freeze a CIFAR10 graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib

import tensorflow as tf

import job
import models.select

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('run', 0,
                                """Run from which we want to freeze the model.""")

def get_checkpoint_dir(log_dir, model_name, run):
    return os.path.join(log_dir, model_name, '%d' % run, 'train')

def main(argv=None):
    # Deduce checkpoint dir from log directory, model and run
    checkpoint_dir = get_checkpoint_dir(FLAGS.log_dir, FLAGS.model, FLAGS.run)
    # Get the latest valid checkpoint
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        input_graph_path = os.path.join(checkpoint_dir, 'graph.pbtxt')
        input_saver_def_path = ""
        input_binary = False
        checkpoint_path = ckpt.model_checkpoint_path
        output_node_names = "predictions"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_graph_path = os.path.join(checkpoint_dir, '%s.pb' % FLAGS.model)
        clear_devices = False

        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_path, clear_devices, "")
if __name__ == "__main__":
    tf.app.run()
