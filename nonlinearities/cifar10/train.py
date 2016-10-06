# Copyright 2015 Google Inc. All Rights Reserved.
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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import contextlib
import os
import sys
import time
from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
import cifar10.model

import cifar10.nonlinearities as nls

from tfutils import Saver


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_float('max_train_hours', 1000000,
                          """Max Hours to Run.""")

gp = tf.python.platform.default._flags._global_parser
gp.add_argument("--nonlinearity",
                default="ReLU",
                help="type of nonlinearity to use",
                type=str,
                choices=nls.NAMES)


gp.add_argument("--clear",
                action='store_true',
                help="delete data instead of continuing previous run")

class simple(object):
    pass

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False,name="global_step")

        # build the input streams
        with tf.name_scope('training_data'):
            # Get images and labels for CIFAR-10.
            training = simple()
            training.images, training.labels = cifar10.model.inputs(eval_data=False)

        with tf.name_scope('validation_data'):
            validation = simple()
            validation.images, validation.labels = cifar10.model.inputs(eval_data=True)

        #build the model
        with tf.name_scope('model'):
            model = cifar10.model.Model()


        with tf.name_scope('train'):
            # Build a Graph that computes the logits predictions from the
            # inference model.
            training.logits = model(training.images,train=True)
            # Calculate loss.
            training.loss = cifar10.model.loss(training.logits, training.labels,global_step)

        with tf.name_scope('validate'):
            validation.logits = model(validation.images,train=False)
            validation.loss = cifar10.model.loss(validation.logits, validation.labels,global_step)



        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.model.train(training.loss, global_step)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = Saver(FLAGS.train_dir, session=sess, var_list=tf.all_variables())
        saver.restore()

        # Start the queue runners.
        queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
        coordinator = tf.train.Coordinator()
        queue_threads = tf.train.start_queue_runners(sess=sess,coord=coordinator)


        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)

        start_time = time.time()

        for step in range(global_step.eval(sess), FLAGS.max_steps):
            iter_start = time.time()
            if (iter_start - start_time) / 3600 > FLAGS.max_train_hours:
                break

            _, loss_value, _ = sess.run([train_op, training.loss,validation.loss])
            duration = time.time() - iter_start

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save('model.ckpt', global_step=step)




@contextlib.contextmanager
def tempflags(**kwargs):
    gp = tf.python.platform.default._flags._global_parser

    old_flag_dict = FLAGS.__flags.copy()
    old_parsed = FLAGS.__parsed

    args = []
    for key, value in kwargs.items():
        args.append('--' + key)
        args.append(value)

    result = gp.parse_args(args)

    FLAGS.__dict__['__flags'] = vars(result)
    FLAGS.__dict__['__parsed'] = True

    try:
        yield

    finally:
        FLAGS.__dict__['__flags'] = old_flag_dict
        FLAGS.__dict__['__parsed'] = old_parsed


def main(**kwargs):  # pylint: disable=unused-argument

    with tempflags(**kwargs):
        try:
            gfile.MakeDirs(FLAGS.train_dir)
        except:
            pass

        cifar10.model.maybe_download_and_extract()
        train()


if __name__ == '__main__':
    gp.formatter_class=argparse.ArgumentDefaultsHelpFormatter

    if FLAGS.clear:
        try:
            gfile.DeleteRecursively(FLAGS.train_dir)
        except gfile.GOSError:
            pass

    try:
        os.mkdir(FLAGS.train_dir)
    except OSError:
        pass

    cifar10.model.maybe_download_and_extract()
    train()
