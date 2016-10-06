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

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import tensorflow.python.platform
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10.input
import cifar10.nonlinearities as nls

from tensorflow.python.platform import gfile


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', os.path.join(os.path.split(__file__)[0],'data'),
                           """Path to the CIFAR-10 data directory.""")

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _generate_image_and_label_batch(image, label, min_queue_examples):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'FLAGS.batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    #tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [FLAGS.batch_size])


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Raises:
      ValueError: if no data_dir

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin',
                              'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    for f in filenames:
      if not gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = cifar10.input.read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.image.random_crop(reshaped_image, [height, width])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties:
    #   you want a lot of things in the queue for good mixing, because you put them in in order,
    #   but pull out a random one
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples)


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Raises:
      ValueError: if no data_dir

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')


    if not eval_data:
        filenames = [os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin',
                                  'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
      filenames = [os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin',
                                'test_batch.bin')]
      num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = cifar10.input.read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples)



class Model(object):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    def __init__(self):
        conv = nls.convs[FLAGS.nonlinearity]
        fc = nls.fcs[FLAGS.nonlinearity]

        channels = 32


        self.chains = [
            nls.Chain([
                conv([3, 3, 3, channels]),
                nls.pool2,
                conv([3, 3, channels, channels]),
                nls.pool2]),
            nls.Chain([
                conv([3, 3, channels, channels]),
                conv([3, 3, channels, channels]),
                conv([3, 3, channels, channels]),
                conv([3, 3, channels, channels]),
                nls.pool2]),
            nls.Chain([
                conv([3, 3, channels, channels]),
                conv([3, 3, channels, channels]),
                conv([3, 3, channels, channels]),
                conv([3, 3, channels, channels]),
                nls.global_average_pooling,
                fc([int(channels),int(384)])]),
            nls.Chain([
                fc([384,192]),
                nls.Linear([192,NUM_CLASSES],stddev=0.01)])
        ]

    def __call__(self,x,train=None):
        assert isinstance(train,bool)

        for chain in self.chains[:-1]:
            x = chain(x)
            if train:
                x = tf.nn.dropout(x,0.5)

        logits = self.chains[-1](x)
        return logits


def loss(logits, labels, global_step):
    """Add L2Loss to all the trainable variables.

    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    with tf.name_scope('cost') as scope:
        with tf.name_scope('one_hot'):
            # Reshape the labels into a dense Tensor of
            # shape [batch_size, NUM_CLASSES].
            sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
            indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
            concated = tf.concat(1, [indices, sparse_labels])
            dense_labels = tf.sparse_to_dense(concated,
                                              [FLAGS.batch_size, NUM_CLASSES],
                                              1.0, 0.0)
        with tf.name_scope('xent'):
            # Calculate the average cross entropy loss across the batch.
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, dense_labels)/tf.log(10.0)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

        with tf.name_scope('ExpAverage'):
            exp_xent = tf.train.ExponentialMovingAverage(0.998, name='xent_avg',num_updates=global_step-1)
            exp_xent_op = exp_xent.apply([cross_entropy_mean])

        with tf.control_dependencies([exp_xent_op]):
            cross_entropy_mean2 = tf.identity(cross_entropy_mean)

        tf.scalar_summary(scope+'raw', cross_entropy_mean)
        tf.scalar_summary(scope+'avg', exp_xent.average(cross_entropy_mean))

        return cross_entropy_mean2


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    opt = tf.train.AdamOptimizer()

    grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    return apply_gradient_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filepath,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                                 reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
