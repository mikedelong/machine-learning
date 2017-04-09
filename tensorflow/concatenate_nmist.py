import logging
import time

import numpy
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

start_time = time.time()


def accuracy(arg_predictions, arg_labels):
    t0 = numpy.sum(numpy.argmax(arg_predictions, 1) == numpy.argmax(arg_labels, 1))
    result = (100.0 * t0 / arg_predictions.shape[0])
    return result


def weight_variable(arg_shape):
    distribution = tensorflow.truncated_normal(arg_shape, stddev=0.1)
    result = tensorflow.Variable(distribution)
    return result


def bias_variable(arg_shape):
    constant = tensorflow.constant(0.1, shape=arg_shape)
    result = tensorflow.Variable(constant)
    return result


def conv2d(arg_input, arg_filter):
    result = tensorflow.nn.conv2d(arg_input, arg_filter, strides=[1, 1, 1, 1], padding='SAME')
    return result


def max_pool(arg_value):
    result = tensorflow.nn.max_pool(arg_value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return result


def model(arg_data, arg_weights1, arg_weights2, arg_weights3, arg_weights4, arg_bias1, arg_bias2, arg_bias3, arg_bias4,
          arg_keep_fraction):
    conv = conv2d(arg_data, arg_weights1)
    pool = max_pool(conv)
    hidden = tensorflow.nn.relu(pool + arg_bias1)
    conv = conv2d(hidden, arg_weights2)
    pool = max_pool(conv)
    hidden = tensorflow.nn.relu(pool + arg_bias2)
    shape = hidden.get_shape().as_list()
    reshape = tensorflow.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tensorflow.nn.relu(tensorflow.matmul(reshape, arg_weights3) + arg_bias3)
    drop = tensorflow.nn.dropout(hidden, arg_keep_fraction)
    result = tensorflow.matmul(drop, arg_weights4) + arg_bias4
    return result


logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
data_train = numpy.reshape(mnist.train.images[10000:], [-1, 28, 28, 1])
labels_train = mnist.train.labels[10000:]
logging.debug('training data has shape %d x %d x %d x %d' % data_train.shape)
logging.debug('training labels have shape %d x %d' % labels_train.shape)
data_valid = numpy.reshape(mnist.train.images[:10000], [-1, 28, 28, 1])
labels_valid = mnist.train.labels[:10000]
logging.debug('validation data has shape %d x %d x %d x %d' % data_valid.shape)
logging.debug('validation labels have shape %d x %d' % labels_valid.shape)
data_test = numpy.reshape(mnist.test.images, [-1, 28, 28, 1])
labels_test = mnist.test.labels
logging.debug('test data has shape %d x %d x %d x %d' % data_test.shape)
logging.debug('test labels have shape %d x %d' % labels_test.shape)

batch_size = 16
channel_count = 1
depth = 16
drop_keep_fraction = 0.7
hidden_count = 128  # was 64
image_height = 28
image_width = 28
label_count = 10
patch_size = 5
random_seed = 1
step_size = 200
steps_limit = 10001

graph = tensorflow.Graph()
with graph.as_default():
    tensorflow.set_random_seed(random_seed)
    train_set = tensorflow.placeholder(tensorflow.float32, shape=(batch_size, image_height, image_width, channel_count))
    train_labels = tensorflow.placeholder(tensorflow.float32, shape=(batch_size, label_count))
    valid_set = tensorflow.constant(data_valid)
    test_set = tensorflow.constant(data_test)
    keep_fraction = tensorflow.placeholder(tensorflow.float32)

    weights_conv1 = weight_variable([patch_size, patch_size, channel_count, depth])
    bias_conv1 = bias_variable([depth])
    weights_conv2 = weight_variable([patch_size, patch_size, depth, depth])
    bias_conv2 = bias_variable([depth])
    weights_conv3 = weight_variable([image_height // 4 * image_width // 4 * depth, hidden_count])
    bias_conv3 = bias_variable([hidden_count])
    weights_conv4 = weight_variable([hidden_count, label_count])
    bias_conv4 = bias_variable([label_count])

    logits = model(train_set, weights_conv1, weights_conv2, weights_conv3, weights_conv4, bias_conv1, bias_conv2,
                   bias_conv3, bias_conv4, keep_fraction)
    loss = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(logits, train_labels))
    optimizer = tensorflow.train.GradientDescentOptimizer(0.05).minimize(loss)
    train_prediction = tensorflow.nn.softmax(logits)
    validation_logits = model(valid_set, weights_conv1, weights_conv2, weights_conv3, weights_conv4, bias_conv1,
                              bias_conv2, bias_conv3, bias_conv4, keep_fraction)
    valid_prediction = tensorflow.nn.softmax(validation_logits)
    test_logits = model(test_set, weights_conv1, weights_conv2, weights_conv3, weights_conv4, bias_conv1, bias_conv2,
                        bias_conv3, bias_conv4, keep_fraction)
    test_prediction = tensorflow.nn.softmax(test_logits)

with tensorflow.Session(graph=graph, config=tensorflow.ConfigProto(device_count={'GPU': 0})) as session:
    tensorflow.initialize_all_variables().run()
    logging.debug('Initialized')
    batch_data = []
    batch_labels = []
    for step in range(steps_limit):
        offset = (step * batch_size) % (labels_train.shape[0] - batch_size)
        batch_data = data_train[offset:(offset + batch_size), :, :, :]
        batch_labels = labels_train[offset:(offset + batch_size), :]
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict={train_set: batch_data,
                                                                                        train_labels: batch_labels,
                                                                                        keep_fraction: drop_keep_fraction})
        if step % step_size == 0:
            logging.debug('Minibatch loss at step %d: %f' % (step, l))
            logging.debug('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            feed_dict = {train_set: batch_data,
                         train_labels: batch_labels,
                         keep_fraction: 1.0}
            logging.debug(
                'Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(feed_dict=feed_dict), labels_valid))
    logging.info('Test accuracy: %.1f%%' % accuracy(
        test_prediction.eval(feed_dict={train_set: batch_data, train_labels: batch_labels, keep_fraction: 1.0}),
        labels_test))

finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logging.info("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
