import logging
import os
import pickle
import random
from scipy import ndimage

import numpy
import tensorflow

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

# todo fix the image size; our images aren't square
image_size =  28  # Pixel width and height.
image_height =  28
image_width = 5 * 28
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = numpy.ndarray(shape=(len(image_files), image_height, image_width), dtype=numpy.float32)
    logging.debug(folder)
    correct_values = []
    num_images = 0
    for image in image_files:
        correct_value = image.split('.')[0].split('_')[1]
        correct_values.append(correct_value)
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_height, image_width):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:
            logging.warn('Could not read: %s : %s - it\'s ok, skipping.' % (image_file, e))

    logging.debug('correct values: %s' % correct_values)
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    logging.debug('Full dataset tensor: %s' % str(dataset.shape))
    logging.debug('Mean: %s' % numpy.mean(dataset))
    logging.debug('Standard deviation: %s' % numpy.std(dataset))
    return dataset, correct_values


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            logging.debug('%s already present - Skipping pickling.' % set_filename)
        else:
            logging.debug('Pickling %s.' % set_filename)
            dataset, correct_values = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump([dataset, correct_values], f, pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                logging.warn('Unable to save data to', set_filename, ':', e)

    return dataset_names


def make_arrays(nb_rows, arg_image_height, arg_image_width):
    if nb_rows:
        dataset = numpy.ndarray((nb_rows, arg_image_height, arg_image_width), dtype=numpy.float32)
        labels = numpy.ndarray(nb_rows, dtype=numpy.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


# todo split labels into 5 one-hot result sets
def split_data(arg_pickle_file_name, arg_train_size, arg_validation_size, arg_test_size, arg_image_height,
               arg_image_width):
    logging.debug(arg_pickle_file_name)
    with open(arg_pickle_file_name, 'rb') as f:
        letter_set, correct_values = pickle.load(f)
        state = numpy.random.get_state()
        numpy.random.shuffle(letter_set)
        numpy.random.set_state(state)
        numpy.random.shuffle(correct_values)
    start_train = 0
    end_train = start_train + arg_train_size
    start_validation = arg_train_size
    end_validation = start_validation + arg_validation_size
    start_test = arg_train_size + arg_validation_size
    end_test = start_test + arg_test_size
    result_train_data = numpy.ndarray((end_train - start_train + 1, arg_image_height, arg_image_width),
                                      dtype=numpy.float32)
    result_train_data[0:end_train - start_train] = letter_set[start_train:end_train]
    result_train_correct = correct_values[start_train:end_train]

    result_validation_data = numpy.ndarray((end_validation - start_validation + 1, arg_image_height, arg_image_width),
                                           dtype=numpy.float32)
    result_validation_data[0:end_validation - start_validation] = letter_set[start_validation:end_validation]
    result_validation_correct = correct_values[start_validation:end_validation]

    result_test_data = numpy.ndarray((end_test - start_test + 1, arg_image_height, arg_image_width),
                                     dtype=numpy.float32)
    result_test_data[0:end_test - start_test] = letter_set[start_test:end_test]
    result_test_correct = correct_values[start_test:end_test]

    return result_train_data, result_train_correct, result_validation_data, result_validation_correct, \
           result_test_data, result_test_correct


def special_ord(arg, arg_index):
    char = arg[arg_index]
    result = ord(char)
    result = result if result != 32 else 64
    result -= 64
    return result


vector_special_ord = numpy.vectorize(special_ord)


def reformat_as_list(dataset, arg_labels, arg_num_labels, arg_image_height, arg_image_width):
    dataset = dataset.reshape((-1, arg_image_height * arg_image_width)).astype(numpy.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    size = len(arg_labels)
    result = []
    for index in range(0, 5):
        t0 = vector_special_ord(arg_labels, 0)
        t1 = numpy.zeros((size, 11))
        t1[numpy.arange(size), t0] = 1
        result.append(t1)

    return dataset, result

def reformat(dataset, arg_labels, arg_num_labels, arg_image_height, arg_image_width):
    dataset = dataset.reshape((-1, arg_image_height * arg_image_width)).astype(numpy.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    size = len(arg_labels)
    # result = numpy.ndarray(shape=(5, arg_num_labels, 11))
    t2 = []
    for index in range(0, 5):
        t0 = vector_special_ord(arg_labels, 0)
        t1 = numpy.zeros((size, 11))
        t1[numpy.arange(size), t0] = 1
        t2.append(t1)
    result = numpy.asanyarray(t2)

    return dataset, result

def accuracy(predictions, labels):
    t0 = numpy.argmax(predictions, 1)
    t1 = numpy.argmax(labels, 1)
    t2 = t0 == t1
    t3 = numpy.sum(t2)
    result = 100.0 * t3 / predictions.shape[0]
    return result
    # return (100.0 * numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
    #         / predictions.shape[0])


pickle_file_name = maybe_pickle(['concatenate_output'], 1800)

with open('concatenate_output.pickle', 'rb') as file_pointer:
    t, _ = pickle.load(file_pointer)
    total_data = len(t)
validation_size = total_data / 20
test_size = total_data / 20
train_size = total_data - validation_size - test_size

train_data, train_labels, validation_data, validation_labels, test_data, test_labels = \
    split_data(pickle_file_name[0], train_size, validation_size, test_size, image_height, image_width)

logging.debug('Training: %d %d' % (len(train_data), len(train_labels)))
logging.debug('Validation: %d %d' % (len(validation_data), len(validation_labels)))
logging.debug('Testing: %d %d' % (len(test_data), len(test_labels)))

num_labels = 11
train_dataset, train_labels = reformat(train_data, train_labels, num_labels, image_height, image_width)
valid_dataset, valid_labels = reformat(validation_data, validation_labels, num_labels, image_height, image_width)
test_dataset, test_labels = reformat(test_data, test_labels, num_labels, image_height, image_width)
logging.info('Training set: %s %s' % (train_dataset.shape, train_labels.shape)) # was train_labels[0].shape
logging.info('Validation set: %s %s' % (valid_dataset.shape, valid_labels.shape))
logging.debug('Test set: %s %s ' % (test_dataset.shape, test_labels.shape))
batch_size = 128

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000

graph = tensorflow.Graph()
with graph.as_default():
    tf_train_dataset = tensorflow.placeholder(tensorflow.float32, shape=(batch_size, image_height * image_width))
    tf_train_labels = tensorflow.placeholder(tensorflow.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tensorflow.constant(valid_dataset)
    tf_test_dataset = tensorflow.constant(test_dataset)
    weights = tensorflow.Variable(tensorflow.truncated_normal([image_height * image_width, num_labels]))

    biases = tensorflow.Variable(tensorflow.zeros([num_labels]))
    logits = tensorflow.matmul(tf_train_dataset, weights) + biases
    loss = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    optimizer = tensorflow.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_prediction = tensorflow.nn.softmax(logits)
    valid_prediction = tensorflow.nn.softmax(tensorflow.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tensorflow.nn.softmax(tensorflow.matmul(tf_test_dataset, weights) + biases)

num_steps = 10001 # 3001

with tensorflow.Session(graph=graph, config=tensorflow.ConfigProto(device_count={'GPU': 0})) as session:
    tensorflow.global_variables_initializer().run()
    logging.debug("Initialized.")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels[0].shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[0][offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            logging.info("Minibatch loss at step %d: %f" % (step, l))
            logging.info("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            logging.info("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    logging.info("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# with tensorflow.Session(graph=graph, config=tensorflow.ConfigProto(device_count={'GPU': 0})) as session:
#     tensorflow.global_variables_initializer().run()
#     print('Initialized.')
#     for step in range(num_steps):
#         # Run the computations. We tell .run() that we want to run the optimizer,
#         # and get the loss value and the training predictions returned as numpy arrays
#         _, l, predictions = session.run([optimizer, loss, train_prediction])
#         if (step % 100 == 0):
#             print('Loss at step %d: %f' % (step, l))
#             print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels[:train_subset, :]))
#             # Calling .eval() on valid_prediction is basically like calling run(), but
#             # just to get that one numpy array. Note that it recomputes all its graph
#             # dependencies.
#             print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels[0]))
#     print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
