import logging
import os
import pickle
import random
from scipy import ndimage

import numpy

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

# todo fix the image size; our images aren't square
image_size = 28  # Pixel width and height.
image_height = 28
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
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    print(correct_values)
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    logging.debug('Full dataset tensor:' % dataset.shape)
    logging.debug('Mean:' % numpy.mean(dataset))
    logging.debug('Standard deviation:' % numpy.std(dataset))
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


def reformat(dataset, arg_labels, arg_num_labels, arg_image_height, arg_image_width):
    dataset = dataset.reshape((-1, arg_image_height * arg_image_width)).astype(numpy.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    # t0 = labels[:]
    labels = (numpy.arange(arg_num_labels) == arg_labels[:]).astype(numpy.float32)

    # labels = (numpy.arange(arg_num_labels) == labels[:, None]).astype(numpy.float32)
    return dataset, labels


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

train_dataset, train_labels = reformat(train_data, train_labels, 10, image_height, image_width)
valid_dataset, valid_labels = reformat(validation_data, validation_labels, 10, image_height, image_width)
test_dataset, test_labels = reformat(test_data, test_labels, 10, image_height, image_width)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
