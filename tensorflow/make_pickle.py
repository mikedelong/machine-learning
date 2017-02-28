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


def split_data(arg_pickle_file_name, arg_train_size, arg_validation_size, arg_test_size):
    logging.debug(arg_pickle_file_name)
    with open(arg_pickle_file_name, 'rb') as f:
        letter_set, correct_values = pickle.load(f)
        all_data = list(zip(letter_set, correct_values))
        random.shuffle(all_data)
        data, labels = zip(*all_data)
    start_train = 0
    end_train = start_train + arg_train_size
    start_validation = arg_train_size
    end_validation = start_validation + arg_validation_size
    start_test = arg_train_size + arg_validation_size
    end_test = start_test + arg_test_size
    return \
        [each[0] for each in data[start_train:end_train]], \
        [each[1] for each in data[start_train:end_train]], \
        [each[0] for each in data[start_validation:end_validation]], \
        [each[1] for each in data[start_validation:end_validation]], \
        [each[0] for each in data[start_test:end_test]], \
        [each[1] for each in data[start_test:end_test]]


pickle_file_name = maybe_pickle(['concatenate_output'], 1800)

with open('concatenate_output.pickle', 'rb') as file_pointer:
    t, _ = pickle.load(file_pointer)
    total_data = len(t)
validation_size = total_data / 20
test_size = total_data / 20
train_size = total_data - validation_size - test_size

train_data, train_labels, validation_data, validation_labels, test_data, test_labels = \
    split_data(pickle_file_name[0], train_size, validation_size, test_size)

logging.debug('Training: %d %d' % (len(train_data), len(train_labels)))
logging.debug('Validation: %d %d' % (len(validation_data), len(validation_labels)))
logging.debug('Testing: %d %d' % (len(test_data), len(test_labels)))
