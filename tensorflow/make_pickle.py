import os
import pickle
import random
from scipy import ndimage

import numpy

# todo fix the image size; our images aren't square
image_size = 28  # Pixel width and height.
image_height = 28
image_width = 5 * 28
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = numpy.ndarray(shape=(len(image_files), image_height, image_width), dtype=numpy.float32)
    print(folder)
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
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    print(correct_values)
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', numpy.mean(dataset))
    print('Standard deviation:', numpy.std(dataset))
    return dataset, correct_values


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset, correct_values = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump([dataset, correct_values], f, pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(['concatenate_output'], 1800)


def make_arrays(nb_rows, image_height, image_width):
    if nb_rows:
        dataset = numpy.ndarray((nb_rows, image_height, image_width), dtype=numpy.float32)
        labels = numpy.ndarray(nb_rows, dtype=numpy.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, arg_image_height, arg_image_width, valid_size=0):
    # num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, arg_image_height, arg_image_width)
    train_dataset, train_labels = make_arrays(train_size, arg_image_height, arg_image_width)
    vsize_per_class = valid_size  # // num_classes
    print ('vsize per class: ' + str(vsize_per_class))
    tsize_per_class = train_size  # // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set, correct_values = pickle.load(f)
                all_data = list(zip(letter_set, correct_values))
                random.shuffle(all_data)
                letter_set, correct_values = zip(*all_data)

                if valid_dataset is not None:
                    try :
                        valid_letter = letter_set[:vsize_per_class, :, :]
                    except TypeError as typeError:
                        print ( typeError)
                        print (vsize_per_class)
                        print (letter_set[:vsize_per_class, :, :])
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


total_data = 100000
valid_size = total_data / 20
test_size = total_data / 20
train_size = total_data - valid_size - test_size
# train_size = 200000
# valid_size = 10000
# test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, image_height,
                                                                          image_width, valid_size)
# _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
# print('Validation:', valid_dataset.shape, valid_labels.shape)
# print('Testing:', test_dataset.shape, test_labels.shape)
