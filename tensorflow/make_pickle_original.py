import os
import numpy
import pickle
from scipy import ndimage
import tarfile
import sys

# todo fix the image size; our images aren't square
image_size = 28  # Pixel width and height.
image_height = 28
image_width = 28
pixel_depth = 255.0  # Number of levels per pixel.

num_classes = 10
numpy.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    # dataset = numpy.ndarray(shape=(len(image_files), image_size, image_size), dtype=numpy.float32)
    dataset = numpy.ndarray(shape=(len(image_files), image_height, image_width), dtype=numpy.float32)
    print(folder)
    correct_values = []
    num_images = 0
    for image in image_files:
        # correct_value = image.split('.')[0].split('_')[1]
        # correct_values.append(correct_value)
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            # if image_data.shape != (image_size, image_size):
            if image_data.shape != (image_height, image_width):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

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
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_filename = 'notMNIST_large.tar.gz'
train_filename = 'notMNIST_small.tar.gz'
train_folders = maybe_extract(train_filename)

train_datasets = maybe_pickle(train_folders, 45)
pass
# test_datasets = maybe_pickle(test_folders, 1800)
