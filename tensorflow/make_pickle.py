import os
import numpy
import pickle
from scipy import ndimage

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
    correct_values  = []
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
