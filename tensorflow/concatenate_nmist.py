import json
import logging
import os
import random
import shutil
import time

import idx2numpy
import numpy

start_time = time.time()

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

with open('test-nmist-settings.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    blank_file = data['blank_file']
    files_to_generate = data['files_to_generate']
    input_folder = data['input_folder']
    output_folder = data['output_folder']
    random_seed = data['random_seed']

random.seed(random_seed)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    shutil.rmtree(output_folder)
    os.makedirs(output_folder)

count = 0


def createSequences(arg_dataset_size, arg_image_height, arg_image_width, arg_ndarr, arg_labels_raw):
    dataset = numpy.ndarray(shape=(arg_dataset_size, arg_image_height, arg_image_width), dtype=numpy.float32)

    data_labels = []

    index = 0
    word = 0
    while index < arg_dataset_size:
        temp = numpy.hstack(
            [arg_ndarr[word], arg_ndarr[word + 1], arg_ndarr[word + 2], arg_ndarr[word + 3], arg_ndarr[word + 4]])
        dataset[index, :, :] = temp
        temp_str = (
            labels_raw[word], labels_raw[word + 1], labels_raw[word + 2], labels_raw[word + 3], labels_raw[word + 4])
        data_labels.append(temp_str)
        word += 5
        index += 1

    numpy.array(data_labels)

    return dataset, data_labels


# read data and convert idx file to numpy array
ndarr = idx2numpy.convert_from_file('./train-images-idx3-ubyte')
labels_raw = idx2numpy.convert_from_file('./train-labels-idx1-ubyte')

dataset_size = 10000
data, labels = createSequences(dataset_size, 28, 140, ndarr, labels_raw)
logging.debug('we have %d data items and %d labels; expected %d' % (len(data), len(labels), dataset_size))

finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logging.info("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
