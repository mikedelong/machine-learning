import json
import logging
import os
import random
import shutil
import time

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

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

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
X_train = np.reshape(mnist.train.images[10000:], [-1, 28, 28, 1])
y_train = mnist.train.labels[10000:]
logging.debug('training data has shape %d x %d x %d x %d' % X_train.shape)
logging.debug('training labels have shape %d x %d' % y_train.shape)
X_valid = np.reshape(mnist.train.images[:10000], [-1, 28, 28, 1])
y_valid = mnist.train.labels[:10000]
logging.debug('validation data has shape %d x %d x %d x %d' % X_valid.shape)
logging.debug('validation labels have shape %d x %d' % y_valid.shape)
X_test = np.reshape(mnist.test.images, [-1, 28, 28, 1])
y_test = mnist.test.labels
logging.debug('test data has shape %d x %d x %d x %d' % X_test.shape)
logging.debug('test labels have shape %d x %d' % y_test.shape)

finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logging.info("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
