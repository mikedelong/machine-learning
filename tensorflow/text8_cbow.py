# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import collections
import datetime
import hashlib
import logging
import math
import os
import random
import time
import zipfile
from urllib import urlretrieve

import numpy as np
import tensorflow as tf
from matplotlib import pylab
from sklearn.manifold import TSNE

start_time = time.time()


def maybe_download(arg_filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(arg_filename):
        arg_filename, _ = urlretrieve(url + arg_filename, arg_filename)
    statinfo = os.stat(arg_filename)
    if statinfo.st_size == expected_bytes:
        logging.info('Found and verified %s' % arg_filename)
    else:
        logging.warn(statinfo.st_size)
        raise Exception('Failed to verify ' + arg_filename + '. Can you get to it with a browser?')
    return arg_filename


def read_data(arg_filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(arg_filename) as f:
        result = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return result


def build_dataset(arg_words):
    result_count = [['UNK', -1]]
    # result_count.extend(collections.Counter(arg_words).most_common(vocabulary_size - 1))
    word_counter = collections.Counter()
    for item in arg_words:
        word_counter[item] += 1
    result_count.extend(word_counter.most_common(vocabulary_size - 1))
    result_dictionary = dict()
    for word, _ in result_count:
        result_dictionary[word] = len(result_dictionary)
    result_data = list()
    unk_count = 0
    for word in arg_words:
        if word in result_dictionary:
            index = result_dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        result_data.append(index)
    result_count[0][1] = unk_count
    result_reverse_dictionary = dict(zip(result_dictionary.values(), result_dictionary.keys()))
    return result_data, result_count, result_dictionary, result_reverse_dictionary


def generate_batch(arg_batch_size, arg_bag_window):
    global data_index
    span = 2 * arg_bag_window + 1  # [ skip_window target skip_window ]
    result_batch = np.ndarray(shape=(arg_batch_size, span - 1), dtype=np.int32)
    result_labels = np.ndarray(shape=(arg_batch_size, 1), dtype=np.int32)
    local_buffer = collections.deque(maxlen=span)
    for _ in range(span):
        local_buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for index in range(arg_batch_size):
        buffer_list = list(local_buffer)
        result_labels[index, 0] = buffer_list.pop(arg_bag_window)
        result_batch[index] = buffer_list
        local_buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return result_batch, result_labels


def plot(arg_embeddings, arg_labels, arg_file_name):
    assert arg_embeddings.shape[0] >= len(arg_labels), 'More labels than embeddings'
    pylab.figure(figsize=(18, 18))  # in inches
    for index, label in enumerate(arg_labels):
        x, y = arg_embeddings[index, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.savefig(arg_file_name + '.png')
    pylab.savefig(arg_file_name + '.pdf')
    pylab.show()


random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)
logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)
url = 'http://mattmahoney.net/dc/'
filename = maybe_download('text8.zip', 31344016)
words = read_data(filename)
logging.info('Data size %d' % len(words))
vocabulary_size = 50000
data, count, dictionary, reverse_dictionary = build_dataset(words)
logging.info('Most common words (+UNK): %s' % count[:5])
logging.info('Sample data: %s' % data[:10])
del words  # Hint to reduce memory.
# data_index = 0
logging.info('data: %s' % [reverse_dictionary[di] for di in data[:8]])
for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(arg_batch_size=8, arg_bag_window=skip_window)
    logging.info('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    logging.info('    batch: %s' % [[reverse_dictionary[bj] for bj in bi] for bi in batch])
    logging.info('    labels: %s' % [reverse_dictionary[li] for li in labels.reshape(8)])
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    tf.set_random_seed(random_seed)
    # Input data.
    # train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size, num_skips])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0, seed=random_seed))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size),
                            seed=random_seed))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    # embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    embed = tf.zeros([batch_size, embedding_size])
    for j in range(num_skips):
        embed += tf.nn.embedding_lookup(embeddings, train_dataset[:, j])
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                   train_labels, num_sampled, vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = tf.divide(embeddings, norm)
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100001
config = tf.ConfigProto(device_count={'GPU': 0})

with tf.Session(graph=graph, config=config) as session:
    tf.global_variables_initializer().run()
    logging.info('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, skip_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            logging.info('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                logging.info(log)
    final_embeddings = normalized_embeddings.eval()

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])

words = [reverse_dictionary[i] for i in range(1, num_points + 1)]

md5 = hashlib.md5()
md5.update(str(datetime.datetime.now()))
output_file_name_root = md5.hexdigest()[:5]
logging.debug(output_file_name_root)

# to get a reasonably stable estimate of elapsed time we want to stop the clock before we plot.
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logging.info("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))

plot(two_d_embeddings, words, output_file_name_root)
