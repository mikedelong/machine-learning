import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 0})

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=(2, 2))
    result = tf.reduce_mean(x)

with tf.Session(graph=graph, config=config) as session:
    print (session.run(result, feed_dict={x: [[1.0, 1.0], [2.0, 2.0]]}))
