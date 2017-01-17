import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 0})

graph = tf.Graph()

with graph.as_default():
    x = tf.Variable(tf.constant([[1.0, 1.0], [2.0, 2.0]]))
    result = tf.reduce_mean(x), tf.reduce_mean(x, 0), tf.reduce_mean(x, 1)

with tf.Session(graph=graph, config=config) as session:
    tf.global_variables_initializer().run()
    print (session.run(result))
