import tensorflow as tf

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.mul(a, b)

sess = tf.Session()

print   sess.run(y, feed_dict={a: 3, b: 3})
