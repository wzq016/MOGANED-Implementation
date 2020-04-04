# import tensorflow as tf

# a = tf.constant([[1,2],[3,4]],tf.int32)
# b = a = tf.constant([[1,2],[3,4]],tf.int32)

# c = tf.expand_dims(a,axis=2)
# d = tf.expand_dims(b,axis=1)

# e = tf.multiply(c,d)
# print(e)
# with tf.Session() as sess:
#     f = sess.run(e)

a = [1,2,3,4]
import random
random.seed(10)

from pure_test2 import f
a = f(a)
print(a)
a = f(a)
print(a)
a = f(a)
print(a)
a = f(a)
print(a)
a = f(a)
print(a)