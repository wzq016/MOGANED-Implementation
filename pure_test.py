# import tensorflow as tf
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'

# a = tf.constant([0,1,0,4],tf.int32)
# c = tf.constant([2,3,4,5],tf.int32)
# b = (5-1)*(1-tf.cast(tf.equal(a,0),tf.float32))+1
# with tf.Session() as sess:
#     print(sess.run(b))

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('./stanford-corenlp-full-2018-10-05')
word,span = nlp.word_tokenize('hello, i\'m ben',True)
print(word)
print(span)
nlp.close()
