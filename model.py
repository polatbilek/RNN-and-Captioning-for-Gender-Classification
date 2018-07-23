import tensorflow as tf
from parameters import FLAGS

class network(object):

	def __init__(self):
		self.prediction = []
		self.X = tf.placeholder(tf.int32, [None, None])
		self.Y = tf.placeholder(tf.float64, [FLAGS.batch_size, FLAGS.num_classes])
		self.sequence_length = tf.placeholder(tf.int32, [FLAGS.batch_size])
		self.reg_param = tf.placeholder(tf.float32, shape=[])


	def rnn(self):
		pass

	def captioning(self):
		pass

	def cnn(self):
		pass



