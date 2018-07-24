import tensorflow as tf
from parameters import FLAGS

class network(object):

	def __init__(self):
		self.prediction = []

		#RNN placeholders
		self.X = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
		self.Y = tf.placeholder(tf.float64, [FLAGS.batch_size, FLAGS.num_classes])
		self.sequence_length = tf.placeholder(tf.int32, [FLAGS.batch_size])
		self.reg_param = tf.placeholder(tf.float32, shape=[])

		#CNN placeholders
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")



	def rnn(self):
		pass

	def captioning(self):
		pass

	def cnn(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):

		# Embedding layer
		with tf.name_scope("embedding"):
			W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
			self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b-noreg")
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooled_outputs, 3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		return self.h_pool_flat



