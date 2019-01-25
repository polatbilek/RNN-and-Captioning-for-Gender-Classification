import tensorflow as tf
from parameters import FLAGS
import numpy as np

class network(object):



	############################################################################################################################
	def __init__(self, embeddings):
		with tf.device('/device:GPU:0'):


			# RNN placeholders
			self.X = tf.placeholder(tf.int32, [FLAGS.batch_size*FLAGS.tweet_per_user, None])
			self.Y = tf.placeholder(tf.float64, [FLAGS.batch_size, FLAGS.num_classes])
			self.reg_param = tf.placeholder(tf.float32, shape=[])

			# weigths
			self.weights = {'fc1': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size, FLAGS.num_classes]), name="fc1-weights"),
							'att1-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size, 2 * FLAGS.rnn_cell_size]), name="att1-weights"),
							'att1-v': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att1-vector"),
							'att2-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size, 2 * FLAGS.rnn_cell_size]), name="att2-weights"),
							'att2-v': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att2-vector")}
			# biases
			self.bias = {'fc1': tf.Variable(tf.random_normal([FLAGS.num_classes]), name="fc1-bias-noreg"),
					     'att1-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att1-bias-noreg"),
					     'att2-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att2-bias-noreg")}


			# initialize the computation graph for the neural network
			# self.rnn()
			self.rnn_with_attention()
			self.architecture()
			self.backward_pass()





    ############################################################################################################################
	def architecture(self):
		with tf.device('/device:GPU:0'):
			#user level attention
			self.att_context_vector_word = tf.tanh(tf.tensordot(self.attention_output, self.weights["att2-w"], axes=1) + self.bias["att2-w"])
			self.attentions_word = tf.nn.softmax(tf.tensordot(self.att_context_vector_word, self.weights["att2-v"], axes=1))
			self.attention_output_word = tf.reduce_sum(self.attention_output * tf.expand_dims(self.attentions_word, -1), 1)

			# FC layer for reducing the dimension to 2(# of classes)
			self.logits = tf.tensordot(self.attention_output_word, self.weights["fc1"], axes=1) + self.bias["fc1"]

			# predictions
			self.prediction = tf.nn.softmax(self.logits)

			# calculate accuracy
			self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

			return self.prediction







    ############################################################################################################################
	def backward_pass(self):
		with tf.device('/device:GPU:0'):
			# calculate loss
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))

			# add L2 regularization
			self.l2 = self.reg_param * sum(
				tf.nn.l2_loss(tf_var)
				for tf_var in tf.trainable_variables()
				if not ("noreg" in tf_var.name or "bias" in tf_var.name)
			)
			self.loss += self.l2

			# optimizer
			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			self.train = self.optimizer.minimize(self.loss)

			return self.accuracy, self.loss, self.train





 







	############################################################################################################################
	def cnn(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):





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



