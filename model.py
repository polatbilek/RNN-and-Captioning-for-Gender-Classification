import tensorflow as tf
from parameters import FLAGS
import numpy as np

class network(object):



	############################################################################################################################
	def __init__(self, embeddings):
		with tf.device('/device:GPU:0'):
			self.prediction = []

			# create word embeddings
			self.tf_embeddings = tf.Variable(tf.constant(0.0, shape=[embeddings.shape[0], embeddings.shape[1]]), trainable=False, name="tf_embeddings")
			self.embedding_placeholder = tf.placeholder(tf.float32, [embeddings.shape[0], embeddings.shape[1]])
			self.embedding_init = self.tf_embeddings.assign(self.embedding_placeholder)  # initialize this once  with sess.run when the session begins

			# create GRU cells
			with tf.variable_scope("tweet"):
				self.cell_fw = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.rnn_cell_size, activation=tf.sigmoid)
				self.cell_bw = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.rnn_cell_size, activation=tf.sigmoid)

			# RNN placeholders
			self.X = tf.placeholder(tf.int32, [FLAGS.batch_size*FLAGS.tweet_per_user, None])
			self.Y = tf.placeholder(tf.float64, [FLAGS.batch_size, FLAGS.num_classes])
			self.sequence_length = tf.placeholder(tf.int32, [FLAGS.batch_size*FLAGS.tweet_per_user])
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
			# self.rnn_with_attention()
			self.captioning()
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
	def rnn(self):

		with tf.device('/device:GPU:0'):	
			# embedding layer
			self.rnn_input = tf.nn.embedding_lookup(self.tf_embeddings, self.X)

			# rnn layer
			(self.outputs, self.output_states) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.rnn_input, self.sequence_length, dtype=tf.float32,scope="tweet")

			# concatenate the backward and forward cells
			self.rnn_output_raw = tf.concat([self.output_states[0], self.output_states[1]], 1)
		
			#reshape the output for the next layers
			self.rnn_output = tf.reshape(self.rnn_output_raw, [FLAGS.batch_size, FLAGS.tweet_per_user, 2*FLAGS.rnn_cell_size])

			return self.rnn_output







    ############################################################################################################################
	def rnn_with_attention(self):

		with tf.device('/device:GPU:0'):
			# embedding layer
			self.rnn_input = tf.nn.embedding_lookup(self.tf_embeddings, self.X)

			# rnn layer
			(self.outputs, self.output_states) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.rnn_input, self.sequence_length, dtype=tf.float32,scope="tweet")

			# concatenate the backward and forward cells
			self.concat_outputs = tf.concat(self.outputs, 2)

			# attention layer
			self.att_context_vector = tf.tanh(tf.tensordot(self.concat_outputs, self.weights["att1-w"], axes=1) + self.bias["att1-w"])
			self.attentions = tf.nn.softmax(tf.tensordot(self.att_context_vector, self.weights["att1-v"], axes=1))
			self.attention_output_raw = tf.reduce_sum(self.concat_outputs * tf.expand_dims(self.attentions, -1), 1)

			#reshape the output for the next layers
			self.attention_output = tf.reshape(self.attention_output_raw, [FLAGS.batch_size, FLAGS.tweet_per_user, 2*FLAGS.rnn_cell_size])

			return self.attention_output









	############################################################################################################################
	def captioning(self):
		
		with tf.device('/device:GPU:0'):
			# embedding layer
			self.rnn_input = tf.nn.embedding_lookup(self.tf_embeddings, self.X)

			# rnn layer
			(self.outputs, self.output_states) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.rnn_input, self.sequence_length, dtype=tf.float32,scope="tweet")

			# concatenate the backward and forward cells
			self.concat_outputs = tf.concat(self.outputs, 2)

			# attention layer
			self.att_context_vector = tf.tanh(tf.tensordot(self.concat_outputs, self.weights["att1-w"], axes=1) + self.bias["att1-w"])
			self.attentions = tf.nn.softmax(tf.tensordot(self.att_context_vector, self.weights["att1-v"], axes=1))
			self.attention_output_raw = tf.reduce_sum(self.concat_outputs * tf.expand_dims(self.attentions, -1), 1)

			#reshape the output for the next layers
			self.attention_output = tf.reshape(self.attention_output_raw, [FLAGS.batch_size, FLAGS.tweet_per_user, 2*FLAGS.rnn_cell_size])

			return self.attention_output









	############################################################################################################################
	def cnn(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):

		with tf.device('/device:GPU:0'):
			# CNN placeholders
			self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
			self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

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



