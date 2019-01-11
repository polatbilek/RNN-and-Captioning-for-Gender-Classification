import tensorflow as tf
from parameters import FLAGS
import numpy as np

class network(object):



	############################################################################################################################
	def __init__(self):

		with tf.device('/device:GPU:0'):	

			num_of_total_filters = len(FLAGS.filter_sizes.split(",")) * FLAGS.num_filters

			#placeholders
			self.reg_param = tf.placeholder(tf.float32, shape=[])
			self.Y = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_classes])
			self.rnn_input = tf.placeholder(tf.float32, [FLAGS.batch_size, 2 * FLAGS.rnn_cell_size])

			# weigths
			self.weights = {'fc1': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size, FLAGS.num_classes]), name="fc1-weights"),
					'att3-fusion-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size, 2 * FLAGS.rnn_cell_size]), name="att3-weights"), #fusion	  
					'att3-fusion-v': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att3-vector")}	#fusion
			# biases
			self.bias = {'fc1': tf.Variable(tf.random_normal([FLAGS.num_classes]), name="fc1-bias-noreg"),
				     'att3-fusion-w': tf.Variable(tf.random_normal([num_of_total_filters]), name="att4-bias-noreg")} #fusion


			# initialize the computation graph for the neural network
			self.architecture()
			self.backward_pass()








    ############################################################################################################################
	def architecture(self):

		with tf.device('/device:GPU:0'):

			#fusion of rnn and cnn
			#self.temp = tf.expand_dims(self.attention_output_rnn, 1)
			#self.temp2 = tf.expand_dims(self.attention_output_cnn, 1)
			#self.concat_output = tf.concat([self.temp, self.temp2], 1)
			#self.att_context_vector_fusion = tf.tanh(tf.tensordot(self.concat_output, self.weights["att3-fusion-w"], axes=1) + self.bias["att3-fusion-w"])
			#self.attentions_fusion = tf.nn.softmax(tf.tensordot(self.att_context_vector_fusion, self.weights["att3-fusion-v"], axes=1))
			#self.attention_output_fusion = tf.reduce_sum(self.concat_output * tf.expand_dims(self.attentions_fusion, -1), 1)


			# FC layer for reducing the dimension to 2(# of classes)
			#self.logits = tf.matmul(self.attention_output_fusion, self.weights["fc1"]) + self.bias["fc1"]
			self.logits = tf.matmul(self.rnn_input, self.weights["fc1"]) + self.bias["fc1"]

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

			return self.loss, self.train














