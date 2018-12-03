import tensorflow as tf
from parameters import FLAGS
import numpy as np

class network(object):



	############################################################################################################################
	def __init__(self, embeddings):
		with tf.device('/device:GPU:0'):

			# placeholders
			self.textrnn_input = tf.placeholder(tf.float64, [FLAGS.batch_size, FLAGS.textrnn_size])
			self.textcnn_input = tf.placeholder(tf.float64, [FLAGS.batch_size, FLAGS.textcnn_size])
			self.imagernn_input = tf.placeholder(tf.float64, [FLAGS.batch_size, FLAGS.imagernn_size])
			self.Y = tf.placeholder(tf.float64, [FLAGS.batch_size, FLAGS.num_classes])
			self.reg_param = tf.placeholder(tf.float32, shape=[])

			# weigths
			self.weights = {'fc1': tf.Variable(tf.random_normal([FLAGS.textrnn_size + FLAGS.textcnn_size + FLAGS.imagernn_size, FLAGS.num_classes]), name="fc1-weights")}
			# biases
			self.bias = {'fc1': tf.Variable(tf.random_normal([FLAGS.num_classes]), name="fc1-bias-noreg")}


			# initialize the computation graph for the neural network
			self.architecture()
			self.backward_pass()







    ############################################################################################################################
	def architecture(self):

		with tf.device('/device:GPU:0'):

			#concatenation of features
			self.concatenated_input = tf.concat([self.textrnn_input, self.textcnn_input, self.imagernn_input], 1)

			# FC layer for reducing the dimension to 2(# of classes)
			self.logits = tf.matmul(self.concatenated_input, self.weights["fc1"]) + self.bias["fc1"]

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


























