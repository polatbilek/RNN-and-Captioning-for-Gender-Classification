import tensorflow as tf
from parameters import FLAGS
import numpy as np

class network(object):



	############################################################################################################################
	def __init__(self):

		#with tf.device('/device:GPU:0'):
		self.reg_param = tf.placeholder(tf.float32, shape=[])
		self.Y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
		self.X = tf.placeholder(tf.float32, [None, 19, 19, 1024]) # faltten batchsize,10

		filter_size = 1024
		filter_shape = [1, 1, 1024, filter_size]


		self.weights = { 'layer1': tf.Variable(tf.random_normal([filter_size, FLAGS.ffnn]), name="layer1-weights"),
						 'layer2': tf.Variable(tf.random_normal([FLAGS.ffnn, FLAGS.num_classes]), name="layer2-weights"),
						 'conv' : tf.Variable(tf.random_normal(filter_shape), name="conv_weight"),
						 'att-w' : tf.Variable(tf.random_normal([filter_size, filter_size]), name="att-weight"),
						 'att-v': tf.Variable(tf.random_normal([filter_size]), name="att-vector")}

		self.bias = {'layer1': tf.Variable(tf.random_normal([FLAGS.ffnn]), name="layer1-bias-noreg"),
						'layer2': tf.Variable(tf.random_normal([FLAGS.num_classes]), name="layer2-bias-noreg"),
					 'conv': tf.Variable(tf.random_normal([filter_size]), name="b-noreg"),
					 'att-w': tf.Variable(tf.random_normal([filter_size]), name="att-weight-noreg")}

		# Convolutional + MaxPool
		self.conv = tf.nn.conv2d(
			self.X,
			self.weights['conv'],
			strides=[1, 1, 1, 1],
			padding="VALID",
			name="conv")

		self.conv = tf.squeeze(self.conv)

		self.h = tf.nn.relu(tf.nn.bias_add(self.conv, self.bias['conv']), name="relu1")

		self.pooled = tf.nn.avg_pool(
			self.h,
			ksize=[1, 19, 19, 1], #ksize dikkat!
			strides=[1, 1, 1, 1],
			padding='VALID',
			name="pool")

		self.pooled = tf.squeeze(self.pooled)

		self.cnn_output = tf.reshape(self.pooled, [FLAGS.batch_size, 10, 1024])

		# Attention
		self.att_context_vector = tf.tanh(tf.tensordot(self.cnn_output, self.weights["att-w"], axes=1) + self.bias["att-w"], name="tanh-attention_context_vector")
		self.attentions = tf.nn.softmax(tf.tensordot(self.att_context_vector, self.weights["att-v"], axes=1), name="softmax-attentions")
		self.attention_output = tf.reduce_sum(self.cnn_output * tf.expand_dims(self.attentions, -1), 1, name="reduce_sum-attention_output")

		self.attention_output = tf.squeeze(self.attention_output)

		# Fully Connected
		self.hidden_layer = tf.tensordot(self.attention_output, self.weights["layer1"], 1) + self.bias["layer1"]
		self.activated_hidden = tf.nn.relu(self.hidden_layer, name="relu2")
		self.logits = tf.tensordot(self.activated_hidden, self.weights["layer2"], 1) + self.bias["layer2"]


		# Prediction and Accuracy
		self.prediction = tf.nn.softmax(self.logits, name="softmax-prediction")

		self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name="reduce_mean-accuracy")

		# Loss + Regularization + Backprop
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y, name="cross_entropy"),name="reduce_mean-loss")

		# add L2 regularization
		self.l2 = self.reg_param * sum(
			tf.nn.l2_loss(tf_var)
			for tf_var in tf.trainable_variables()
			if not ("noreg" in tf_var.name or "bias" in tf_var.name)
		)
		self.loss += self.l2

		# optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name="optimizer")
		self.train = self.optimizer.minimize(self.loss)


