from parameters import FLAGS
import tensorflow as tf
from preprocess import *
import numpy as np
from model import network
import sys


###########################################################################################################################
##trains and validates the model
###########################################################################################################################
def train(network, target_values):
	saver = tf.train.Saver(max_to_keep=None)

	with tf.Session() as sess:

		# init variables
		init = tf.global_variables_initializer()
		sess.run(init)

		# load the model from checkpoint file if it is required
		if FLAGS.use_pretrained_model == True:
			load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
			saver.restore(sess, load_as)
			print("Loading the pretrained model from: " + str(load_as))

		# for each epoch
		for epoch in range(FLAGS.num_epochs):

			usernames = target_values.keys()
			user_gender = target_values.values()

			print(len(usernames))
			c = list(zip(usernames, user_gender))
			random.shuffle(c)
			usernames, user_gender = zip(*c)
			user_gender = list(user_gender)

			training_index = int(len(usernames) * FLAGS.training_set_size)

			training_users = usernames[0:training_index]
			training_target = user_gender[0:training_index]

			valid_users = usernames[training_index:]
			valid_target = user_gender[training_index:]


			epoch_loss = 0.0
			epoch_accuracy = 0.0
			num_batches = 0.0
			batch_accuracy = 0.0
			batch_loss = 0.0
			training_batch_count = 2#int(len(training_users) / int(FLAGS.batch_size))
			#valid_batch_count = int(len(valid_users) / int(FLAGS.batch_size))

			prev_index = 0

			# TRAINING
			for batch in range(training_batch_count):
				current_index = (batch+1)*FLAGS.batch_size

				if current_index <= len(training_users):
					batch_users = training_users[prev_index:current_index]
					batch_y = training_target[prev_index:current_index]
				else:
					batch_users = training_users[prev_index:]
					batch_y = training_target[prev_index:]

				prev_index = current_index
				##########################################################################
				new_y = []

				for y in batch_y:
					for i in range(10):
						new_y.append(y)

				batch_y = np.asarray(new_y)

				##########################################################################
				batch_x = readVectors(FLAGS.image_vector_dump_folder, batch_users)

				shape = np.shape(batch_x)

				batch_x = np.reshape(batch_x, [shape[0]*shape[1], shape[2], shape[3], shape[4]])
				# run the graph
				feed_dict = {network.X: batch_x, network.Y: batch_y,  network.reg_param: FLAGS.l2_reg_lambda}
				_, loss, prediction, accuracy = sess.run(
					[network.train, network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)

				# calculate the metrics
				batch_loss += loss
				epoch_loss += loss
				batch_accuracy += accuracy
				epoch_accuracy += accuracy
				num_batches += 1

				# print the accuracy and progress of the training
				if batch % FLAGS.evaluate_every == 0 and batch != 0:
					batch_accuracy /= num_batches
					print("Epoch " + "{:2d}".format(epoch) + " , Batch " + "{0:5d}".format(batch) + "/" + str(
						training_batch_count) + " , loss= " + "{0:5.4f}".format(batch_loss) +
						  " , accuracy= " + "{0:0.5f}".format(batch_accuracy) + " , progress= " + "{0:2.2f}".format(
						(float(batch) / training_batch_count) * 100) + "%")
					batch_loss = 0.0
					batch_accuracy = 0.0
					num_batches = 0.0

			sys.exit()
			# VALIDATION
			batch_accuracy = 0.0
			batch_loss = 0.0
			prev_index = 0

			for batch in range(valid_batch_count):
				current_index = (batch + 1) * FLAGS.batch_size

				if current_index <= len(valid_users):
					batch_users = valid_users[prev_index:current_index]
					batch_y = valid_target[prev_index:current_index]
				else:
					batch_users = valid_users[prev_index:]
					batch_y = valid_target[prev_index:]

				prev_index = current_index

				batch_x = readVectors(FLAGS.image_vector_dump_folder, batch_users)

				# run the graph
				feed_dict = {network.X: batch_x, network.Y: batch_y, network.reg_param: FLAGS.l2_reg_lambda}
				_, loss, prediction, accuracy = sess.run(
					[network.train, network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)

				# calculate the metrics
				batch_loss += loss
				batch_accuracy += accuracy

			# print the accuracy and progress of the validation
			batch_accuracy /= valid_batch_count
			epoch_accuracy /= training_batch_count
			print("Epoch " + str(epoch) + " training loss: " + "{0:5.4f}".format(epoch_loss))
			print("Epoch " + str(epoch) + " training accuracy: " + "{0:0.5f}".format(epoch_accuracy))
			print("Epoch " + str(epoch) + " validation loss: " + "{0:5.4f}".format(batch_loss))
			print("Epoch " + str(epoch) + " validation accuracy: " + "{0:0.5f}".format(batch_accuracy))

			# take the logs
			if FLAGS.optimize:
				f = open(FLAGS.log_path, "a")

				training_loss_line = "Epoch " + str(epoch) + " training loss: " + str(epoch_loss) + "\n"
				training_accuracy_line = "Epoch " + str(epoch) + " training accuracy: " + str(epoch_accuracy) + "\n"
				validation_loss_line = "Epoch " + str(epoch) + " validation loss: " + str(batch_loss) + "\n"
				validation_accuracy_line = "Epoch " + str(epoch) + " validation accuracy: " + str(batch_accuracy) + "\n"

				f.write(training_loss_line)
				f.write(training_accuracy_line)
				f.write(validation_loss_line)
				f.write(validation_accuracy_line)

				f.close()

			# save the model if it performs above the threshold
			# naming convention for the model : {"language"}-model-{"learning rate"}-{"reg. param."}-{"epoch number"}
			if batch_accuracy >= FLAGS.model_save_threshold:
				model_name = str(FLAGS.lang) + "-model-" + str(FLAGS.learning_rate) + "-" + str(
					FLAGS.l2_reg_lambda) + "-" + str(epoch) + ".ckpt"
				save_as = os.path.join(FLAGS.model_path, model_name)
				save_path = saver.save(sess, save_as)
				print("Model saved in path: %s" % save_path)


####################################################################################################################
# main function for standalone runs
####################################################################################################################
if __name__ == "__main__":

	print("---PREPROCESSING STARTED---")

	print("\treading tweets...")
	target_values = readData(FLAGS.training_data_path)

	# single run on training data
	if FLAGS.optimize == False:

		# prin specs
		print("---TRAINING STARTED---")
		model_specs = "with parameters: Learning Rate:" + str(
			FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda)
		model_specs += ", cell size:" + str(FLAGS.rnn_cell_size) + ", embedding size:" + str(
			FLAGS.word_embedding_size) + ", language:" + FLAGS.lang
		print(model_specs)

		# run the network
		tf.reset_default_graph()
		net = network()
		train(net, target_values)

	# hyperparameter optimization
	else:
		for learning_rate in FLAGS.l_rate:
			for regularization_param in FLAGS.reg_param:
				# prep the network
				tf.reset_default_graph()
				FLAGS.learning_rate = learning_rate
				FLAGS.l2_reg_lambda = regularization_param
				net = network()

				# print specs
				print("---TRAINING STARTED---")
				model_specs = "with parameters: Learning Rate:" + str(
					FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda)
				model_specs += ", cell size:" + str(FLAGS.rnn_cell_size) + ", embedding size:" + str(
					FLAGS.word_embedding_size) + ", language:" + FLAGS.lang
				print(model_specs)

				# take the logs
				f = open(FLAGS.log_path, "a")
				f.write("---TRAINING STARTED---\n")
				model_specs += "\n"
				f.write(model_specs)
				f.close()

				# start training
				train(net, target_values)