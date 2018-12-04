from parameters import FLAGS
import tensorflow as tf
from preprocess import *
import numpy as np
from model import network



###########################################################################################################################
##trains and validates the model
###########################################################################################################################
def train(network, training_textrnn_vectors, training_textcnn_vectors, training_imagernn_vectors, training_users, \
          valid_textrnn_vectors, valid_textcnn_vectors, valid_imagernn_vectors, valid_users, target_values):

	saver = tf.train.Saver(max_to_keep=None)

	with tf.device('/device:GPU:0'):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

			# init variables
			init = tf.global_variables_initializer()
			sess.run(init)


			#load the model from checkpoint file if it is required
			if FLAGS.use_pretrained_model == True:
				load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
				saver.restore(sess, load_as)
				print("Loading the pretrained model from: " + str(load_as))


			#for each epoch
			for epoch in range(FLAGS.num_epochs):
				epoch_loss = 0.0
				epoch_accuracy = 0.0
				num_batches = 0.0            
				batch_accuracy = 0.0
				batch_loss = 0.0
				training_batch_count = int(len(training_textrnn_vectors) / (FLAGS.batch_size))
				valid_batch_count = int(len(valid_textrnn_vectors) / FLAGS.batch_size)


				#TRAINING
				for batch in range(training_batch_count):
					#prepare the batch
					training_batch_textrnn, training_batch_textcnn, training_batch_imagernn, training_batch_y = \
											prepVectorBatchData(training_textrnn_vectors, training_textcnn_vectors, training_imagernn_vectors, training_users, target_values, batch)


					#run the graph
					feed_dict = {network.textrnn_input: training_batch_textrnn, network.textcnn_input: training_batch_textcnn, network.imagernn_input: training_batch_imagernn, \
								network.Y: training_batch_y, network.reg_param: FLAGS.l2_reg_lambda}
					_, loss, prediction, accuracy = sess.run([network.train, network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)

					#calculate the metrics
					batch_loss += loss
					epoch_loss += loss
					batch_accuracy += accuracy
					epoch_accuracy += accuracy
					num_batches += 1

					#print the accuracy and progress of the training
					if batch % FLAGS.evaluate_every == 0 and batch != 0:
						batch_accuracy /= num_batches
						print("Epoch " +"{:2d}".format(epoch)+ " , Batch " +"{0:5d}".format(batch)+ "/" +str(training_batch_count)+ " , loss= " +"{0:5.4f}".format(batch_loss)+ 
							" , accuracy= " + "{0:0.5f}".format(batch_accuracy))
						batch_loss = 0.0
						batch_accuracy = 0.0
						num_batches = 0.0



				#VALIDATION     
				batch_accuracy = 0.0
				batch_loss = 0.0

				for batch in range(valid_batch_count):

					#prepare the batch
					valid_batch_textrnn, valid_batch_textcnn, valid_batch_imagernn, valid_batch_y = \
											prepWordBatchData(valid_textrnn_vectors, valid_textcnn_vectors, valid_imagernn_vectors, valid_users, target_values, batch)


					#run the graph
					feed_dict = {network.textrnn_input: training_batch_textrnn, network.textcnn_input: training_batch_textcnn, network.imagernn_input: training_batch_imagernn, \
								network.Y: training_batch_y, network.reg_param: FLAGS.l2_reg_lambda}
					loss, prediction, accuracy = sess.run([network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)

					#calculate the metrics
					batch_loss += loss
					batch_accuracy += accuracy

				#print the accuracy and progress of the validation
				batch_accuracy /= valid_batch_count
				epoch_accuracy /= training_batch_count
				print("Epoch " + str(epoch) + " training loss: " + "{0:5.4f}".format(epoch_loss))
				print("Epoch " + str(epoch) + " training accuracy: " + "{0:0.5f}".format(epoch_accuracy))
				print("Epoch " + str(epoch) + " validation loss: " + "{0:5.4f}".format(batch_loss))
				print("Epoch " + str(epoch) + " validation accuracy: " + "{0:0.5f}".format(batch_accuracy))

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


				#save the model if it performs above the threshold
				#naming convention for the model : {"language"}-model-{"learning rate"}-{"reg. param."}-{"epoch number"}
				if batch_accuracy >= FLAGS.model_save_threshold:
					model_name = str(FLAGS.lang) + "-model-" + str(FLAGS.fc_size) + "-" + str(FLAGS.learning_rate) + "-" + str(FLAGS.l2_reg_lambda) + "-" + str(epoch) + ".ckpt"
					save_as = os.path.join(FLAGS.model_path, model_name)
					save_path = saver.save(sess, save_as)
					print("Model saved in path: %s" % save_path)











####################################################################################################################
#main function for standalone runs
####################################################################################################################
if __name__ == "__main__":

	print("---PREPROCESSING STARTED---")

	print("\treading vectors...")
	textrnn_vectors, textcnn_vectors, imagernn_Vectors, users, target_values = readVectors(FLAGS.training_data_path)

	print("\tconstructing datasets and network...")
	training_textrnn_vectors, training_textcnn_vectors, training_imagernn_vectors, training_users, \
	valid_textrnn_vectors, valid_textcnn_vectors, valid_imagernn_vectors, valid_users = partite_dataset_vectors(textrnn_vectors, textcnn_vectors, imagernn_Vectors, users)


	#single run on training data
	if FLAGS.optimize == False:

		#print specs
		print("---TRAINING STARTED---")
		model_specs = "with parameters: Learning Rate:" + str(FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) + ", fully connected size:"
		model_specs+=  str(FLAGS.fc_size) + ", language:" + FLAGS.lang
		print(model_specs)

		#run the network
		tf.reset_default_graph()
		net = network()
		train(net, training_textrnn_vectors, training_textcnn_vectors, training_imagernn_vectors, training_users, \
		     valid_textrnn_vectors, valid_textcnn_vectors, valid_imagernn_vectors, valid_users, target_values)

	#hyperparameter optimization
	else:
		for fullyconnected_size in FLAGS.fc_sizes:
			for learning_rate in FLAGS.l_rate:
				for regularization_param in FLAGS.reg_param:

					#prep the network
					tf.reset_default_graph()
					FLAGS.learning_rate = learning_rate
					FLAGS.l2_reg_lambda = regularization_param
					FLAGS.fc_size = fullyconnected_size
					net = network()

					#print specs
					print("---TRAINING STARTED---")
					model_specs = "with parameters: Learning Rate:" + str(FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) + ", fully connected size:"
					model_specs+=  str(FLAGS.fc_size) + ", language:" + FLAGS.lang					
					print(model_specs)

					#take the logs
					f = open(FLAGS.log_path,"a")
					f.write("---TRAINING STARTED---\n")
					model_specs += "\n"
					f.write(model_specs)
					f.close()

					#start training
					train(net, training_textrnn_vectors, training_textcnn_vectors, training_imagernn_vectors, training_users, \
		     			  valid_textrnn_vectors, valid_textcnn_vectors, valid_imagernn_vectors, valid_users, target_values)






