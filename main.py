import tensorflow as tf
import os
from parameters import FLAGS
from preprocess import *
from model import network
from train import *
from eval import *



##combines the train and eval into a single script
if __name__ == "__main__":


	print("---PREPROCESSING STARTED---")

	print("\treading vectors...")
	textrnn_vectors, textcnn_vectors, imagernn_Vectors, users, target_values = readVectors(FLAGS.training_data_path)

	print("\tconstructing datasets and network...")
	training_textrnn_vectors, training_textcnn_vectors, training_imagernn_vectors, training_users, \
	valid_textrnn_vectors, valid_textcnn_vectors, valid_imagernn_vectors, valid_users = partite_dataset_vectors(textrnn_vectors, textcnn_vectors, imagernn_Vectors, users)


	#hyperparameter optimization if it is set
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


	print("---TESTING STARTED---")
	print("\treading vectors for test...")
	textrnn_vectors, textcnn_vectors, imagernn_Vectors, users, target_values = readVectors(FLAGS.test_data_path)
	print("\ttest set size: " + str(len(tweets)))


	#finds every model in FLAGS.model_path and runs every single one
	if FLAGS.optimize == True:
		models = os.listdir(FLAGS.model_path)
		for model in models:
			if model.endswith(".ckpt.index"):
				FLAGS.model_name = model[:-6]
				tf.reset_default_graph()

				for size in FLAGS.fc_sizes:
					if str(size) in FLAGS.model_name:
						FLAGS.fc_size = size
						break

				net = network()
				test(net, textrnn_vectors, textcnn_vectors, imagernn_Vectors, users, target_values)
	#just runs  single model specified in FLAGS.model_path and FLAGS.model_name
	else:
		tf.reset_default_graph()
		net = network()
		test(net, textrnn_vectors, textcnn_vectors, imagernn_vectors, users, target_values)







