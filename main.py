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

	print("\treading tweets for target values...")
	target_values = readData(FLAGS.training_data_path)


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
		train(net, target_values)

	else:
		for learning_rate in FLAGS.l_rate:
			for regularization_param in FLAGS.reg_param:

				#prep the network
				tf.reset_default_graph()
				FLAGS.learning_rate = learning_rate
				FLAGS.l2_reg_lambda = regularization_param
				FLAGS.rnn_cell_size = FLAGS.rnn_cell_sizes[i]
				FLAGS.num_filters = FLAGS.cnn_filter_counts[i]
				net = network()

				#print specs
				print("---TRAINING STARTED---")
				model_specs = "with parameters: Learning Rate:" + str(FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) + ", rnn cell size:"
				model_specs+=  str(FLAGS.rnn_cell_size) + ", filter size:" + str(FLAGS.num_filters) + ", language:" + FLAGS.lang
				print(model_specs)

				#take the logs
				f = open(FLAGS.log_path,"a")
				f.write("---TRAINING STARTED---\n")
				model_specs += "\n"
				f.write(model_specs)
				f.close()

				#start training
				train(net, target_values)


	print("---TESTING STARTED---")
	print("\treading tweets for test...")
	target_values= readData(FLAGS.test_data_path)
	print("\ttest set size: " + str(len(target_values.keys())))


	#finds every model in FLAGS.model_path and runs every single one
	if FLAGS.optimize == True:
		models = os.listdir(FLAGS.model_path)
		for model in models:
			if model.endswith(".ckpt.index"):
				FLAGS.model_name = model[:-6]
				tf.reset_default_graph()

				if "90" in FLAGS.model_name:
					FLAGS.num_filters = 60
					FLAGS.rnn_cell_size = 90
				elif "120" in FLAGS.model_name:
					FLAGS.num_filters = 80
					FLAGS.rnn_cell_size = 120
				elif "150" in FLAGS.model_name:
					FLAGS.num_filters = 100
					FLAGS.rnn_cell_size = 150

				net = network()
				test(net, tweets, users, seq_lengths, target_values, vocabulary_word, vocabulary_char, embeddings_char, embeddings_word)
	#just runs  single model specified in FLAGS.model_path and FLAGS.model_name
	else:
		tf.reset_default_graph()
		net = network()
		test(net, target_values)







