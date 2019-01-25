class flags(object):

	def __init__(self):

		#set sizes
		self.test_set_size = 0.0
		self.validation_set_size = 0.2
		self.training_set_size = 0.8class flags(object):

	def __init__(self):

		#set sizes
		self.test_set_size = 0.0
		self.validation_set_size = 0.2
		self.training_set_size = 0.8

		#input file paths
		self.training_data_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-training-2018-02-27"
		self.test_data_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-test-2018-03-20"
		self.rnn_vectors_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-training-2018-02-27"
		self.cnn_vectors_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-training-2018-02-27"
		self.yolo_vectors_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-training-2018-02-27"

		#output file paths
		self.model_path = "/media/cvrg/HDD/darg/models/es"
		self.model_name = "en-model-0.001-0.0001-3.ckpt"
		self.log_path = "/home/cvrg/darg/logs/logs_yolo_es.txt"

		#optimization parameters
		self.lang = "es"
		self.model_save_threshold = 0.60
		self.optimize = True #if true below values will be used for hyper parameter optimization, or if testing is run: all the models in model_path will be tested
							 #if false hyperparameters specified in "model hyperparameters" will be used, and for testing model with model_name and model_path will be used
		self.l_rate = [0.01, 0.001]
		self.reg_param = [0.0005, 0.0001, 0.00005, 0.00001, 0.000005]




		#########################################################################################################################
		# Model Hyperparameters
		self.l2_reg_lambda = 0.0001
		self.learning_rate = 0.001
		self.num_classes = 2
			
			#CNN
		self.num_filters = 75
		self.yolo_output_size = 200





		##########################################################################################################################
		# Training parameters
		self.use_pretrained_model = False
		self.image_per_user = 10
		self.batch_size = 10
		self.num_epochs = 25
		self.evaluate_every = 5



FLAGS = flags()

		#input file paths
		self.word_embed_path = "/home/cvrg/darg/glove/glove.twitter.27B.200d.txt" #change word embedding size too
		self.training_data_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-training-2018-02-27"
		self.test_data_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-test-2018-03-20"
		#self.word_embed_path = "C:\\Users\\polat\\Desktop\\PAN_files\\glove\\glove.twitter.27B.50d.txt"
		#self.training_data_path = "C:\\Users\\polat\\Desktop\\PAN_files\\PAN_data_sets\\pan18-author-profiling-training-2018-02-27"

		#output file paths
		self.model_path = "/media/cvrg/HDD/darg/models/es"
		self.model_name = "en-model-0.001-0.0001-3.ckpt"
		self.log_path = "/media/cvrg/HDD/darg/logs/logs_rnn_es.txt"
		#self.log_path = "C:\\Users\\polat\\Desktop\\logs_rnn.txt"

		#optimization parameters
		self.lang = "es"
		self.model_save_threshold = 0.75
		self.optimize = True #if true below values will be used for hyper parameter optimization, or if testing is run: all the models in model_path will be tested
							 #if false hyperparameters specified in "model hyperparameters" will be used, and for testing model with model_name and model_path will be used
		self.l_rate = [0.01, 0.001]
		self.reg_param = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
		self.rnn_cell_sizes = [50, 100, 150]




		#########################################################################################################################
		# Model Hyperparameters
		self.l2_reg_lambda = 0.0001
		self.learning_rate = 0.001
		self.num_classes = 2
			#CNN
		self.num_filters = 75
			#RNN
		self.word_embedding_size = 200
		self.rnn_cell_size = 150





		##########################################################################################################################
		# Training parameters
		self.use_pretrained_model = False
		self.tweet_per_user = 100
		self.batch_size = 10
		self.num_epochs = 25
		self.evaluate_every = 5



FLAGS = flags()
