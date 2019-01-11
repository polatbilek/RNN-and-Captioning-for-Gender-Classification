class flags(object):

	def __init__(self):

		#set sizes
		self.test_set_size = 0.0
		self.validation_set_size = 0.2
		self.training_set_size = 0.8

		#input file paths
		self.word_embed_path = "/home/cvrg/darg/glove/glove.twitter.27B.200d.txt" #change word embedding size too
		self.training_data_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-training-2018-02-27"
		self.test_data_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-test-2018-03-20"
		self.char_embed_path = "/home/cvrg/darg/glove/char_embeddings.27B.25d.txt"


		#output file paths
		self.model_path = "/media/cvrg/HDD/darg/models"
		self.model_path_rnn = "/media/cvrg/HDD/darg/models/zzz-backup/RNNwA/first run/en"
		self.model_path_cnn = "/media/cvrg/HDD/darg/models/zzz-backup/CNNwA/second_run/en"
		self.model_name = "temp.ckpt"
		self.model_name_rnn = "en-model-150-0.01-5e-05-15.ckpt"
		self.model_name_cnn = "en-model-100-0.01-1e-06-11.ckpt"
		self.log_path = "/home/cvrg/darg/logs/logs_CNN_RNN_en.txt"


		#optimization parameters
		self.lang = "en"
		self.model_save_threshold = 0.79
		self.optimize = True #if true below values will be used for hyper parameter optimization, or if testing is run: all the models in model_path will be tested
							 #if false hyperparameters specified in "model hyperparameters" will be used, and for testing model with model_name and model_path will be used
		self.l_rate = [0.00001]
		self.reg_param = [0]
		




		#########################################################################################################################
		# Model Hyperparameters
		self.l2_reg_lambda = 0.0001
		self.learning_rate = 0.001
		self.num_classes = 2
			#CNN
		self.num_filters = 100
		self.sequence_length = 190
		self.char_embedding_size = 25
		self.filter_sizes = "3,6,9"
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
