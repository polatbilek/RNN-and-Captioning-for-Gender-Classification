class flags(object):

	def __init__(self):

		#set sizes
		self.validation_set_size = 0.2
		self.training_set_size = 0.8

		#input file paths
		self.training_data_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-training-2018-02-27"
		self.test_data_path = "/home/cvrg/darg/pan_data/pan18-author-profiling-test-2018-03-20"

		#output file paths
		self.model_path = "/media/cvrg/HDD/darg/models/en"
		self.model_name = "en-model-0.001-0.0001-3.ckpt"
		self.log_path = "/home/cvrg/darg/logs/logs_rnn_caption_en.txt"



		#########################################################################################################################
		#optimization parameters
		self.lang = "en"
		self.model_save_threshold = 0.82
		self.optimize = True #if true below values will be used for hyper parameter optimization, or if testing is run: all the models in model_path will be tested
							 #if false hyperparameters specified in "model hyperparameters" will be used, and for testing model with model_name and model_path will be used
		self.l_rate = [0.01, 0.001, 0.0001]
		self.reg_param = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
		self.fc_sizes = [100] #it is dummy for now: will be used later


		#########################################################################################################################
		# Model Hyperparameters
		self.l2_reg_lambda = 0.0001
		self.learning_rate = 0.001
		self.num_classes = 2

		self.textrnn_size = 300 #vector sizes
		self.textcnn_size = 300
		self.imagernn_size = 50

		self.fc_size = 100 #fully connected layer size


		##########################################################################################################################
		# Training parameters
		self.use_pretrained_model = False
		self.batch_size = 10
		self.num_epochs = 25
		self.evaluate_every = 5



FLAGS = flags()
