
class flags(object):

	def __init__(self):
		self.dev_sample_percentage = 0.1
		self.word_embed_path = "C:\\Users\\polat\\Desktop\\PAN_files\\glove.twitter.27B.50d.txt"
		self.training_data_path = "C:\\Users\\polat\\Desktop\\PAN_files\\PAN_data_sets\\pan18-author-profiling-training-2018-02-27"
		self.model_path = "./models/cnnonly"
		self.log_path = "./runlogs_cnn.txt"
		self.lang = "en"

		# Model Hyperparameters
		self.embedding_dim = 25
		self.num_filters = 75
		self.l2_reg_lambda = 0.0005
		self.word_embedding_size = 50
		self.learning_rate = 0.0001
		self.num_classes = 2
		self.sequence_length = 190

		# Training parameters
		self.batch_size = 100
		self.num_epochs = 20
		self.evaluate_every = 25


FLAGS = flags()