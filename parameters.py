class flags(object):

    def __init__(self):

		#set sizes
        self.test_set_size = 0.1
        self.validation_set_size = 0.1
        self.training_set_size = 0.8

		#input file paths
        self.word_embed_path = "/media/darg1/Data/dataset/glove/glove.twitter.27B/glove.twitter.27B.50d.txt" #change word embedding size too
        self.training_data_path = "/media/darg1/Data/dataset/PAN2018/author-profiling/pan18-author-profiling-training-2018-02-27"
        #self.word_embed_path = "C:\\Users\\polat\\Desktop\\PAN_files\\glove.twitter.27B.50d.txt"
        #self.training_data_path = "C:\\Users\\polat\\Desktop\\PAN_files\\PAN_data_sets\\pan18-author-profiling-training-2018-02-27"

		#output file paths
        self.model_path = "/home/darg1/Desktop/model/author_profiling_rnn"
        self.model_name = "rnn_en.ckpt"
        self.log_path = "/home/darg1/Desktop/logs"

		#optimization parameters
        self.lang = "es"
        self.model_save_threshold = 0.99
        self.l_rate = [0.001]
        self.reg_param = [0.00001]
		



		#########################################################################################################################
        # Model Hyperparameters
        self.l2_reg_lambda = 0.00001
        self.learning_rate = 0.001
        self.num_classes = 2
			#CNN
        self.num_filters = 75
			#RNN
        self.word_embedding_size = 50
        self.rnn_cell_size = 50





		##########################################################################################################################
        # Training parameters
        self.use_pretrained_model = False
        self.batch_size = 100
        self.num_epochs = 20
        self.evaluate_every = 25



FLAGS = flags()
