class flags(object):

    def __init__(self):
        self.test_set_size = 0.1
        self.validation_set_size = 0.1
        self.training_set_size = 0.8
        self.word_embed_path = "C:\\Users\\polat\\Desktop\\PAN_files\\glove.twitter.27B.50d.txt"
        self.training_data_path = "C:\\Users\\polat\\Desktop\\PAN_files\\PAN_data_sets\\pan18-author-profiling-training-2018-02-27"
        self.model_path = "./models/cnnonly"
        self.log_path = "./runlogs_cnn.txt"
        self.lang = "en"

        # Model Hyperparameters
        self.num_filters = 75
        self.l2_reg_lambda = 0.0005
        self.word_embedding_size = 50
        self.learning_rate = 0.0001
        self.num_classes = 2

        # Training parameters
        self.batch_size = 10
        self.num_epochs = 20
        self.evaluate_every = 25


FLAGS = flags()
