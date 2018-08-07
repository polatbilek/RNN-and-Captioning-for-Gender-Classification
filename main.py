from parameters import FLAGS
from preprocess import *
import numpy as np
from model import network
from train import *
import random
from eval import *


if __name__ == "__main__":

    print("---PREPROCESSING STARTED---")

    print("\treading word embeddings...")
    vocabulary, embeddings = readGloveEmbeddings(FLAGS.word_embed_path, FLAGS.word_embedding_size)

    print("\treading tweets...")
    tweets, users, target_values, seq_lengths = readData(FLAGS.training_data_path)

    print("\tconstructing datasets and network...")
    training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, test_tweets, test_users, test_seq_lengths = partite_dataset(tweets, users,  seq_lengths)


	#hyperparameter optimization
    for learning_rate in FLAGS.l_rate:
        for regularization_param in FLAGS.reg_param:
            tf.reset_default_graph()
            net = network(embeddings)
            FLAGS.learning_rate = learning_rate
            FLAGS.l2_reg_lambda = regularization_param

            print("---TRAINING STARTED---")
            print("with parameters: Learning Rate:" + str(FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) 
                    + ", cell size:" + str(FLAGS.rnn_cell_size) + ", embedding size:" + str(FLAGS.word_embedding_size))
            train(net, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, target_values, vocabulary, embeddings)


    print("---TESTING STARTED---")
    #test(net, test_tweets, test_users, test_seq_lengths, target_values, vocabulary, embeddings)


