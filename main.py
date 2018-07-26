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
    net = network(embeddings)


    print("---TRAINING STARTED---")
    train(net, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, target_values, vocabulary, embeddings)

    print("---TESTING STARTED---")
    #test(net, valid_tweets, valid_users, valid_seq_lengths, target_values, vocabulary)


