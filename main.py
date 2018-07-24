from parameters import FLAGS
from preprocess import *
import numpy as np
from model import network
from train import *


if __name__ == "__main__":
    vocabulary, embeddings = readGloveEmbeddings(FLAGS.word_embed_path, FLAGS.word_embedding_size)
    tweets, users, target_values, seq_lengths = readData(FLAGS.training_data_path)


    net = network()

    print(type(vocabulary))
    print(vocabulary["UNK"])

    train(net, tweets, users, target_values, seq_lengths, vocabulary)


