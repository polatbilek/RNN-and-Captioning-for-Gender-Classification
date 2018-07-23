from parameters import FLAGS
import tensorflow as tf
from preprocess import *

def train(network, tweets, users, target_values, seq_lengths, vocabulary):

    with tf.Session() as sess:

        for epoch in range(FLAGS.num_epochs):

            batch_count = int(len(tweets) / FLAGS.batch_size)
            for batch in range(batch_count):
                batch_x, batch_y, batch_seqlen = prepWordBatchData(tweets, users, target_values, seq_lengths, batch)
                batch_x = word2id(batch_x, vocabulary)

                feed_dict = {network.X: batch_x, network.Y: batch_y, network.sequence_length: batch_seqlen, network.reg_param: FLAGS.l2_reg_lambda}

                predictions = sess.run([network.prediction], feed_dict=feed_dict)