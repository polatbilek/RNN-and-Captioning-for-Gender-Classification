from parameters import FLAGS
import tensorflow as tf
from preprocess import *

def test(network, test_tweets, test_users, test_seq_lengths, target_values, vocabulary):
    with tf.Session() as sess:

        for epoch in range(FLAGS.num_epochs):

            batch_count = int(len(test_tweets) / FLAGS.batch_size)
            for batch in range(batch_count):
                training_batch_x, training_batch_y, training_batch_seqlen = prepWordBatchData(test_tweets, test_users, target_values, test_seq_lengths, batch)
                training_batch_x = word2id(training_batch_x, vocabulary)

                feed_dict = {network.X: training_batch_x, network.Y: training_batch_y, network.sequence_length: training_batch_seqlen, network.reg_param: FLAGS.l2_reg_lambda}

                predictions = sess.run([network.prediction], feed_dict=feed_dict)
