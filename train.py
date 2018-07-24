from parameters import FLAGS
import tensorflow as tf
from preprocess import *

def train(network, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, target_values, vocabulary):

    with tf.Session() as sess:

        for epoch in range(FLAGS.num_epochs):

            batch_count = int(len(training_tweets) / FLAGS.batch_size)
            for batch in range(batch_count):
                training_batch_x, training_batch_y, training_batch_seqlen = prepWordBatchData(training_tweets, training_users, target_values, training_seq_lengths, batch)
                training_batch_x = word2id(training_batch_x, vocabulary)

                feed_dict = {network.X: training_batch_x, network.Y: training_batch_y, network.sequence_length: training_batch_seqlen, network.reg_param: FLAGS.l2_reg_lambda}

                predictions = sess.run([network.prediction], feed_dict=feed_dict)

                if batch % FLAGS.evaluate_every == 0:
                    valid_batch_count = int(len(valid_tweets) / FLAGS.batch_size)

                    for i in range(valid_batch_count):
                        valid_batch_x, valid_batch_y, valid_batch_seqlen = prepWordBatchData(valid_tweets, valid_users, target_values, valid_seq_lengths, i)
                        valid_batch_x = word2id(valid_batch_x, vocabulary)

                        feed_dict = {network.X: valid_batch_x, network.Y: valid_batch_y, network.sequence_length: valid_batch_seqlen,
                                     network.reg_param: FLAGS.l2_reg_lambda}

                        predictions = sess.run([network.prediction], feed_dict=feed_dict)

