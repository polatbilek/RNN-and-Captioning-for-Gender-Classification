from parameters import FLAGS
import tensorflow as tf
from preprocess import *

def train(network, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, target_values, vocabulary, embeddings):


    with tf.Session() as sess:

        # init variables
        init = tf.global_variables_initializer()
        sess.run(network.embedding_init, feed_dict={network.embedding_placeholder: embeddings})
        sess.run(init)
        

        for epoch in range(FLAGS.num_epochs):

            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0.0            
            batch_accuracy = 0.0
            batch_loss = 0.0
            batch_count = int(len(training_tweets) / FLAGS.batch_size)

            for batch in range(batch_count):

                #prepare the batch
                training_batch_x, training_batch_y, training_batch_seqlen = prepWordBatchData(training_tweets, training_users, target_values, training_seq_lengths, batch)
                training_batch_x = word2id(training_batch_x, vocabulary)
                

                #run the graph
                feed_dict = {network.X: training_batch_x, network.Y: training_batch_y, network.sequence_length: training_batch_seqlen, network.reg_param: FLAGS.l2_reg_lambda}
                _, loss, prediction, accuracy = sess.run([network.train, network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)
                
                #for i in range(len(training_batch_y)):
                    #print(str(prediction[i]) + "---" + str(training_batch_y[i]))

                #calculate the metrics
                batch_loss += loss
                epoch_loss += loss
                batch_accuracy += accuracy
                epoch_accuracy += accuracy
                num_batches += 1

                #print the accuracy and progress of the training
                if batch % FLAGS.evaluate_every == 0 and batch != 0:
                    batch_accuracy /= num_batches
                    print("Epoch " +"{:2d}".format(epoch)+ " , Batch " +"{0:6d}".format(batch)+ "/" +str(batch_count)+ " , loss= " +"{0:5.4f}".format(batch_loss)+ 
                        " , accuracy= " + "{0:0.5f}".format(batch_accuracy) + " , progress= " +"{0:2.2f}".format((float(batch) / batch_count) * 100) + "%")
                    batch_loss = 0.0
                    batch_accuracy = 0.0
                    num_batches = 0.0
