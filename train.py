from parameters import FLAGS
import tensorflow as tf
from preprocess import *

def train(network, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, target_values, vocabulary, embeddings):

    saver = tf.train.Saver()
    count=0
    acc=0

    with tf.Session() as sess:

        # init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(network.embedding_init, feed_dict={network.embedding_placeholder: embeddings})
        

        #load the model from checkpoint file if it is required
        if FLAGS.use_pretrained_model == True:
            load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
            saver.restore(sess, load_as)
            print("Loading the pretrained model from: " + str(load_as))

  
        #for each epoch
        for epoch in range(FLAGS.num_epochs):
            acc=0
            count=0
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0.0            
            batch_accuracy = 0.0
            batch_loss = 0.0
            training_batch_count = int(len(training_tweets) / FLAGS.batch_size)
            valid_batch_count = int(len(valid_tweets) / FLAGS.batch_size)

        
            #TRAINING
            for batch in range(training_batch_count):

                #prepare the batch
                training_batch_x, training_batch_y, training_batch_seqlen = prepWordBatchData(training_tweets, training_users, target_values, training_seq_lengths, batch)
                training_batch_x = word2id(training_batch_x, vocabulary)
                
                #run the graph
                feed_dict = {network.X: training_batch_x, network.Y: training_batch_y, network.sequence_length: training_batch_seqlen, network.reg_param: FLAGS.l2_reg_lambda}
                _, loss, prediction, accuracy = sess.run([network.train, network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)

                #calculate the metrics
                batch_loss += loss
                epoch_loss += loss
                batch_accuracy += accuracy
                epoch_accuracy += accuracy
                num_batches += 1

                #print the accuracy and progress of the training
                if batch % FLAGS.evaluate_every == 0 and batch != 0:
                    batch_accuracy /= num_batches
                    print("Epoch " +"{:2d}".format(epoch)+ " , Batch " +"{0:5d}".format(batch)+ "/" +str(training_batch_count)+ " , loss= " +"{0:5.4f}".format(batch_loss)+ 
                        " , accuracy= " + "{0:0.5f}".format(batch_accuracy) + " , progress= " +"{0:2.2f}".format((float(batch) / training_batch_count) * 100) + "%")
                    batch_loss = 0.0
                    batch_accuracy = 0.0
                    num_batches = 0.0



            #VALIDATION            
            num_batches = 0
            batch_accuracy = 0.0
            batch_loss = 0.0

            for batch in range(valid_batch_count):

                #prepare the batch
                valid_batch_x, valid_batch_y, valid_batch_seqlen = prepWordBatchData(valid_tweets, valid_users, target_values, valid_seq_lengths, batch)
                valid_batch_x = word2id(valid_batch_x, vocabulary)
                
                #run the graph
                feed_dict = {network.X: valid_batch_x, network.Y: valid_batch_y, network.sequence_length: valid_batch_seqlen, network.reg_param: FLAGS.l2_reg_lambda}
                loss, prediction, accuracy = sess.run([network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)
                
                #for i in range(len(training_batch_y)):
                    #print(str(prediction[i]) + "---" + str(training_batch_y[i]))

                #calculate the metrics
                batch_loss += loss
                batch_accuracy += accuracy
                num_batches += 1

            #print the accuracy and progress of the validation
            batch_accuracy /= num_batches
            epoch_accuracy /= training_batch_count
            print("Epoch " + str(epoch) + " training loss: " + "{0:5.4f}".format(epoch_loss))
            print("Epoch " + str(epoch) + " training accuracy: " + "{0:0.5f}".format(epoch_accuracy))
            print("Epoch " + str(epoch) + " validation loss: " + "{0:5.4f}".format(batch_loss))
            print("Epoch " + str(epoch) + " valdiation accuracy: " + "{0:0.5f}".format(batch_accuracy))


            #save the model if it performs above the threshold
            #naming convention for the model : {"language"}-model-{"learning rate"}-{"reg. param."}-{"epoch number"}
            if batch_accuracy >= FLAGS.model_save_threshold:
                model_name = str(FLAGS.lang) + "-model-" + str(FLAGS.learning_rate) + "-" + str(FLAGS.l2_reg_lambda) + "-" + str(epoch) + ".ckpt"
                save_as = os.path.join(FLAGS.model_path, model_name)
                save_path = saver.save(sess, save_as)
                print("Model saved in path: %s" % save_path)









