from parameters import FLAGS
import tensorflow as tf
from preprocess import *
from model import network





#####################################################################################################################
##loads a model and tests it
#####################################################################################################################
def test(network, test_tweets, test_users, test_seq_lengths, target_values, vocabulary, embeddings):
	
	saver = tf.train.Saver()

	with tf.Session() as sess:

		# init variables
		init = tf.global_variables_initializer()
		sess.run(init)
		sess.run(network.embedding_init, feed_dict={network.embedding_placeholder: embeddings})
		user_pred = {}
		acc = 0
		count = 0
		batch_loss = 0.0
		batch_accuracy = 0.0
		num_batches = 0

		#load the model from checkpoint file
		load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
		print("Loading the pretrained model from: " + str(load_as))
		saver.restore(sess, load_as)


		#start evaluating each batch of test data
		batch_count = int(len(test_tweets) / (FLAGS.batch_size*FLAGS.tweet_per_user))

		print("Testing...")
		for batch in range(batch_count):

			#prepare the batch
			test_batch_x, test_batch_y = prepCharBatchData_tweet(test_tweets, test_users, target_values, batch)
			test_batch_x = char2id_tweet(test_batch_x, vocabulary)

			#run the graph
			feed_dict = {network.input_x: test_batch_x, network.input_y: test_batch_y, network.reg_param: FLAGS.l2_reg_lambda}
			loss, prediction, accuracy = sess.run([network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)

			#calculate the metrics
			batch_loss += loss
			batch_accuracy += accuracy
			num_batches += 1

			for i in range(len(prediction)):
				try:
					score = user_pred[test_users[i + (batch * 100)]]
					score[0] += prediction[i][0]
					score[1] += prediction[i][1]
					user_pred[test_users[i + (batch * 100)]] = score
				except:
					user_pred[test_users[i + (batch * 100)]] = prediction[i]

		for key, value in user_pred.items():
			count += 1
			if np.argmax(value) == np.argmax(target_values[key]):
				acc += 1
		print("number of users: " + str(count))
		print("user level accuracy:" + str(float(acc) / count))

		#print the accuracy and progress of the validation
		batch_accuracy /= (batch_count-1)
		print("Test loss: " + "{0:5.4f}".format(batch_loss))
		print("Test accuracy: " + "{0:0.5f}".format(batch_accuracy))






#main function for standalone runs
if __name__ == "__main__":

	print("---PREPROCESSING STARTED---")

	print("\treading word embeddings...")
	vocabulary, embeddings = readGloveEmbeddings(FLAGS.word_embed_path, FLAGS.word_embedding_size)

	print("\treading tweets...")
	tweets, users, target_values, seq_lengths = readData(FLAGS.test_data_path)

	print("\tconstructing datasets and network...")
	tf.reset_default_graph()
	net = network(embeddings)

	print("---TESTING STARTED---")
	test(net, tweets, users, seq_lengths, target_values, vocabulary, embeddings)




