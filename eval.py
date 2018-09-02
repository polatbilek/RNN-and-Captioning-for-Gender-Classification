from parameters import FLAGS
import tensorflow as tf
from preprocess import *
from model import network





#####################################################################################################################
##loads a model and tests it
#####################################################################################################################
def test(network, test_tweets, test_users, test_seq_lengths, target_values, vocabulary, embeddings):
	
	saver = tf.train.Saver(max_to_keep=None)

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

		#load the model from checkpoint file
		load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
		print("Loading the pretrained model from: " + str(load_as))
		saver.restore(sess, load_as)


		#start evaluating each batch of test data
		batch_count = int(len(test_tweets) / (FLAGS.batch_size*FLAGS.tweet_per_user))

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

			for i in range(len(prediction)):
				try:
					score = user_pred[test_users[i + (batch * 100)]]
					score[0] += prediction[i][0]
					score[1] += prediction[i][1]
					user_pred[test_users[i + (batch * 100)]] = score
				except:
					user_pred[test_users[i + (batch * 100)]] = prediction[i]

		#calculate user level accuracy
		for key, value in user_pred.items():
			count += 1
			if np.argmax(value) == np.argmax(target_values[key]):
				acc += 1

		#print the accuracy and progress of the test
		batch_accuracy /= batch_count
		print("Tweet level test loss: " + "{0:5.4f}".format(batch_loss))
		print("Tweet level test accuracy: " + "{0:0.5f}".format(batch_accuracy))
		print("number of users: " + str(count))
		print("user level accuracy:" + str(float(acc) / count))

		#take the logs
		if FLAGS.optimize:
			f = open(FLAGS.log_path, "a")
			f.write("\nwith model:" + load_as + "\n")
			f.write("Test loss: " + "{0:5.4f}".format(batch_loss) + "\n")
			f.write("Test accuracy: " + "{0:0.5f}".format(batch_accuracy) + "\n")
			f.write("Number of users: " + str(count) + "\n")
			f.write("User level test accuracy:" + str(float(acc)/count) + "\n")
			f.close()









#main function for standalone runs
if __name__ == "__main__":

	print("---PREPROCESSING STARTED---")

	print("\treading char embeddings...")
	vocabulary, embeddings = readCharEmbeddings(FLAGS.char_embed_path, FLAGS.char_embedding_size)

	print("\treading tweets...")
	tweets, users, target_values, seq_lengths = readData(FLAGS.test_data_path)
	print("\ttest set size: " + str(len(tweets)))


	print("---TESTING STARTED---")
	#finds every model in FLAGS.model_path and runs every single one
	if FLAGS.optimize == True:
		models = os.listdir(FLAGS.model_path)
		for model in models:
			if model.endswith(".ckpt.index"):
				FLAGS.model_name = model[:-6]
				tf.reset_default_graph()
				net = network(embeddings)
				test(net, tweets, users, seq_lengths, target_values, vocabulary, embeddings)
	#just runs  single model specified in FLAGS.model_path and FLAGS.model_name
	else:
		tf.reset_default_graph()
		net = network(embeddings)
		test(net, tweets, users, seq_lengths, target_values, vocabulary, embeddings)




