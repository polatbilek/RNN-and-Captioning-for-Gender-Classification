from parameters import FLAGS
import tensorflow as tf
from preprocess import *
from model import network


#####################################################################################################################
##loads a model and tests it
#####################################################################################################################
def test(network, target_values):
    saver = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:

        # init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        batch_loss = 0.0
        batch_accuracy = 0.0

        # load the model from checkpoint file
        load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
        print("Loading the pretrained model from: " + str(load_as))
        saver.restore(sess, load_as)

        user_names = target_values.keys()
        user_targets = target_values.values()

        # start evaluating each batch of test data
        batch_count = int(len(user_names) / FLAGS.batch_size)

	#input path
	yolo_vector_path_test = os.path.join(os.path.join(FLAGS.test_data_path, FLAGS.lang), "yolo-vectors")

        prev_index = 0

        for batch in range(batch_count):

            current_index = (batch + 1) * FLAGS.batch_size

            if current_index <= len(user_names):
                batch_users = user_names[prev_index:current_index]
                batch_y = user_targets[prev_index:current_index]
            else:
                batch_users = user_names[prev_index:]
                batch_y = user_targets[prev_index:]

            prev_index = current_index

            batch_x = readVectors(yolo_vector_path_test, batch_users)

            # run the graph
            feed_dict = {network.X: batch_x, network.Y: batch_y, network.reg_param: FLAGS.l2_reg_lambda}
            loss, prediction, accuracy = sess.run([network.loss, network.prediction, network.accuracy],
                                                  feed_dict=feed_dict)

            # calculate the metrics
            batch_loss += loss
            batch_accuracy += accuracy

        # print the accuracy and progress of the validation
        batch_accuracy /= batch_count
        print("Test loss: " + "{0:5.4f}".format(batch_loss))
        print("Test accuracy: " + "{0:0.5f}".format(batch_accuracy))

        # take the logs
        if FLAGS.optimize:
            f = open(FLAGS.log_path, "a")
            f.write("\n---TESTING STARTED---\n")
            f.write("\nwith model:" + load_as + "\n")
            f.write("Test loss: " + "{0:5.4f}".format(batch_loss) + "\n")
            f.write("Test accuracy: " + "{0:0.5f}".format(batch_accuracy) + "\n")
            f.close()


# main function for standalone runs
if __name__ == "__main__":

    print("---PREPROCESSING STARTED---")

    print("\treading tweets...")
    target_values= readData(FLAGS.test_data_path)
    print("\ttest set size: " + str(len(target_values.keys())))

    # finds every model in FLAGS.model_path and runs every single one
    if FLAGS.optimize == True:
        models = os.listdir(FLAGS.model_path)
        for model in models:
            if model.endswith(".ckpt.index"):
                FLAGS.model_name = model[:-6]
                tf.reset_default_graph()
                net = network()
                test(net, target_values)
    # just runs  single model specified in FLAGS.model_path and FLAGS.model_name
    else:
        tf.reset_default_graph()
        net = network()
        test(net, target_values)
