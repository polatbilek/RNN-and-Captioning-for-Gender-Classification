from parameters import FLAGS
import preprocess



if __name__ == "__main__":
    tweets, target_values, sequence_lengths = preprocess.readData(FLAGS.training_data_path)

    for tweet in tweets:
        print(tweet)


