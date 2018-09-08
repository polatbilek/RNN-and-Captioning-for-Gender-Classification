import os
from parameters import FLAGS
import sys
import operator

logs = open(FLAGS.log_path, "r")


model_name = ""
models_with_tweet_accuracy = {}
models_with_user_accuracy = {}
count = 0


###############
# models_with_accuracy is a dict of list
# list's index 0 represents tweet level accuracy, index 1 user level accuracy
for line in logs:
	if line == "---TESTING STARTED---\n" and  count == 0:
		count += 1
	elif count == 1:
		model_name = line.strip().split("/")[-1]
		count += 1
	elif count == 2:
		count += 1
	elif count == 3:
		tweet_accuracy = float(line.strip().split(": ")[-1])
		count += 1
	elif count == 4:
		count +=1
	elif count == 5:
		user_accuracy = float(line.strip().split(":")[-1])
		models_with_tweet_accuracy[model_name] = tweet_accuracy
		models_with_user_accuracy[model_name] = user_accuracy
		count = 0


#sort the models
tweet_sorted_list = sorted(models_with_tweet_accuracy.items(), key=operator.itemgetter(1))
user_sorted_list = sorted(models_with_user_accuracy.items(), key=operator.itemgetter(1))
print("there are " + str(len(tweet_sorted_list)) + " models after sorted")



#create the list of to be deleted items
remove_threshold = 5
i = 0
deletion_list = []
for i in range(0,len(tweet_sorted_list)-remove_threshold):
	deletion_list.append(tweet_sorted_list[i][0])


#remove from the list the most successful user level models
for i in range(1,remove_threshold+1):
	if user_sorted_list[-i][0] in deletion_list:
		deletion_list.pop(deletion_list.index(user_sorted_list[-i][0]))


# delete the remaining models
for model in deletion_list:
	fullname = model + ".index"
	os.remove(os.path.join(FLAGS.model_path, fullname))
	fullname = model + ".meta"
	os.remove(os.path.join(FLAGS.model_path, fullname))
	fullname = model + ".data-00000-of-00001"
	os.remove(os.path.join(FLAGS.model_path, fullname))


print("there are " + str(len(os.listdir(FLAGS.model_path))) + " after deletion in the folder")









