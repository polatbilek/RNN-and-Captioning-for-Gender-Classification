import os
from parameters import FLAGS

logs = open(FLAGS.log_path, "r")

parse = False
model_name = ""
models_with_tweet_accuracy = {}
models_with_user_accuracy = {}
count = 0
###############
# models_with_accuracy is a dict of list
# list's index 0 represents tweet level accuracy, index 1 user level accuracy
for line in logs:

    if line == "---TESTING STARTED---" and  count == 0:
        count += 1

    elif count == 1:
        model_name = line.strip().split("/")[-1]
        count += 1

    elif count == 2:
        count += 1
    elif count == 3:
        tweet_accuracy = float(line.strip().split(": ")[-1])
    elif count == 4:
        count +=1
    elif count == 5:
        user_accuracy = float(line.strip().split(":")[-1])
        models_with_tweet_accuracy[model_name] = tweet_accuracy
        models_with_user_accuracy[model_name] = user_accuracy
        count = 0


tweet_sorted_models = {}
user_sorted_models = {}

tweet_sorted_list = sorted(models_with_tweet_accuracy.values())
user_sorted_list = sorted(models_with_user_accuracy.values())

for sortedKey in tweet_sorted_list:
    for key, value in models_with_tweet_accuracy.items():
        if value == sortedKey:
            tweet_sorted_models[key] = value


for sortedKey in user_sorted_list:
    for key, value in models_with_user_accuracy.items():
        if value == sortedKey:
            user_sorted_list[key] = value


remove_threshold = 3
i = 0
deletion_list = []
print("there are " + str(len(tweet_sorted_models.values())) + " models after sorted")
# add poor tweet level models to deletion list
if len(tweet_sorted_models.values()) > remove_threshold:
    for key, value in models_with_tweet_accuracy.items():
        if i > (remove_threshold-1):
            deletion_list.append(key)
        i += 1
# remove a good user level model from deletion list
    i = 0
    for key, value in models_with_user_accuracy.items():
        if i <= (remove_threshold-1):
            if key in deletion_list:
                deletion_list.pop(key)
        i += 1

# delete the remaining models
for model in deletion_list:
    fullname = model + ".index"
    os.remove(os.path.join(FLAGS.model_path, fullname))
    fullname = model + ".meta"
    os.remove(os.path.join(FLAGS.model_path, fullname))
    fullname = model + ".data-00000-of-00001"
    os.remove(os.path.join(FLAGS.model_path, fullname))

print("there are " + str(len(os.listdir(FLAGS.model_path))) + " after deletion in the folder")