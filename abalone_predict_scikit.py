#!/usr/bin/python
#encoding=utf8

import config
import logging
from sklearn.tree.tree import DecisionTreeClassifier
logging.basicConfig(level=logging.INFO)


train_file_path = config.abalone_predict_train_data_path
test_file_path = config.abalone_predict_test_data_path
train_data_features = []
train_data_target = []
test_data_features = []
test_data_target = []

# get train data from file

logging.info("start reading the training data")


gender_to_int = {"F": 0, "M": 1, "I": 0.5}


for line in open(train_file_path, 'r'):
    split_line = line.rstrip('\n').split(',')
    split_line[0] = float(gender_to_int[split_line[0]])
    for i in range(1, len(split_line)):
        try:
            split_line[i] = float(split_line[i])
        except:
            pass

    train_data_features.append(split_line[0:-1])
    train_data_target.append(split_line[-1])

logging.info("finish reading the training data")
logging.info("start reading the test data")

# get test data from file
for line in open(test_file_path, 'r'):
    split_line = line.rstrip('\n').split(',')
    split_line[0] = float(gender_to_int[split_line[0]])
    for i in range(1, len(split_line)):
        try:
            split_line[i] = float(split_line[i])
        except:
            pass
    test_data_features.append(split_line[0:-1])
    test_data_target.append(split_line[-1])

logging.info("finish reading the test data")

clf = DecisionTreeClassifier()


# pickle_path = 'tree_store.pickle'
# feature_count_path = 'feature_count.pickle'
# force_fit = False
#
# if os.path.exists(pickle_path) and not force_fit:
#     f = open(pickle_path, 'r')
#     clf.tree = pickle.load(f)
#     f.close()
#     feature_f = open(feature_count_path, 'r')
#     clf.n_features = pickle.load(feature_f)
#     feature_f.close()
#
# else:
#     clf.fit(train_data_features, train_data_target)
#     f = open(pickle_path, 'w')
#     pickle.dump(clf.tree, f)
#     f.close()
#
#     feature_f = open(feature_count_path, 'w')
#     pickle.dump(clf.n_features, feature_f)
#     feature_f.close()

clf.fit(train_data_features, train_data_target)
test_data_prediction = clf.predict(test_data_features)

all_count = len(test_data_prediction)
variance = 0
for i in range(all_count):
    variance += abs(test_data_prediction[i] - test_data_target[i])

print "variance %d / %d = %f" % (variance, all_count, 1.0 * variance / all_count)
