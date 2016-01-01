#!/usr/bin/python
#encoding=utf8

import config
import logging
from mlscikit.tree.tree import DecisionTreeClassifier


FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('start program---------------------')

train_file_path = config.abalone_predict_train_data_path
test_file_path = config.abalone_predict_test_data_path
train_data_features = []
train_data_target = []
test_data_features = []
test_data_target = []

# get train data from file

logging.info("start reading the training data")

for line in open(train_file_path, 'r'):
    split_line = line.rstrip('\n').split(',')
    for i in range(len(split_line)):
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
    for i in range(len(split_line)):
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

correct_count = 0
variance = 0
all_count = len(test_data_prediction)
for i in range(all_count):
    variance += abs(test_data_prediction[i] - test_data_target[i])

print "variance %d / %d = %f" % (variance, all_count, 1.0 * variance / all_count)

logging.info('end program---------------------')