#!/usr/bin/python
#encoding=utf8

import config
import logging
import datetime
from mlscikit.tree.tree import DecisionTreeClassifier

start_time = datetime.datetime.now()
FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('start program-----------------')
train_file_path = config.train_data_path
test_file_path = config.test_data_path
train_data_features = []
train_data_target = []
test_data_features = []
test_data_target = []

# get train data from file
for line in open(train_file_path, 'r'):
    split_line = line.rstrip('\n').split(',')
    for i in range(1, len(split_line)):
        split_line[i] = int(split_line[i])
    train_data_features.append(split_line[1:])
    train_data_target.append(split_line[0])

# get test data from file
for line in open(test_file_path, 'r'):
    split_line = line.rstrip('\n').split(',')
    for i in range(1, len(split_line)):
        split_line[i] = int(split_line[i])
    test_data_features.append(split_line[1:])
    test_data_target.append(split_line[0])

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
all_count = len(test_data_prediction)
for i in range(all_count):
    if test_data_prediction[i] == test_data_target[i]:
        correct_count += 1
logging.info("precision %d / %d = %.2f%%" % (correct_count, all_count, 1.0 * correct_count / all_count * 100))
end_time = datetime.datetime.now()
logging.info('total running time: %.2f second' % (end_time - start_time).seconds)
logging.info('end program-----------------')