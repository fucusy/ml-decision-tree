#!/usr/bin/python
#encoding=utf8
import sys
import os

if len(sys.argv) < 2:
    print "usage: ./split_data.py {path_of_data_file}"
    exit(0)
train_data_part = 0.8

test_data_part = 1 - train_data_part
data_file_path = sys.argv[1]

if not os.path.exists(data_file_path):
    print "%s doest not exist" % data_file_path


train_data_file_path = '%s_%s_train_part' % (data_file_path, train_data_part)
test_data_file_path = '%s_%s_test_part' % (data_file_path, test_data_part)

if os.path.exists(train_data_file_path):
    temp_count = 1
    while os.path.exists('%s_%d' %(train_data_file_path, temp_count)):
        temp_count += 1
    train_data_file_path = '%s_%d' %(train_data_file_path, temp_count)

train_data_file = open(train_data_file_path, 'w')


if os.path.exists(test_data_file_path):
    temp_count = 1
    while os.path.exists('%s_%d' %(test_data_file_path, temp_count)):
        temp_count += 1
    test_data_file_path = '%s_%d' %(test_data_file_path, temp_count)

test_data_file = open(test_data_file_path, 'w')
line_count = 0
for line in open(data_file_path, 'r'):
    line_count += 1
train_data_count = line_count * train_data_part

temp_count = 0
for line in open(data_file_path, 'r'):
    temp_count += 1
    if temp_count <= train_data_count:
        train_data_file.write(line)
    else:
        test_data_file.write(line)

train_data_file.close()
test_data_file.close()







