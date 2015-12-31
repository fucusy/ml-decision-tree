__author__ = 'fucus'


from ..helper.calculate import cal_gini_index, max_in_dic, get_label_statistics, get_variance
import logging

class Node:
    """node in tree

    """

    def __init__(self
                 ,is_leaf
                 ,class_label
                 ,order
                 ,go_left_params
                 ,is_number=True
                 ):
        """

        is_leaf: the node in binary tree is leaf or not, if it's true, the class_label value is meaningful,
                and other attributes is meaningless, else otherwise
        class_label: it's meaningful when is_leaf is True, it's the class label result of decision tree

        order: the node in binary tree contributed to the {order} of input vector
        go_left_params:
            the {order} of vector value less or equal {go_left_params}, the next level of
            check node is left child node, else is right child node


        """

        self.is_leaf = is_leaf
        self.class_label = class_label
        self.go_left_params = go_left_params
        self.order = order

    def go_left_tree(self, val):
        return Node.go_left(val, self.go_left_params)

    @staticmethod
    def go_left(val, param):
        if type(val) is str and type(param) is str:
            return val == param
        else:
            try:
                return val <= param
            except ValueError, e:
                logging.error("%s" % e)
    def is_leaf(self):
        return self.is_leaf is True




class DepthFirstTreeBuilder:

    @staticmethod
    def build(X, y, n_features, search_range, feature_is_number_list, is_classification=True):
        tree = Tree()
        n_sample = len(X)
        if n_sample != len(y):
                raise ValueError("Number of labels=%d does not match "
                                 "number of samples=%d" % (len(y), n_sample))

        # stop split, get the most class label
        if is_classification:
            label_statistics = get_label_statistics(y)
            label_gini_index = cal_gini_index(label_statistics.values())
            if label_gini_index <= 0.0:
                most_class_label, count = max_in_dic(label_statistics)
                tree.root = Node(is_leaf=True
                                 ,class_label=most_class_label
                                ,order=Node
                                ,go_left_params=Node
                                 )
                return tree
        else:
            variance = get_variance(y)
            if variance <= 0.0:
                if len(y) > 0:
                    most_class_label = y[0]
                else:
                    most_class_label = -1
                tree.root = Node(is_leaf=True
                                 ,class_label=most_class_label
                                ,order=Node
                                ,go_left_params=Node
                                 )
                return tree


        # store the best param
        best_feature = -1
        min_error_val = 100000
        best_param = -1
        X_best_left = []
        X_best_right = []
        Y_best_left = []
        Y_best_right = []

        # todo stop split when the percent of one kind class greater than generate_leaf_param

        # gini index as split condition
        for i in range(0, n_features):
            # for every feature, cal the min val
            feature_min_gini_val = min_error_val
            feature_best_param = -1
            param_X_split_left = []
            param_X_split_right = []
            param_Y_split_left = []
            param_Y_split_right = []

            if feature_is_number_list[i]:
                step = (search_range[i][1] - search_range[i][0]) * 1.0 / 20
                param = search_range[i][0] - step
            else:
                feature_enum_i = -1

            while (feature_is_number_list[i] and param <= search_range[i][1])\
                    or (not feature_is_number_list[i] and feature_enum_i + 1 < len(search_range[i])):

                if feature_is_number_list[i]:
                    param += step
                else:
                    feature_enum_i += 1
                    param = search_range[i][feature_enum_i]

                # split X and Y
                X_split_left = []
                X_split_right = []
                Y_split_left = []
                Y_split_right = []
                for k in range(n_sample):
                    if Node.go_left(X[k][i], param):
                        X_split_left.append(X[k])
                        Y_split_left.append(y[k])
                    else:
                        X_split_right.append(X[k])
                        Y_split_right.append(y[k])

                if is_classification:
                    # cal the gini index
                    left_label_statistics = get_label_statistics(Y_split_left)
                    left_gini_index = cal_gini_index(left_label_statistics.values())

                    right_label_statistics = get_label_statistics(Y_split_right)
                    right_gini_index = cal_gini_index(right_label_statistics.values())

                else:

                    left_gini_index = get_variance(Y_split_left)
                    right_gini_index = get_variance(Y_split_right)

                if n_sample == 0.0:
                    raise ValueError("n_sample is zero")

                gini_index = len(Y_split_left) * 1.0 / n_sample * left_gini_index \
                             + len(Y_split_right) * 1.0 / n_sample * right_gini_index

                if gini_index < feature_min_gini_val:
                    feature_min_gini_val = gini_index
                    feature_best_param = param
                    param_X_split_left = X_split_left
                    param_X_split_right = X_split_right
                    param_Y_split_left = Y_split_left
                    param_Y_split_right = Y_split_right

            # now we got best param for feature {i}
            if feature_min_gini_val < min_error_val:
                best_feature = i
                best_param = feature_best_param
                min_error_val = feature_min_gini_val
                X_best_left = param_X_split_left
                X_best_right = param_X_split_right
                Y_best_left = param_Y_split_left
                Y_best_right = param_Y_split_right


        print "best_feature: %s" % best_feature
        print "param: %s" % best_param
        print "gini index %f" % min_error_val

        # left_statistics = get_label_statistics(Y_best_left)
        # right_statistics = get_label_statistics(Y_best_right)

        # now we got best feature and param to split X
        # todo add gini index, statistics data, samples count to the node

        if not is_classification and (len(Y_best_right) == 0 or len(Y_best_left) == 0):
            if len(y) > 0:
                most_class_label = sum(y) * 1.0 / len(y)
            else:
                most_class_label = -1
            tree.root = Node(is_leaf=True
                             ,class_label=most_class_label
                            ,order=Node
                            ,go_left_params=Node
                             )
            return tree

        tree.root = Node(order=best_feature
                              ,go_left_params=best_param
                              ,is_leaf=False
                              ,class_label=Node
                              )
        tree.left = DepthFirstTreeBuilder.build(X_best_left, Y_best_left, n_features, search_range, feature_is_number_list, is_classification = is_classification)
        tree.right = DepthFirstTreeBuilder.build(X_best_right, Y_best_right, n_features, search_range, feature_is_number_list, is_classification = is_classification)
        return tree

class Tree:
    """
    representation of a binary decision tree.
    """
    def __init__(self):
        self.root = Node
        self.left = Tree
        self.right = Tree
        self.max_depth = 20
    def predict(self, features_list):
        current = self
        while current.root.is_leaf is False:
            if current.root.go_left_tree(features_list[current.root.order]):
                current = current.left
            else:
                current = current.right
        return current.root.class_label

class DecisionTreeClassifier:
    """a decision tree classifier.

    """

    def __init__(self):
        self.tree = Tree
        self.n_features = 0
        self.is_classification = True
        self.feature_is_number_list = []
        self.feature_step_count = 20
        self.feature_step = 1

    def fit(self, X, y):
        """Build  a decision tree from the training set (X, y).

        :param X:
        :param y:
        :return:
        """

        n_samples = len(X)
        self.n_features = len(X[0])
        self.feature_is_number_list = self.n_features * [True]
        search_range = self.n_features * [[]]
        if len(y) > 0 and type(y[0]) is not str:
            self.is_classification = False

        # check the training data
        # todo check feature value type in the same column data is the same
        for i in range(self.n_features):
            if type(X[0][i]) is str:
                self.feature_is_number_list[i] = False
                logging.info("feature %d is str" % i)
            elif type(X[0][i]) is float or type(X[0][i]) is int:
                self.feature_is_number_list[i] = True
                logging.info("feature %d is %s" % (i, type(X[0][i])))
            else:
                raise Exception("feature %d 's type is not in (str, float, int)" % i)

        # scan all the data to find feature value range or value list
        for i in range(self.n_features):

            if self.feature_is_number_list[i]:
                # max val and min val of feature {i}
                search_range[i] = [X[0][i], X[0][i]]
                for n in range(n_samples):
                    if X[n][i] < search_range[i][0]:
                        search_range[i][0] = X[n][i]

                    if X[n][i] > search_range[i][1]:
                        search_range[i][1] = X[n][i]
            else:
                for n in range(n_samples):
                    if X[n][i] not in search_range[i]:
                        search_range[i].append(X[n][i])

        self.tree = DepthFirstTreeBuilder.build(X, y, self.n_features, search_range, self.feature_is_number_list, is_classification = self.is_classification)

    def predict(self, X):
        """Predict class or regression value for X

        :param X:
            X is a feature list or a list of feature list
        :return:
            if X is a feature list, return the predicted class label
            else if X is a list of feature list, return a list of predicted class label
        """
        is_matrix = False
        n_sample = 1

        if type(X[0]) is int:
            is_matrix = False
            n_sample = 1
            feature_count = len(X)
        elif type(X[0]) is list and type(X[0][0] is int):
            is_matrix = True
            n_sample = len(X)
            feature_count = len(X[0])
        else:
            if type(X[0]) is not list:
                feature_type = type(X[0])
            else:
                feature_type = type(X[0][0])
            raise ValueError("the feature value must be integer, "
                             "but the input feature type is %s" % feature_type
                             )

        if feature_count != self.n_features:
            raise ValueError("Number of features of the model must "
                                         "match the input. Model n_features is %s and "
                                         "input n_features is %s "
                                         % (self.n_features, feature_count))

        if is_matrix:
            y = []
            for i in range(n_sample):
                y.append(self.tree.predict(X[i]))
        else:
            y = self.tree.predict(X)
        return y
