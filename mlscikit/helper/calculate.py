__author__ = 'fucus'


def get_label_statistics(label_list):
    """
    get count for every label from label list

    :param label_list:
    :return: a diction key is label, value is the count of label

    Example:
    >>> get_label_statistics(['A', 'B', 'A'])
    {'A': 2, 'B': 1}
    >>> get_label_statistics([])
    {}
    """
    statistics_label = {}
    for label in label_list:
        if label not in statistics_label:
            statistics_label[label] = 1
        else:
            statistics_label[label] += 1
    return statistics_label
def max_in_dic(value_as_number_dic):
    """
    Function to get the max value index, and max value in a diction where the value is all number


    :param value_as_number_dic:
    :return: max value index and max value

    Example:
    >>> max_in_dic({'A': 12, 'B': 1, 'C': 14})
    ('C', 14)
    >>> max_in_dic({'A': 12, 'B': 1, 'C': 1})
    ('A', 12)
    """

    is_init = False
    max_val = 0
    max_key = -1
    for (key, val) in value_as_number_dic.items():
        if not is_init:
            max_val = val
            max_key = key
            is_init = True
        elif val > max_val:
            max_val = val
            max_key = key
    return max_key, max_val


def cal_gini_index(label_statistics_list):
    """
    Function to calculate the gini index from distribution statistics

    :param label_statistics_list:
    :return: the gini index

    Example:

    >>> cal_gini_index([1, 1])
    0.5
    >>> cal_gini_index([2, 2])
    0.5
    >>> cal_gini_index([2])
    0.0
    >>> cal_gini_index([])
    1.0
    """
    count_sum = 0
    for count in label_statistics_list:
        count_sum += count
    gini_index = 1.0
    for count in label_statistics_list:
        gini_index -= (count * 1.0 / count_sum) * (count * 1.0 / count_sum)
    return gini_index


if __name__ == '__main__':
    import doctest
    doctest.testmod()