import pandas as pd
import numpy as np

SICK = 'M'
HEALTHY = 'B'


def euclidean_distance(example_1, example_2):
    squares_sum = 0.0
    for i in range(1, len(example_2)):
        # print(i)
        squares_sum += ((example_1[i]) - float(example_2[i])) ** 2
    e_distance = np.sqrt(squares_sum)
    return e_distance


def load_data(filename):
    """this function gets a csv file name as a string and return a list of lists,
    each inner list represent a line in the file
    """
    data = pd.read_csv(filename, sep=',')
    return data.values.tolist()


def majority_class(examples_group):
    """this function gets a set of examples and returns the most common label
    in this set"""
    # print(examples_group)
    class_1_type = examples_group[0][0]
    class_1, class_2 = [examples_group[0]], []
    for i in range(1, len(examples_group)):
        if examples_group[i][0] is class_1_type:
            class_1.append(examples_group[i])
        else:
            class_2.append(examples_group[i])
    if len(class_1) >= len(class_2):
        return class_1_type
    return class_2[0][0]


def is_consistent(examples):
    """this function receives a set of examples and determine if they are all with
    the same label, if they are it returns the label"""
    if len(examples) == 1:
        example = examples[0]
        class_c = example[1]
        # print("got here")
        return True, class_c
    example1 = examples[0]
    class_c = example1[0]
    for e in examples:
        if e[0] is not class_c:
            return False, None
    return True, class_c


def entropy(x, y):
    """this function calculate the entropy of two integers"""
    if x is 0 or y is 0:
        return 0
    sum = x + y
    entropy = (((-x / sum) * np.log2(x / sum)) - ((y / sum) * np.log2(y / sum)))
    # print(entropy)
    # print(entropy)
    return entropy


def information_gain(examples_group, class_c):
    """maxIG iff min entropy
    this function calculate the best IG for node and return the selected feature,
     the split val, and the two new sets of examples"""
    size = len(examples_group)
    min_entro = float('inf')
    selected_feature = -1
    higher_final, lower_final = [], []
    split_val = None
    for i in range(1, len(examples_group[0])):
        # print(i)
        tmp_min_entro = float('inf')
        tmp_split_val = None
        values = []
        for item in examples_group:
            if float(item[i]) not in values:
                values.append(float(item[i]))
        values = np.unique(values)

        final_values = []
        for j in range(len(values) - 1):
            if values[j] is not values[j + 1]:
                final_values.append((float(values[j]) + float(values[j + 1])) / 2)
        tmp_h, tmp_l = [], []
        for j in range(len(final_values)):
            lower, higher = [], []
            for ex in examples_group:
                if float(ex[i]) < final_values[j]:
                    lower.append(ex)
                else:
                    higher.append(ex)
            lower_type_1, lower_type_2 = [], []
            for ex in lower:
                if ex[0] == class_c:
                    lower_type_1.append(ex)
                else:
                    lower_type_2.append(ex)

            lower_entropy = entropy(len(lower_type_1), len(lower_type_2))

            higher_type_1, higher_type_2 = [], []
            for ex in higher:
                if ex[0] == class_c:
                    higher_type_1.append(ex)
                else:
                    higher_type_2.append(ex)
            higher_entropy = entropy(len(higher_type_1), len(higher_type_2))
            tmp_entro = (len(lower) * lower_entropy + len(higher) * higher_entropy) / size
            if tmp_entro < tmp_min_entro:
                tmp_min_entro = tmp_entro
                tmp_split_val = final_values[j]
                tmp_h = higher
                tmp_l = lower
        if tmp_min_entro <= min_entro:
            min_entro = tmp_min_entro
            split_val = tmp_split_val
            selected_feature = i
            higher_final = tmp_h
            lower_final = tmp_l
    return selected_feature, split_val, lower_final, higher_final


def lost(FP, FN, tester_size):
    # print("im here")
    loss = (0.1 * FP + FN) / tester_size
    return loss


def find_KNN_examples(data, example, k_param):
    KNN_list = []
    distance_list = []
    for i in range(len(data)):
        e_distance = euclidean_distance(example, data[i][0])
        #print(e_distance)
        distance_list.append((e_distance, data[i]))
    distance_list.sort(key=lambda x: x[0])
    # print(distance_list)
    nearest = []
    for i in range(k_param):
        nearest.append(distance_list[i])
    return nearest


class Node:
    """this class represent a binary tree
    for the decision tree"""

    def __init__(self, one_class_type, m_param):
        self.split_feature = None
        # bigger than equal val -> right, else -> left
        self.split_val = None
        self.right = None
        self.left = None
        self.one_class_type = one_class_type
        self.classification = None
        self.m_param = m_param

    def find_class_by_example(self, example):
        if self.classification is None and self.right is None and self.right is None:
            # print("didn't find classification")
            return -1
        if self.classification is not None:
            return self.classification
        elif example[self.split_feature] < self.split_val:
            return self.left.find_class_by_example(example)
        return self.right.find_class_by_example(example)

    def build(self, examples, default):
        """this function called by train function in ID3
        it builds the decision tree using the information gain function.
        finally, it returns a decision tree in which each node that is not a leaf we have
        feature and split val, and each leaf holds a label (classification)"""
        if len(examples) == 0:
            # print("empty node")
            return
        res, classification = is_consistent(examples)
        if res:
            # print("consistent node")
            self.classification = classification
            return
        elif self.m_param is not None and len(examples) < self.m_param:
            # print(majority_class(examples))
            self.classification = default
            return
        new_default = majority_class(examples)
        self.split_feature, self.split_val, left, right = information_gain(examples, self.one_class_type)
        self.left = Node(self.one_class_type, self.m_param)
        self.right = Node(self.one_class_type, self.m_param)
        if len(left) != 0:
            self.left.build(left, new_default)
        if len(right) != 0:
            self.right.build(right, new_default)



