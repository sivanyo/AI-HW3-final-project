import pandas as pd
import numpy as np

SICK = 'M'
HEALTHY = 'B'


class Node:
    """this class represent a binary tree
    for the decision tree"""

    def __init__(self, m_param):
        self.split_feature = None
        # bigger than equal val -> right, else -> left
        self.split_val = None
        self.right = None
        self.left = None
        self.classification = None
        self.m_param = m_param

    def calc_height(self):
        if self.right is None and self.left is None:
            return 1
        if self.right and self.left:
            return 1 + max(self.right.calc_height(), self.left.calc_height())
        if self.right:
            return 1 + self.right.calc_height()
        return self.left.calc_height()

    def find_class_by_example(self, example):
        if self.classification is None and self.right is None and self.left is None:
            return -1
        if self.classification is not None:
            return self.classification
        elif example[self.split_feature] < self.split_val:
            return self.left.find_class_by_example(example)
        return self.right.find_class_by_example(example)

    def build(self, examples, default, information_gain_func):
        """this function called by train function in ID3
        it builds the decision tree using the information gain function.
        finally, it returns a decision tree in which each node that is not a leaf we have
        feature and split val, and each leaf holds a label (classification)"""
        if len(examples) == 0:
            return
        res, classification = is_consistent(examples)
        if res:
            self.classification = classification
            return
        elif self.m_param is not None and len(examples) < self.m_param:
            self.classification = default
            return
        new_default = majority_class(examples)
        self.split_feature, self.split_val, left, right = information_gain_func(examples)
        self.left = Node(self.m_param)
        self.right = Node(self.m_param)
        if len(left) != 0:
            self.left.build(left, new_default, information_gain_func)
        if len(right) != 0:
            self.right.build(right, new_default, information_gain_func)

    def find_features(self, features_num):
        relevant = []
        for index in range(1, features_num):
            if self.is_exist(index):
                relevant.append(index)
        return relevant

    def is_exist(self, index):
        if self.split_feature == index:
            return True
        if self.right is not None and self.left is not None:
            return self.right.find_features(index) or self.left.find_features(index)
        if self.right is not None:
            return self.right.find_features(index)
        if self.left is not None:
            return self.left.find_features(index)
        return False


def load_data(filename):
    """this function gets a csv file name as a string and return a list of lists,
    each inner list represent a line in the file
    """
    data = pd.read_csv(filename, sep=',')
    return data.values.tolist()


"""Basic ID3 functions"""


def majority_class(examples_group):
    """this function gets a set of examples and returns the most common label
    in this set"""
    class_sick, class_health = [], []
    for i in range(len(examples_group)):
        if examples_group[i][0] is SICK:
            class_sick.append(examples_group[i])
        else:
            class_health.append(examples_group[i])
    if len(class_sick) >= len(class_health):
        return SICK
    return HEALTHY


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
    return entropy


def information_gain(examples_group):
    """maxIG iff min entropy
    this function calculate the best IG for node and return the selected feature,
     the split val, and the two new sets of examples"""
    size = len(examples_group)
    min_entro = float('inf')
    selected_feature = -1
    higher_final, lower_final = [], []
    split_val = None
    for i in range(1, len(examples_group[0])):
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
        final_values.sort()
        tmp_h, tmp_l = [], []
        for j in range(len(final_values)):
            lower, higher = [], []
            for ex in examples_group:
                if float(ex[i]) < final_values[j]:
                    lower.append(ex)
                else:
                    higher.append(ex)
            lower_sick, lower_healthy = [], []
            for ex in lower:
                if ex[0] == SICK:
                    lower_sick.append(ex)
                else:
                    lower_healthy.append(ex)

            lower_entropy = entropy(len(lower_sick), len(lower_healthy))

            higher_sick, higher_healthy = [], []
            for ex in higher:
                if ex[0] == SICK:
                    higher_sick.append(ex)
                else:
                    higher_healthy.append(ex)
            higher_entropy = entropy(len(higher_healthy), len(higher_sick))
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
    loss = (0.1 * FP + FN) / tester_size
    return loss


"""Basic KNN functions"""


def euclidean_distance(example_1, example_2):
    squares_sum = 0.0
    for i in range(1, len(example_2)):
        squares_sum += ((example_1[i]) - float(example_2[i])) ** 2
    e_distance = np.sqrt(squares_sum)
    return e_distance


def euclidean_distance_for_improved(example_1, example_2, min_max_tupple):
    squares_sum = 0.0
    for i in range(1, len(example_2)):
        if example_2[i] != 0.0:
            normalized_val = minmax_normalization(example_1[i], min_max_tupple[i+1][0], min_max_tupple[i+1][1])
            squares_sum += (normalized_val - float(example_2[i])) ** 2
    e_distance = np.sqrt(squares_sum)
    return e_distance


def minmax_normalization(value, min_val, max_val):
    normalized_val = float((float(value) - float(min_val)) / (float(max_val) - float(min_val)))
    return normalized_val


def calc_centroid(examples):
    centroid = []
    size = len(examples)
    for i in range(1, len(examples[0])):
        sum = 0.0
        for j in range(len(examples)):
            sum += examples[j][i]
        average = sum / size
        centroid.append(average)
    # print(centroid)
    return centroid


def find_KNN_examples(data, example, k_param):
    distance_list = []
    for i in range(len(data)):
        e_distance = euclidean_distance(example, data[i][0])
        height = data[i][1]
        distance_list.append((e_distance, height, data[i]))
    distance_list.sort(key=lambda x: x[0])
    nearest = []
    for i in range(k_param):
        nearest.append(distance_list[i])
    return nearest


"""re-write functions for improved """


def majority_class_for_knn(examples_group):
    """this function gets a set of examples and returns the most common label
    in this set"""
    class_sick, class_healthy = [], []
    for i in range(len(examples_group)):
        if examples_group[i][0] is SICK:
            class_sick.append(examples_group[i])
        else:
            class_healthy.append(examples_group[i])
    if len(class_sick) == len(class_healthy):
        return None
    if len(class_sick) > len(class_healthy):
        return SICK
    return HEALTHY


def information_gain_for_cost_sensitive(examples_group):
    """maxIG iff min entropy
    this function calculate the best IG for node and return the selected feature,
     the split val, and the two new sets of examples"""
    size = len(examples_group)
    min_entro = float('inf')
    selected_feature = -1
    higher_final, lower_final = [], []
    split_val = None
    for i in range(1, len(examples_group[0])):
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
        final_values.sort()
        tmp_h, tmp_l = [], []
        for j in range(len(final_values)):
            lower, higher = [], []
            for ex in examples_group:
                if float(ex[i]) < final_values[j]:
                    lower.append(ex)
                else:
                    higher.append(ex)
            lower_sick, lower_healthy = [], []
            for ex in lower:
                if ex[0] == SICK:
                    lower_sick.append(ex)
                else:
                    lower_healthy.append(ex)

            lower_entropy = entropy(len(lower_sick), len(lower_healthy))

            higher_sick, higher_healthy = [], []
            for ex in higher:
                if ex[0] == SICK:
                    higher_sick.append(ex)
                else:
                    higher_healthy.append(ex)
            higher_entropy = entropy(len(higher_healthy), len(higher_sick))
            tmp_entro = (len(lower) * lower_entropy + len(higher) * higher_entropy) / size

            if tmp_entro < tmp_min_entro:
                tmp_min_entro = tmp_entro
                tmp_split_val = values[j]
                if len(lower) > 0:
                    if majority_class(lower) is SICK:
                        """means that people with less than split val will classify as sick,
                        we want to inc this val a little bit just to be sure"""
                        tmp_split_val = 1.01 * final_values[j]
                    elif majority_class(lower) is HEALTHY:
                        """we will dec this val a little bit just to be sure"""
                        tmp_split_val = 0.99 * final_values[j]
                tmp_h = higher
                tmp_l = lower
        if tmp_min_entro <= min_entro:
            min_entro = tmp_min_entro
            split_val = tmp_split_val
            selected_feature = i
            higher_final = tmp_h
            lower_final = tmp_l
    return selected_feature, split_val, lower_final, higher_final


def information_gain_for_improved_knn(examples_group):
    """maxIG iff min entropy
    this function calculate the best IG for node and return the selected feature,
     the split val, and the two new sets of examples"""
    size = len(examples_group)
    min_entro = float('inf')
    selected_feature = -1
    higher_final, lower_final = [], []
    split_val = None
    for i in range(1, len(examples_group[0])):
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
        final_values.sort()
        tmp_h, tmp_l = [], []
        for j in range(len(final_values)):
            lower, higher = [], []
            for ex in examples_group:
                if float(ex[i]) < final_values[j]:
                    lower.append(ex)
                else:
                    higher.append(ex)
            lower_sick, lower_healthy = [], []
            for ex in lower:
                if ex[0] == SICK:
                    lower_sick.append(ex)
                else:
                    lower_healthy.append(ex)

            lower_entropy = entropy(len(lower_sick), len(lower_healthy))

            higher_sick, higher_healthy = [], []
            for ex in higher:
                if ex[0] == SICK:
                    higher_sick.append(ex)
                else:
                    higher_healthy.append(ex)
            higher_entropy = entropy(len(higher_healthy), len(higher_sick))
            tmp_entro = (len(lower) * lower_entropy + len(higher) * higher_entropy) / size

            if tmp_entro < tmp_min_entro:
                tmp_min_entro = tmp_entro
                tmp_split_val = values[j]
                tmp_h = higher
                tmp_l = lower
        if majority_class_for_knn(tmp_h) is None and majority_class_for_knn(tmp_l) is None:
            if tmp_min_entro <= min_entro:
                min_entro = tmp_min_entro
                split_val = tmp_split_val
                selected_feature = i
                higher_final = tmp_h
                lower_final = tmp_l
    return selected_feature, split_val, lower_final, higher_final


def find_KNN_examples_for_improved(data, example, k_param):
    distance_list = []
    for i in range(len(data)):
        e_distance_for_improved = euclidean_distance_for_improved(example, data[i][0], data[i][3])
        height = data[i][1]
        distance_list.append((e_distance_for_improved, height, data[i]))
    distance_list.sort(key=lambda x: x[0] * 0.9 + x[1] * 0.1)
    nearest = []
    for i in range(k_param):
        nearest.append(distance_list[i])
    return nearest


def find_relevant_features(decision_tree, features_num):
    """this function will go over the tree, and will return all the indexes that will
    effect the decisions"""
    relevant = []

    for i in range(1, features_num):
        if i in decision_tree.find_features(i):
            relevant.append(i)
    return relevant

