from utils import is_consistent
from utils import majority_class
from utils import entropy
from utils import SICK
import numpy as np
from utils import information_gain_for_cost_sensitive


def IG_for_feature(examples_group, feature_index):
    size = len(examples_group)
    split_val = None
    values = []
    for item in examples_group:
        if float(item[feature_index]) not in values:
            values.append(float(item[feature_index]))
    values = np.unique(values)
    final_values = []
    for j in range(len(values) - 1):
        if values[j] is not values[j + 1]:
            final_values.append((float(values[j]) + float(values[j + 1])) / 2)
    final_values.sort()
    tmp_h, tmp_l = [], []
    tmp_min_entro = float('inf')
    for j in range(len(final_values)):
        lower, higher = [], []
        for ex in examples_group:
            if float(ex[feature_index]) < final_values[j]:
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
            split_val = final_values[j]
            tmp_h = higher
            tmp_l = lower

    return split_val, tmp_l, tmp_h

def calc_entropy(A):
    sick, healthy = 0, 0
    for ex in A:
        if ex[0] is SICK:
            sick += 1
        else:
            healthy += 1
    return entropy(sick, healthy)


class BetterNode:
    def __init__(self, m_param):
        self.first_split_feature = None
        self.first_val = None
        self.second_val = None
        self.right = None
        self.middle= None
        self.left = None
        self.splitted = None
        self.classification = None
        self.m_param = m_param

    def find_class_by_example(self, example):
        if self.classification is None and self.right is None and self.left is None:
            return -1
        if self.classification is not None:
            return self.classification
        if self.middle is None:
            if example[self.first_split_feature] < self.first_val:
                return self.left.find_class_by_example(example)
            return self.right.find_class_by_example(example)
        else:
            # there is a middle node, need to split into all options
            if self.splitted == 'left':
                if example[self.first_split_feature] > self.first_val:
                    return self.right.find_class_by_example(example)
                else:
                    if example[self.first_split_feature] < self.second_val:
                        return self.left.find_class_by_example(example)
                    return self.middle.find_class_by_example(example)
            else:
                if example[self.first_split_feature] < self.first_val:
                    return self.left.find_class_by_example(example)
                else:
                    if example[self.first_split_feature] > self.second_val:
                        return self.right.find_class_by_example(example)
                    return self.middle.find_class_by_example(example)

    def build(self, examples, default, information_gain_func, epsilon):
        if len(examples) < self.m_param:
            self.classification = default
            return
        res, classification = is_consistent(examples)
        if res:
            self.classification = classification
            return
        new_dafault = majority_class(examples)
        self.first_split_feature, self.first_val, lower, higher = information_gain_func(examples, epsilon)
        lower_entropy = calc_entropy(lower)
        higher_entropy = calc_entropy(higher)
        if lower_entropy == 0 or higher_entropy == 0:
            self.left = BetterNode(self.m_param)
            self.right = BetterNode(self.m_param)
            if len(lower) != 0:
                self.left.build(lower, new_dafault, information_gain_func, epsilon)
            if len(higher) != 0:
                self.right.build(higher, new_dafault, information_gain_func, epsilon)
            return
        else:
            self.left = BetterNode(self.m_param)
            self.right = BetterNode(self.m_param)
            self.middle = BetterNode(self.m_param)
            if lower_entropy > higher_entropy:
                self.splitted = 'left'
                self.second_val, lower, middle = IG_for_feature(lower, self.first_split_feature)
                if len(lower) != 0:
                    self.left.build(lower, new_dafault, information_gain_func, epsilon)
                if len(higher) != 0:
                    self.right.build(higher, new_dafault, information_gain_func, epsilon)
                if len(middle) != 0:
                    self.middle.build(middle, new_dafault, information_gain_func, epsilon)
                return
            else:
                self.splitted = 'right'
                self.second_val, middle, higher = IG_for_feature(lower, self.first_split_feature)
                if len(lower) != 0:
                    self.left.build(lower, new_dafault, information_gain_func, epsilon)
                if len(higher) != 0:
                    self.right.build(higher, new_dafault, information_gain_func, epsilon)
                if len(middle) != 0:
                    self.middle.build(middle, new_dafault, information_gain_func, epsilon)
                return

