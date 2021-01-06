import numpy as np
import pandas as pd


# example e[0]- class, e[1]-e[last] : features results, e[feature_index] -> e[feature_index+2]

def is_consistent(examples):

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
    if x is 0 or y is 0:
        return 0
    sum = x + y
    entropy = (((-x/sum) * np.log2(x / sum)) - ((y/sum) * np.log2(y / sum)))
    # print(entropy)
    # print(entropy)
    return entropy


def information_gain(examples_group, class_c):
    # maxIG iff min entro
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
            #print(j)
            lower, higher = [], []
            for ex in examples_group:
                #print(ex[i]< final_values[j])
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
            # print(tmp_entro)
            if tmp_entro < tmp_min_entro:
                tmp_min_entro = tmp_entro
                tmp_split_val = final_values[j]
                tmp_h = higher
                tmp_l = lower
        # print("tmp min entro")
        # print(tmp_min_entro)
        if tmp_min_entro < min_entro:
            min_entro = tmp_min_entro
            split_val = tmp_split_val
            selected_feature = i
            higher_final = tmp_h
            lower_final = tmp_l
    return selected_feature, split_val, lower_final, higher_final


def split(feature_index, val, example_group):
    left, right = [], []
    # print(feature_index)
    for ex in example_group:
        if ex[feature_index] < val:
            left.append(ex)
        else:
            right.append(ex)
    return left, right


class Node:
    def __init__(self, one_class_type):
        self.split_feature = None
        # bigger than val -> right, else -> left
        self.split_val = None
        self.right = None
        self.left = None
        self.one_class_type = one_class_type
        self.classification = None

    def find_class_by_example(self, example):
        if self.classification is None and self.right is None and self.right is None:
            print("didn't find classification")
            return -1
        if self.classification is not None:
            return self.classification
        elif example[self.split_feature] < self.split_val:
            return self.left.find_class_by_example(example)
        return self.right.find_class_by_example(example)

    def build(self, examples):
        if len(examples) == 0:
            print("empty node")
            return
        res, classification = is_consistent(examples)

        if res:
            # print("consistent node")
            self.classification = classification
            return

        self.split_feature, self.split_val, left, right = information_gain(examples, self.one_class_type)
        self.left = Node(self.one_class_type)
        self.right = Node(self.one_class_type)
        if len(left) != 0:
            self.left.build(left)
        if len(right) != 0:
            self.right.build(right)


class ID3:

    def __init__(self):
        self.data_arr = self.load_data('train.csv')
        # print(self.data_arr)
        self.examples = self.data_arr[1:]
        # self.test = None
        self.classes = self.find_classes()
        self.root = Node(self.classes[0])
        self.num_of_features = len(self.examples[0]) - 2
        print("start your training, good luck")

    @staticmethod
    def load_data(filename):
        data = pd.read_csv(filename, sep=',')
        return data.values.tolist()

    def find_classes(self):
        classes = []
        for item in self.data_arr:
            if item[0] not in classes:
                classes.append(item[0])
        return classes

    def train(self):
        self.root.build(self.examples)

    def majority_class(self, examples):
        class_dict = {}
        for item in self.classes:
            class_dict[item] = 0
        for e in examples:
            class_dict[e[0]] += 1
        max_value = max(class_dict.values())
        majority_class_res = -1
        for key in class_dict.keys():
            if class_dict[key] is max_value:
                majority_class_res = key
        # print(majority_class_res)
        return majority_class_res

    def test(self, train_filename):
        print("start your test !!! test score:")
        tester = self.load_data(train_filename)
        examples = tester[1:]
        right, wrong = 0, 0
        for i in range(len(examples)):
            if self.root.find_class_by_example(examples[i]) is tester[i + 1][0]:
                right += 1
            else:
                wrong += 1

        print(right * 100 / (right + wrong))


if __name__ == '__main__':
    classifier = ID3()
    classifier.train()
    fileName = "test.csv"
    classifier.test(fileName)
