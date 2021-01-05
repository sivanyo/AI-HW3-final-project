import numpy as np
import pandas as pd

# example - e[0]- ex num, e[1]- class, e[2]-e[last] : features results, e[feature_index] -> e[feature_index+2]


def is_consistent(examples, class_c):
    for e in examples:
        if e[1] is not class_c:
            return False
    return True


def entropy(x, y):
    if x is 0 or y is 0:
        return 0
    sum = x + y
    entropy = ((-x * np.log2(x / sum) / sum) - (-y * np.log2(y / sum) / sum))
    #print(entropy)
    return entropy


def information_gain(examples_group, feature_index, class_c):
    values = []
    for item in examples_group:

        values.append(float(item[feature_index+2]))
    values = np.unique(values)
    # print(values)
    final_values = []
    for i in range(len(values) - 1):
        if values[i] is not values[i + 1]:
            final_values.append((float(values[i]) + float(values[i + 1])) / 2)
    best_val = -1
    min_entropy = 0
    for val in final_values:
        min_e = float('-inf')
        min_val = -1
        lower, bigger = [], []
        for ex in examples_group:
            if float(ex[feature_index + 2]) <= val:
                lower.append(ex)
            else:
                bigger.append(ex)
            # print(lower)
            # print(bigger)
            # lower = examples_group[float(ex[feature_index+2]) <= val]
            # bigger = examples_group[float(ex[feature_index+2]) > val]
        #print(lower)
        #print(bigger)
        # calculate lower entropy
        lower_type_1, lower_type_2 = [], []
        for ex in lower:
            if ex[0] == class_c:
                lower_type_1.append(ex)
            else:
                lower_type_2.append(ex)
        lower_entropy = entropy(len(lower_type_1), len(lower_type_2))

        # calculate bigger entropy
        bigger_type_1, bigger_type_2 = [], []
        for ex in bigger:
            if ex[0] == class_c:
                bigger_type_1.append(ex)
            else:
                bigger_type_2.append(ex)
        bigger_entropy = entropy(len(bigger_type_1), len(bigger_type_2))

        tmp_entropy = (len(lower) * lower_entropy + len(bigger) * bigger_entropy) / (len(lower) + len(bigger))
        if tmp_entropy < min_e:
            min_e = tmp_entropy
            min_val = val
            inner_lower = lower
            inner_bigger = bigger
    if min_e < min_entropy:
        best_val = min_val
        min_entropy = min_e
    return min_entropy, best_val


class Node:
    def __init__(self, split_feature, split_val, left, right):
        self.split_feature = split_feature
        # bigger than val -> right, else -> left
        self.split_val = split_val
        self.right = right
        self.left = left
        self.classification = None

    def find_class_by_example(self, example):
        if self.classification is None and self.right is None and self.right is None:
            print("didn't find classification")
            return -1
        if self.classification is not None:
            return self.classification
        elif example[self.split_feature + 2] <= self.split_val:
            return self.left.find_class_by_example(example)
        return self.right.find_class_by_example(example)


class ID3:
    def __init__(self):
        self.data_arr = self.load_data('train.csv')
        # maybe need to del some items from here
        self.exampels = self.data_arr[1:-1]
        # print(self.exampels)
        # print(self.exampels)
        self.features = self.data_arr[0][1:len(self.data_arr[0]) - 2]
        # print(len(self.features))
        # print(self.features)
        # print(self.features)
        self.test = None
        self.classes = self.find_classes()
        root = Node(None, None, None, None)
        self.tree = self.fit(self.exampels, None, None, root)

    @staticmethod
    def load_data(filename):
        data = pd.read_csv(filename)
        # print(data)
        string_data = pd.DataFrame.to_string(data)
        list_results_tmp = string_data.splitlines()
        data_arr = []
        for line in list_results_tmp:
            tmp = line.split(',')
            list_data = tmp[0].split()
            data_arr.append(list_data)
        return data_arr
        # print(np.loadtxt(filename, delimiter=',', skiprows=1))

    def find_classes(self):
        classes = []
        for item in self.exampels:
            if item[1] not in classes:
                classes.append(item[1])
        # print(classes)
        return classes

    def majority_class(self, examples):
        class_dict = {}
        for item in self.classes:
            class_dict[item] = 0
        for e in examples:
            class_dict[e[1]] += 1
        max_value = max(class_dict.values())
        majority_class_res = -1
        for key in class_dict.keys():
            if class_dict[key] is max_value:
                majority_class_res = key
        # print(majority_class_res)
        return majority_class_res

    def select_feature(self, features, examples, class_c):
        feature_index = -1
        max_IG = 0.0
        split_val = None
        for i in range(len(features)):
            tmp_IG = information_gain(examples, i, class_c)

            if tmp_IG[0] < max_IG:
                max_IG = tmp_IG[0]
                feature_index = i
                split_val = tmp_IG[1]
            elif tmp_IG == max_IG:
                if i > feature_index:
                    feature_index = i
                    split_val = tmp_IG[1]
        return feature_index, split_val

    def create_sub_tree(self, features, class_c, feature, old_f):
        pass

    def fit(self, examples, default, feature, node):
        if examples is []:
            # means the leaf is empty
            node.classification = default
            return node

        class_c = self.majority_class(examples)

        if is_consistent(examples, class_c):
            # means the node is consistent, make it a leaf
            node.classification = class_c
            return node

        feature_f = self.select_feature(self.features, examples, class_c)
        node.split_val = feature_f[1]
        node.split_feature = feature_f[0]
        # new_features = []
        # for f in features:
        #     if f is not feature_f[0]:
        #         new_features.append(f)
        feature_val = feature_f[1]
        left, right = self.split(feature_f, feature_val, examples)

        default_left = self.majority_class(left)
        left_node = Node(feature_f, feature_val, None, None)
        node.left = self.fit(left, default_left, None, left_node)

        default_right = self.majority_class(right)
        right_node = Node(feature_f, feature_val, None, None)
        node.right = self.fit(right, default_right, None, right_node)

        return node

    def split(self, feature_index, val, example_group):
        left, right = [], []
        for ex in example_group:
            if ex[feature_index + 2] < val:
                left.append(ex)
            else:
                right.append(ex)
        return left, right

    def test(self, train_filename):
        self.test = self.load_data(train_filename)
        examples = self.test[1:-1]
        right, wrong = 0, 0
        for i in range(len(examples)):
            if self.tree.find_class_by_example(examples[i]) is self.test[i + 1][0]:
                right += 1
            else:
                wrong += 1
        print(right / right + wrong)


if __name__ == '__main__':
    classifier = ID3()
    classifier.test("test.csv")
