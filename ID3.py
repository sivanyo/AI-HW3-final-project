import numpy as np
import pandas as pd


# example - e[0]- ex num, e[1]- class, e[2]-e[last] : features results

def is_consistent(examples, class_c):
    for e in examples:
        if e[1] is not class_c:
            return False
    return True


def entropy(self, x, y):
    if x is 0 or y is 0:
        return 0
    sum = x + y
    entropy = ((-x * np.log2(x / sum) / sum) - (-y * np.log2(y / sum) / sum))
    print(entropy)
    return entropy


class ID3():
    def __init__(self):
        self.data_arr = self.load_data("train.csv")
        # maybe need to del some items from here
        self.exampels = self.data_arr[1:-1]
        # print(self.exampels)
        self.features = self.data_arr[0][1:len(self.data_arr[0]) - 2]
        self.test = self.load_data("test.csv")
        self.classes = self.find_classes()

    def load_data(self, filename):
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

    def information_gain(self, examples_group, feature_index):
        group_size = len(examples_group)
        values = []
        for item in examples_group:
            values.append(item[feature_index])
        values.sort()
        final_values = []
        for i in range(len(values) - 1):
            final_values.append((values[i] + values[i + 1]) / 2)

    def select_feature(self, features, examples):
        # ID3
        pass

    def create_sub_tree(self, features, class_c, feature, old_f):
        pass

    def fit(self, examples, features, default, feature):
        if examples is []:
            # means the leaf is empty
            return None, [], default

        class_c = self.majority_class(examples)

        if is_consistent(examples, class_c):
            # means the node is consistent, make it a leaf
            return None, [], class_c

        feature_f = self.select_feature(features, examples)
        new_features = features - [feature_f]

        subtree = self.create_sub_tree(new_features, class_c, feature, features)

        return feature_f, subtree, class_c


if __name__ == '__main__':
    classifier = ID3()
    classifier.load_data("train.csv")
    classifier.majority_class(classifier.exampels)
