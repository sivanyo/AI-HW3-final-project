import numpy as np
from utils import load_data
from utils import majority_class
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt


# example e[0]- class, e[1]-e[last] : features results, e[feature_index] -> e[feature_index+2]

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


def experiment(file_name):
    """this function is the experiment
    only need to add a call at the main function - experiment("train.csv")
    and it will run :)
    """
    # need to add the k-fold issue
    m_params_list = [0, 5, 10, 20, 25, 30, 40, 50, 100, 120, 130, 150, 175, 200, 250]
    successes_rate = []
    kf = KFold(n_splits=5, shuffle=True, random_state=318981586)
    data = load_data(file_name)
    for i in range(len(m_params_list)):
        accuracy = []
        for train_index, test_index in kf.split(data):
            train_data, test_data = [], []
            for j in train_index:
                train_data.append(data[j])
            for j in test_index:
                test_data.append(data[j])
            classifier_t = ID3(train_data, m_params_list[i])
            classifier_t.train()
            success_rate = classifier_t.test(test_data)
            # print(success_rate)
            accuracy.append(success_rate)
        successes_rate.append(sum(accuracy) / len(accuracy))
    plt.plot(m_params_list, successes_rate)
    plt.xlabel("M parameter")
    plt.ylabel("successes rate")
    plt.show()


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


class ID3:

    def __init__(self, data_arr, m_param=None):
        self.data_arr = data_arr
        # print(self.data_arr)
        self.examples = self.data_arr[1:]
        # self.test = None
        self.classes = self.find_classes()
        self.root = Node(self.classes[0], m_param)
        self.num_of_features = len(self.examples[0]) - 2
        # print("start your training, good luck")

    def find_classes(self):
        classes = []
        for item in self.data_arr:
            if item[0] not in classes:
                classes.append(item[0])
        return classes

    def train(self):
        """this function calls to the function that build the decision tree,
        and saves it in the classifier root"""
        major_class = majority_class(self.examples)
        self.root.build(self.examples, major_class)

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

    def test(self, test_group):
        """this test receives a set of data, and test the classifier
        it prints the success rate"""
        # print("start your test !!!")
        test_group = test_group
        right, wrong = 0, 0
        for i in range(len(test_group)):
            if self.root.find_class_by_example(test_group[i]) == test_group[i][0]:
                right += 1
        # print("your success rate is:")
        print(right * 100 / (len(test_group)))  # todo : this is the only print that need to appear
        # print("out of", len(tester), "you are right about", right)
        return right * 100 / (len(test_group))

    def test_by_loss(self, test_group):
        """M - sick, B - healthy
        fp += 1 iff test_group[i][0] = B and classify(test_group[i] = M
        fn += 1 iff test_group[i][0] = M and classify(test+group[i] = B"""
        # print("im here")
        test_group = test_group
        FP, FN = 0, 0
        size = len(test_group)
        for i in range(size):
            if self.root.find_class_by_example(test_group[i]) == 'M' and test_group[i][0] == 'B':
                FP += 1
            elif self.root.find_class_by_example(test_group[i]) == 'B' and test_group[i][0] == 'M':
                FN += 1
        loss = lost(FP, FN, size)
        # print(FP, FN)
        #print(loss)
        return loss


if __name__ == '__main__':
    data = load_data("train.csv")
    classifier = ID3(data)
    classifier.train()
    tester = load_data("test.csv")
    classifier.test(tester)
    # loss = classifier.test_by_loss(tester)
    # print(loss)
    # print("now trying to minimize loss")
    # minimize_loss("train.csv")


    # experiment("train.csv")

    # classifier = ID3("train.csv")
    # classifier.train()
    # fileName = "test.csv"
    # classifier.test(fileName)
