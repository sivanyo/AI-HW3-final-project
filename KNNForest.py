from utils import load_data
from ID3 import ID3
from utils import SICK
from utils import HEALTHY
from utils import euclidean_distance
from utils import calc_centroid
from random import sample
from sklearn.model_selection import KFold


def find_KNN_examples(data, example, k_param):
    distance_list = []
    for i in range(len(data)):
        e_distance = euclidean_distance(example, data[i][0])
        distance_list.append((e_distance, data[i][1]))
    distance_list.sort(key=lambda x: x[0])
    nearest = []
    for i in range(k_param):
        nearest.append(distance_list[i])
    return nearest


class KNNForest:
    def __init__(self, n_param):
        self.n_param = n_param
        self.decision_trees = []

    def train(self, data, p_param):
        # kf = KFold(n_splits=3, shuffle=True, random_state=318981586)
        decisions_trees = []

        for i in range(self.n_param):
            size = p_param*self.n_param
            random_examples = sample(data, k=int(size))
            classifier = ID3(random_examples)
            classifier.train()
            centroid = calc_centroid(random_examples)
            decisions_trees.append((centroid, classifier))
        self.decision_trees = decisions_trees

    def classify_example(self, example, k_param):
        features_values = []
        for i in range(len(example)):
            features_values.append(example[i])
        # need to choose the k nearest classifiers

        k_decisions_tree = find_KNN_examples(self.decision_trees, example, k_param)
        sick_num, healthy_num = 0, 0
        for i in range(k_param):
            classification = k_decisions_tree[i][1].root.find_class_by_example(example)
            if classification is SICK:
                sick_num += 1
            else:
                healthy_num += 1
        if sick_num > healthy_num:
            return SICK
        return HEALTHY

    def test(self, test_data, k_param):
        right = 0
        for i in range(len(test_data)):
            if self.classify_example(test_data[i], k_param) is test_data[i][0]:
                right += 1
        # print(right / (len(test_data)))
        return right / (len(test_data))


def experiment(train_data, test_data):
    n_params = []
    for i in range(30, 100, 10):
        n_params.append(i)
    p_params = [0.3, 0.4, 0.5, 0.6, 0.7]
    k_params = []
    for i in range(3, 99, 15):
        k_params.append(i)
    success_rate = []
    for n in n_params:
        for k in k_params:
            if k >= n:
                break
            for p in p_params:
                print("start run")
                forest = KNNForest(n)
                forest.train(train_data, p)
                accuracy = forest.test(test_data, k)
                success_rate.append((accuracy, (n, k, p)))
    success_rate.sort(key=lambda x: x[0])
    return success_rate[0][-1]


def average_accuracy():
    accuracy_list = []
    data = load_data("train.csv")
    test = load_data("test.csv")
    for i in range(20):
        print("start", i, "run")
        classifier = KNNForest(60)
        classifier.train(data, 0.5)
        accuracy = classifier.test(test, 50)
        accuracy_list.append(accuracy)
        print("accuracy is ", accuracy)
    avg = sum(accuracy_list) / len(accuracy_list)
    print("avg accuracy is:", avg)


def merge(file_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=318981586)
    data = load_data(file_name)
    ac=[]
    for train_index, test_index in kf.split(data):
        train_data, test_data = [], []
        for j in train_index:
            train_data.append(data[j])
        for j in test_index:
            test_data.append(data[j])
        classifier = KNNForest(60)
        classifier.train(train_data, 0.69)
        acc = classifier.test(test_data, 51)
        ac.append(acc)
        #print(acc)
    print(sum(ac) / len(ac))


if __name__ == '__main__':
    # average_accuracy()
    data = load_data("train.csv")
    classifier = KNNForest(60)
    classifier.train(data, 0.69)
    tester = load_data("test.csv")
    accuracy = classifier.test(tester, 51)
    print(accuracy)
    # n, k, p = experiment(data, tester)
    # print(n, k, p)
    """after performing the experiment the best (n, k, p) are (60, 50, 0.69)"""
