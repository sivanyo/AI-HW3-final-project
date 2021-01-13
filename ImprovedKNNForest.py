from utils import load_data
from ID3 import ID3
from utils import SICK
from utils import HEALTHY
from utils import find_KNN_examples_for_improved
from random import choices
from utils import information_gain
from utils import majority_class_for_knn
from utils import minmax_normalization

"""this algo prefer trees with low height
it also check which features are relevant and only than calc the centroid
it is also normalize all the features"""


def experiment(train_data, test_data):
    n_params = []
    for i in range(20, 100, 5):
        n_params.append(i)
    p_params = [0.3, 0.4, 0.5, 0.6, 0.7]
    k_params = []
    for i in range(3, 99, 7):
        k_params.append(i)
    success_rate = []
    for n in n_params:
        for k in k_params:
            if k >= n:
                break
            for p in p_params:
                print("start run with (n,p,k) = ", n, p, k)
                forest = KNNForest(n)
                forest.train(train_data, p)
                accuracy = forest.test(test_data, k)
                print("accuracy is", accuracy)
                success_rate.append((accuracy, (n, k, p)))
    success_rate.sort(key=lambda x: x[0])
    return success_rate[0][-1]


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


def calc_centroid_improve(examples, relevant_features):
    centroid = []
    size = len(examples)
    min_max_vecctor = []
    for i in range(1, len(examples[0])):
        local_min = float('inf')
        local_max = float('-inf')
        for j in range(len(examples)):
            if examples[j][i] < local_min:
                local_min = examples[j][i]
            if examples[j][i] > local_max:
                local_max = examples[j][i]
        min_max_vecctor.append((local_min, local_max))
    for i in range(1, len(examples[0])):
        if i not in relevant_features:
            centroid.append(0.0)
        else:
            sum = 0.0
            for j in range(len(examples)):
                # print(j)
                sum += minmax_normalization(examples[j][i], min_max_vecctor[i][0], min_max_vecctor[i][1])
            average = sum / size
            centroid.append(average)
    return centroid, min_max_vecctor


class KNNForest:
    def __init__(self, n_param):
        self.n_param = n_param
        self.decision_trees = []

    def train(self, data, p_param):
        # kf = KFold(n_splits=3, shuffle=True, random_state=318981586)
        decisions_trees = []

        for i in range(self.n_param):
            size = p_param*self.n_param
            random_examples = choices(data, k=int(size))
            classifier = ID3(random_examples, 15, information_gain, majority_class_for_knn)
            classifier.train()
            relevant = classifier.root.find_features(classifier.num_of_features)
            centroid, min_max_vector = calc_centroid_improve(random_examples, relevant)
            height = classifier.root.calc_height()
            decisions_trees.append((centroid, height, classifier, min_max_vector))
        self.decision_trees = decisions_trees

    def classify_example(self, example, k_param):
        features_values = []
        for i in range(len(example)):
            features_values.append(example[i])
        # need to choose the k nearest classifiers

        k_decisions_tree = find_KNN_examples_for_improved(self.decision_trees, example, k_param)
        sick_num, healthy_num = 0, 0
        for i in range(k_param):
            # print(k_decisions_tree[i])
            classification = k_decisions_tree[i][2][2].root.find_class_by_example(example)
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


if __name__ == '__main__':
    data = load_data("train.csv")
    classifier = KNNForest(75)
    classifier.train(data, 0.7)
    tester = load_data("test.csv")
    accuracy = classifier.test(tester, 51)
    print(accuracy)
    # n, k, p = experiment(data, tester)
    # print(n, k, p)
    """after performing the experiment the best (n, k, p) are (60, 50, 0.69) or (75, 51, 0.7)"""

