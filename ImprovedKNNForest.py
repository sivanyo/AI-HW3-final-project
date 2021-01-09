from utils import load_data
from ID3 import ID3
from utils import SICK
from utils import HEALTHY
from utils import find_KNN_examples
from random import choices
from utils import information_gain
from utils import majority_class_for_knn


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
            classifier = ID3(random_examples, None, information_gain, majority_class_for_knn)
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
            classification = k_decisions_tree[i][1][1].root.find_class_by_example(example)
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
    classifier = KNNForest(60)
    classifier.train(data, 0.69)
    tester = load_data("test.csv")
    accuracy = classifier.test(tester, 50)
    print(accuracy)
    # n, k, p = experiment(data, tester)
    # print(n, k, p)
    """after performing the experiment the best (n, k, p) are (60, 50, 0.69)"""




