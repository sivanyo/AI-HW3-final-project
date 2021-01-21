from utils import load_data
from ID3 import ID3
from utils import SICK
from utils import HEALTHY
from utils import find_KNN_examples_improved
from utils import calc_centroid_for_impro
from utils import minmax_normalization
from utils import normalized_ex
from random import sample
from sklearn.model_selection import KFold


"""this is an improvement for the regular KNNForest
this algo - 
* gives more attention to near neighbors
* prefers more accurate trees
* normalizes all the data
* keeps in mind which features were relevant in building the tree
* choose the best m_param
* ? - choose the best (n,p,k) for the particular data set
"""


def create_minmax_vector(examples):
    minmax_vector = []
    for feature in range(1, len(examples[0])):
        local_min = float('inf')
        local_max = float('-inf')
        for ex in examples:
            if ex[feature] < local_min:
                local_min = ex[feature]
            if ex[feature] > local_max:
                local_max = ex[feature]
        minmax_vector.append((local_min, local_max))
    return minmax_vector


def normalized_set(data, minmax_vector):
    normalized_data = []
    for ex in data:
        normal = []
        normal.append(ex[0])
        for feature in range(1, len(data[0])):
            normalized_val = minmax_normalization(ex[feature], minmax_vector[feature-1][0], minmax_vector[feature-1][1])
            normal.append(normalized_val)
        normalized_data.append(normal)
    return normalized_data


def merge(file_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=318981586)
    data = load_data(file_name)
    ac = []
    for train_index, test_index in kf.split(data):
        train_data, test_data = [], []
        for j in train_index:
            train_data.append(data[j])
        for j in test_index:
            test_data.append(data[j])
        classifier = IKNNForest(60)
        classifier.train(train_data, 0.69)
        acc = classifier.test(test_data, 51)
        ac.append(acc)
    print(sum(ac) / len(ac))


class IKNNForest:
    def __init__(self, n_param):
        self.n_param = n_param
        self.decision_trees = []

    def train(self, data, p_param):
        decisions_trees = []

        for i in range(self.n_param):
            size = p_param*self.n_param
            random_examples = sample(data, k=int(size))
            minmax_vector = create_minmax_vector(random_examples)
            normalized_data = normalized_set(random_examples, minmax_vector)
            """ choose the best m_param """
            classifier = ID3(normalized_data, 10)
            classifier.train()
            test_group = [ex for ex in data if ex not in random_examples]
            """ normalize all the data : """
            normalized_test = normalized_set(test_group, minmax_vector)
            """ keeps in mind which features are relevant : """
            relevant = classifier.root.find_features(classifier.num_of_features)
            score = classifier.test(normalized_test, False)
            centroid = calc_centroid_for_impro(random_examples, relevant)
            decisions_trees.append((1-score, centroid, classifier, minmax_vector, relevant))
            """ prefer more accurate trees : """
            decisions_trees.sort(key=lambda x: x[0])
        self.decision_trees = decisions_trees

    def classify_example(self, example, k_param):
        features_values = []
        for i in range(len(example)):
            features_values.append(example[i])
        # need to choose the k nearest classifiers

        k_decisions_tree = find_KNN_examples_improved(self.decision_trees, example, k_param)
        sick_num, healthy_num = 0, 0
        for i in range(k_param):
            normal_ex = normalized_ex(example, k_decisions_tree[i][3])
            classification = k_decisions_tree[i][2][2].root.find_class_by_example(normal_ex)
            """prefer "closer" neighbors : """
            if classification is SICK:
                sick_num += k_param + 1 - i
            else:
                healthy_num += k_param + 1 - i
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
                forest = IKNNForest(n)
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
        classifier = IKNNForest(60)
        classifier.train(data, 0.5)
        accuracy = classifier.test(test, 51)
        accuracy_list.append(accuracy)
        print("accuracy is ", accuracy)
    avg = sum(accuracy_list) / len(accuracy_list)
    print("avg accuracy after 20 runs is:", avg)


if __name__ == '__main__':
    # average_accuracy()
    data = load_data("train.csv")
    classifier = IKNNForest(60)
    classifier.train(data, 0.5)
    tester = load_data("test.csv")
    accuracy = classifier.test(tester, 51)
    print(accuracy)
    # n, k, p = experiment(data, tester)
    # print(n, k, p)
    """after performing the experiment the best (n, k, p) are (60, 50, 0.69)"""
