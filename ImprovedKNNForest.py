from utils import load_data
from ID3 import ID3
from ID3 import BetterID3
from KNNForest import KNNForest
from utils import SICK
from utils import HEALTHY
from utils import find_KNN_examples_for_improved
from random import choices
from utils import information_gain
from utils import majority_class_for_knn
from utils import minmax_normalization
from CostSensitiveID3 import CostSensitiveID3
from utils import information_gain_for_improved_knn

"""this algo prefer trees with low height
it also check which features are relevant and only than calc the centroid
it is also normalize all the features"""

def experiment(train_data, test_data):
    n_params = []
    for i in range(5,70):
        n_params.append(i)
    p_params = [0.3, 0.4, 0.5, 0.6, 0.7]
    k_params = []
    m_params = []
    for i in range(1,25):
        m_params.append(i)
    for i in range(3, 99, 3):
        k_params.append(i)
    success_rate = []
    for n in n_params:
        for k in k_params:
            if k >= n:
                break
            for p in p_params:
                for j in m_params:
                    print("start run with (n,p,k) and m param = ", n, p, k, j)
                    forest = IKNNForest(n, j)
                    forest.train(train_data, p)
                    accuracy = forest.test(test_data, k)
                    print("accuracy is", accuracy)
                    success_rate.append((accuracy, (n, k, p, j)))
    success_rate.sort(key=lambda x: x[0])
    print(success_rate)
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


class IKNNForest:
    def __init__(self, n_param, m_param):
        self.n_param = n_param
        self.decision_trees = []
        self.m_param = m_param
        self.reg = []

    def train(self, data, p_param):
        # kf = KFold(n_splits=3, shuffle=True, random_state=318981586)
        decisions_trees = []

        for i in range(self.n_param):
            size = p_param*self.n_param
            random_examples = choices(data, k=int(size))
            test_group = [ex for ex in data if ex not in random_examples]
            classifier = ID3(random_examples, self.m_param, information_gain, majority_class_for_knn)
            classifier.train()
            score = classifier.test(test_group, False)
            relevant = classifier.root.find_features(classifier.num_of_features)
            centroid = calc_centroid(random_examples)
            height = classifier.root.calc_height()
            decisions_trees.append((centroid, height, classifier, score, relevant))
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
                #sick_num += 1
                sick_num += (k_param - i)
                #sick_num += 1 + (1/(i+1))
            else:
                #healthy_num += 1
                healthy_num += (k_param - i)
                #healthy_num += 1 + (1/(i+1))
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


def average_accuracy():
    accuracy_list = []
    data = load_data("train.csv")
    test = load_data("train.csv")
    for i in range(20):
        print("start", i, "run")
        classifier = IKNNForest(8, 4)
        classifier.train(data, 0.5)
        accuracy = classifier.test(test, 5)
        accuracy_list.append(accuracy)
        print("accuracy is ", accuracy)
    avg = sum(accuracy_list) / len(accuracy_list)
    print("avg accuracy is:", avg)


def find_who_is_better():
    impro_Score, reg_score = 0, 0
    data = load_data("train.csv")
    tester = load_data("train.csv")
    for i in range(20):
        reg = KNNForest(60)
        reg.train(data, 0.69)
        accuracy = reg.test(tester, 50)
        impro = IKNNForest(60, 4)
        impro.train(data, 0.69)
        impro_ac = impro.test(tester, 50)
        if impro_ac >= accuracy:
            impro_Score += 1
        else:
            reg_score += 1
    print("improved knn forest was better at", impro_Score/(impro_Score+reg_score))


if __name__ == '__main__':
    #average_accuracy()
    #find_who_is_better()
    data = load_data("train.csv")
    tester = load_data("train.csv")

    #experiment(data, tester)
    classifier = IKNNForest(40, 5)
    classifier.train(data, 0.7)
    accuracy = classifier.test(tester, 10)
    print(accuracy)
    #n, k, p = experiment(data, tester)
    #print(n, k, p)
    """after performing the experiment the best (n, k, p) are (60, 50, 0.69) or (75, 51, 0.7)"""

