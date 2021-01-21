from ID3 import ID3
from utils import load_data
from utils import information_gain_for_cost_sensitive
from sklearn.model_selection import KFold
from utils import majority_class_for_cost


class CostSensitiveID3(ID3):
    def __init__(self, data_arr, m_param, information_gain_func, majority_class_for_cost, epsilon, delta):
        ID3.__init__(self, data_arr, m_param, information_gain_func, majority_class_for_cost, epsilon, delta)
        self.classifiers = None
        self.m_param = m_param
        self.epsilon = epsilon
    """this is 2 variables function - the dominant one is FN, 
    so we will prefer to classify a bounded person as sick """
    def minimize_loss(self, file_name):
        kf = KFold(n_splits=5, shuffle=True, random_state=318981586)
        data = load_data(file_name)
        classifiers = []

        for train_index, test_index in kf.split(data):
            train_data, test_data = [], []
            for j in train_index:
                train_data.append(data[j])
            for j in test_index:
                test_data.append(data[j])
            classifier = CostSensitiveID3(train_data, self.m_param, information_gain_for_cost_sensitive,
                                          majority_class_for_cost, self.epsilon, 1)
            classifier.train()
            loss = classifier.test_by_loss(test_data)
            classifiers.append((classifier, loss))

        classifiers.sort(key=lambda x: x[1])
        # print(classifiers)
        selected_classifier = classifiers[0][0]
        test_group = load_data("test.csv")
        loss = selected_classifier.test_by_loss(test_group)
        print(loss)


if __name__ == '__main__':
    """ 
    data = load_data("data_big.csv")
    classifier = ID3(data)
    classifier.train()
    test = load_data("test_big.csv")
    loss = classifier.test_by_loss(test)
    print("loss for ID3:", loss)
    print("now try to minimize loss, loss of costSensitiveID3 is:")
    """
    data = load_data("train.csv")
    minimaizer = CostSensitiveID3(data, 25, information_gain_for_cost_sensitive, majority_class_for_cost, 5/100, 1)
    minimaizer.minimize_loss("train.csv")

    """m param = 25, 1.05, 0.95 are the best so far"""
    """"epsilons = []
    test = load_data("test.csv")
    for j in range(1, 30):
        for n in range(1,101,3):
            minimaizer = CostSensitiveID3(data, j, information_gain_for_cost_sensitive, majority_class_for_cost, n /100)
            minimaizer.train()
            loss = minimaizer.test_by_loss(test)
            print(loss, n/100, j)"""
