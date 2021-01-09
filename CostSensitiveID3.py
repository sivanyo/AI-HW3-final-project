from ID3 import ID3
from utils import load_data
from sklearn.model_selection import KFold
from utils import information_gain_for_cost_sensitive
from utils import SICK
import numpy as np
from utils import entropy
from utils import majority_class


class CostSensitiveID3(ID3):
    def __init__(self, data_arr, information_gain_func):
        ID3.__init__(self, data_arr, None, information_gain_func)
    """maybe more than 2 splits ?"""
    """this is 2 variables function - the dominant one is FN, 
    so we will prefer to classify a bounded person as sick """
    def minimize_loss(self, file_name):
        kf = KFold(n_splits=10, shuffle=True, random_state=318981586)
        data = load_data(file_name)
        classifiers = []
        for train_index, test_index in kf.split(data):
            train_data, test_data = [], []
            for j in train_index:
                train_data.append(data[j])
            for j in test_index:
                test_data.append(data[j])
            classifier = CostSensitiveID3(train_data, information_gain_for_cost_sensitive)
            classifier.train()
            loss = classifier.test_by_loss(test_data)
            classifiers.append((classifier, loss))

        classifiers.sort(key=lambda x: x[1])
        print(classifiers)
        # print(classifiers)
        selected_classifier = classifiers[0][0]
        data = load_data(file_name)
        selected_classifier = CostSensitiveID3(data, information_gain_for_cost_sensitive)
        test_group = load_data("test.csv")
        loss = selected_classifier.test_by_loss(test_group)
        print(loss)


if __name__ == '__main__':
    data = load_data("train.csv")
    minimaizer = CostSensitiveID3(data, information_gain_for_cost_sensitive)
    minimaizer.minimize_loss("train.csv")
