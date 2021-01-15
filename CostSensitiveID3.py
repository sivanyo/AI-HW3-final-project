from ID3 import ID3
from utils import load_data
from utils import information_gain_for_cost_sensitive
from utils import minmax_normalization
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from utils import majority_class_for_cost


# def experiment(file_name):
#     """this function is the experiment
#     only need to add a call at the main function - experiment("train.csv")
#     and it will run :)
#     """
#     #m_params_list = [1, 2, 3, 4, 5, 10, 20, 25, 30, 40, 50, 100, 120, 150, 175, 200, 250]
#     m_params_list = []
#     for i in range(0, 20):
#         m_params_list.append(i)
#     successes_rate = []
#     kf = KFold(n_splits=5, shuffle=True, random_state=318981586)
#     data = load_data(file_name)
#     test = load_data("test.csv")
#     experiment = []
#     for i in range(len(m_params_list)):
#         accuracy = []
#         for train_index, test_index in kf.split(data):
#             train_data, test_data = [], []
#             for j in train_index:
#                 train_data.append(data[j])
#             for j in test_index:
#                 test_data.append(data[j])
#             classifier_t = CostSensitiveID3(train_data, m_params_list[i], information_gain_for_cost_sensitive)
#             classifier_t.train()
#             success_rate = classifier_t.test_by_loss(test_data)
#             if success_rate < 0.002:
#                 loss = classifier_t.test_by_loss(test)
#                 print(loss, m_params_list[i])
#             accuracy.append(success_rate)
#         print(sum(accuracy) / len(accuracy))
#         successes_rate.append(sum(accuracy) / len(accuracy))
#         experiment.append((sum(accuracy) / len(accuracy), m_params_list[i]))
#     experiment.sort(key=lambda x: x[0])
#     print(experiment)
#     plt.plot(m_params_list, successes_rate)
#     plt.xlabel("M parameter")
#     plt.ylabel("loss")
#     plt.show()

class CostSensitiveID3(ID3):
    def __init__(self, data_arr, m_param, information_gain_func, majority_class_for_cost, epsilon):
        ID3.__init__(self, data_arr, m_param, information_gain_func, majority_class_for_cost, epsilon)
        self.classifiers = None
        self.m_param = m_param
        self.epsilon = epsilon
    """maybe more than 2 splits ?"""
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
            classifier = CostSensitiveID3(train_data, self.m_param, information_gain_for_cost_sensitive, majority_class_for_cost, self.epsilon)
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
    # experiment("train.csv")
    data = load_data("train.csv")
    minimaizer = CostSensitiveID3(data, 25, information_gain_for_cost_sensitive, majority_class_for_cost, 5/100)
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
