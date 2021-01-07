from utils import majority_class
from utils import load_data
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt


def euclidean_distance(example_1, example_2):
    squares_sum = 0.0
    for i in range(1, len(example_1)):
        squares_sum += ((example_1[i]) - float(example_2[i]))**2
    e_distance = np.sqrt(squares_sum)
    return e_distance


def minmax_normalization(value, min_val, max_val):
    normalized_val = float((float(value)-float(min_val)) / (float(max_val) - float(min_val)))
    return normalized_val


def experiment(file_name):
    """this function is the experiment
    only need to add a call at the main function - experiment("train.csv")
    and it will run :)
    """
    # need to add the k-fold issue
    k_params_list = [0, 5, 10, 20, 25, 30, 40, 50, 100, 120, 130, 150, 175, 200, 250]
    successes_rate = []
    kf = KFold(n_splits=5, shuffle=True, random_state=318981586)
    data = load_data(file_name)
    for i in range(len(k_params_list)):
        accuracy = []
        for train_index, test_index in kf.split(data):
            train_data, test_data = [], []
            for j in train_index:
                train_data.append(data[j])
            for j in test_index:
                test_data.append(data[j])
            classifier_t = KNN(k_params_list[i])
            classifier_t.train(train_data)
            success_rate = classifier_t.test(test_data)
            # print(success_rate)
            accuracy.append(success_rate)
        successes_rate.append(sum(accuracy) / len(accuracy))
    plt.plot(k_params_list, successes_rate)
    plt.xlabel("K parameter")
    plt.ylabel("successes rate")
    plt.show()


class KNN:
    def __init__(self, k_param):
        self.data = None
        self.k_param = k_param
        self.minmax_noramlization = []

    def train(self, data):
        # print(data)
        # print("start trining")
        """first, perform the minmax normalization
        i holds the column, j holds the row"""
        transpose_data = np.array(data).T
        np.array(transpose_data).tolist()
        #print(transpose_data)
        for i in range(1, len(transpose_data)):
            local_min = float('inf')
            local_max = float('-inf')
            for j in range(len(transpose_data[0])):
                #print(i, j)
                if float(transpose_data[i][j]) < local_min:
                    local_min = float(transpose_data[i][j])
                elif float(transpose_data[i][j]) > local_max:
                    local_max = float(transpose_data[i][j])
            #print(local_max, local_min)
            self.minmax_noramlization.append([local_min, local_max])
            for j in range(len(data)):
                normal = minmax_normalization(float(transpose_data[i][j]), float(local_min), float(local_max))
                # print(normal)
                data[j][i] = normal
                print(data[j][i])

        self.data = data

    def find_KNN_examples(self, example):
        KNN_list = []
        distance_list = []
        for i in range(len(self.data)):
            e_distance = float(euclidean_distance(example, self.data[i]))
            # print(e_distance)
            distance_list.append((e_distance, self.data[i]))
        distance_list.sort(key=lambda x: x[0])
        # print(distance_list)

        for i in range(self.k_param):
            # print(distance_list[i])
            KNN_list.append(distance_list[i][1])
        #print(KNN_list)
        return KNN_list

    def classify(self, example):
        KNN_list = self.find_KNN_examples(example)
        # print(KNN_list)
        classify = majority_class(KNN_list)
        return classify

    def normalized_test(self, tester):
        for i in range(len(tester)):
            for j in range(1, len(tester[0])):
                """j is the feature index"""
                tester[i][j] = minmax_normalization(tester[i][j], self.minmax_noramlization[j-1][0],
                                                    self.minmax_noramlization[j-1][1])
        return tester

    def test(self, tester):
        size = len(tester)
        right = 0
        normalized_tester = self.normalized_test(tester)
        for i in range(len(tester)):
            if tester[i][0] is self.classify(tester[i]):
                right += 1
        #print(right / size)
        return right/size


if __name__ == '__main__':
    #c = KNN(3)
    #lister = [['B', 1], ['B', 2], ['B', 3], ['A', 4], ['A', 10]]
    #c.train(lister)
    #print(c.find_KNN_examples(['A',0]))
    experiment("train.csv")

