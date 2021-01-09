from utils import load_data
from utils import SICK
from utils import HEALTHY
from utils import Node
from utils import lost
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from utils import information_gain
from utils import majority_class


# example e[0]- class, e[1]-e[last] : features results, e[feature_index] -> e[feature_index+2]
def experiment(file_name):
    """this function is the experiment
    only need to add a call at the main function - experiment("train.csv")
    and it will run :)
    """
    # need to add the k-fold issue
    m_params_list = [1, 2, 3, 4, 5, 10, 20, 25, 30, 40, 50, 100, 120, 130, 150, 175, 200, 250]
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


class ID3:

    def __init__(self, data_arr, m_param=None, information_gain_func=information_gain,
                 majority_class_func=majority_class):
        self.data_arr = data_arr

        # print(self.data_arr)
        self.examples = data_arr
        # print(self.examples)
        # self.test = None
        self.classes = [SICK, HEALTHY]
        self.root = Node(m_param)
        self.information_gain_func = information_gain_func
        self.majority_class_func = majority_class_func
        self.num_of_features = len(self.examples[0]) - 1
        # print("start your training, good luck")

    def train(self):
        """this function calls to the function that build the decision tree,
        and saves it in the classifier root"""
        major_class = self.majority_class_func(self.examples)
        self.root.build(self.examples, major_class, self.information_gain_func)

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
        print(right / (len(test_group)))  # todo : this is the only print that need to appear
        # print("out of", len(tester), "you are right about", right)
        return right / (len(test_group))

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
    #print(classifier.root.calc_height())
    loss = classifier.test_by_loss(tester)
    print(loss)
    # # print("now trying to minimize loss")
    # # minimize_loss("train.csv")


    #experiment("train.csv")

