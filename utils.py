import pandas as pd


def load_data(filename):
    """this function gets a csv file name as a string and return a list of lists,
    each inner list represent a line in the file
    """
    data = pd.read_csv(filename, sep=',')
    return data.values.tolist()


def majority_class(examples_group):
    """this function gets a set of examples and returns the most common label
    in this set"""
    print(examples_group)
    class_1_type = examples_group[0][0]
    class_1, class_2 = [examples_group[0]], []
    for i in range(1, len(examples_group)):
        if examples_group[i][0] is class_1_type:
            class_1.append(examples_group[i])
        else:
            class_2.append(examples_group[i])
    if len(class_1) >= len(class_2):
        return class_1_type
    return class_2[0][0]

