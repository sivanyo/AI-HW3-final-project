import numpy as np

class TDIDT():
    def __init__(self):
        pass

    def majority_class(self, examples):
        pass

    def is_consistent(self, class_c):
        pass

    def select_feature(self, features, examples):
        pass

    def create_sub_tree(self, features, class_c, feature, old_f):
        pass

    def test(self, examples, features, default, feature):
        if examples is {}:
            # means the leaf is empty
            return None, {}, default

        class_c = self.majority_class(examples)
        if self.is_consistent(class_c):
            # means the node is consistent, make it a leaf
            return None, {}, class_c

        feature_f = self.select_feature(features, examples)
        new_features = features - feature_f

        subtree = self.create_sub_tree(new_features, class_c, feature, features)

        return feature_f, subtree, class_c


class ID3():
    def __init__(self):
        self.decision_tree = None
