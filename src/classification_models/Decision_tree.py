import numpy as np
class Node:
    """
    A class representing a node in a decision tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """feature: The feature used for splitting at this node. Defaults to None.
            threshold: The threshold used for splitting at this node. Defaults to None.
            left: The left child node. Defaults to None.
            right: The right child node. Defaults to None."""
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value #None when not a leaf

    def is_leaf(self):
        return self.value is not None
    def value(self):
        return self.value


class DecisionTree:
    """
    A decision tree classifier for binary classification problems.
    """

    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, random_forest=-1):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features #number of features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])

        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, current_depth=0, random_forest=-1):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #check the stopping condition
        if n_samples < self.min_samples_split or current_depth >= self.max_depth:
            #create a leaf node
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        #find the best split
        feature_idxs = -1
        if random_forest != -1:
            feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_threshold, best_feature = self.best_split(X, y, feature_idxs=feature_idxs)

        #create child node
        left_idxs = X[:,best_feature] <= best_threshold  # left
        right_idxs = X[:,best_feature] > best_threshold  # right
        left_X = X[left_idxs,:]
        right_X = X[right_idxs,:]
        left = self.grow_tree(left_X, y[left_idxs], current_depth + 1, random_forest)
        right = self.grow_tree(right_X, y[right_idxs], current_depth + 1, random_forest)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def most_common_label(self, y):
        """Calculate the most occurring value in the given list of y"""
        y = list(y)
        most_common = max(y, key=lambda x: y.count(x))
        return most_common


        #create children
    def best_split(self, X,y, random_forest=-1, feature_idxs=-1):
        best_gain = -1
        best_feature, best_thresh, best_mse = None, None, float("inf")
        n_samples, n_features = X.shape
        if random_forest == -1:
            for feature in range(self.n_features):
                thresholds = np.unique(X[:, feature])
                X_column = X[:, feature]
                for threshold in thresholds:
                    #the information gain
                    gain = self.information_gain(y, X_column, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_thresh = threshold


        else:
            for feature_idx in feature_idxs:
                X_column = X[:, feature_idx]
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    # calculate the information gain
                    gain = self.information_gain(y, X_column, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_thresh = threshold


        return best_thresh, best_feature

    def information_gain(self, y, X_column, threshold):
        # entropy of parent
        parent_entropy = self.entropy(y)

        # create children
        left_idxs =np.argwhere(X_column<= threshold)  # left
        right_idxs = np.argwhere(X_column > threshold)  # right

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate weighted avg. entropy of children
        n = len(y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)
        #entropy of children
        e_l, e_r = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        children_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        #calculate information gain
        IG = parent_entropy - children_entropy
        return IG

    def entropy(self, y):
        """"""
        entropy = 0

        # Find the unique label values in y and loop over each value
        labels = np.unique(y)
        for label in labels:
            # Find the examples in y that have the current label
            label_examples = y[y == label]
            # Calculate the ratio of the current label in y
            pl = len(label_examples) / len(y)
            # Calculate the entropy using the current label and ratio
            entropy += -pl * np.log2(pl)

        return entropy

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])


    def traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value()

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

