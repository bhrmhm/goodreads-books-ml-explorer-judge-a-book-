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


class RegressionTree:
    """
    A decision tree regression classifier for binary classification problems.
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

        #check the stopping condition
        if n_samples < self.min_samples_split or current_depth >= self.max_depth:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        #find the best split
        feature_idxs = -1
        if random_forest != -1:
            feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        #find the best split
        best_threshold, best_feature = self.best_split(X, y, feature_idxs=feature_idxs)

        #create child node
        I1 = X[:,best_feature] <= best_threshold  # left
        I2 = X[:,best_feature] > best_threshold  # right
        X1 = X[I1,:]
        X2 = X[I2,:]
        left = self.grow_tree(X1, y[I1], current_depth + 1, random_forest)
        right = self.grow_tree(X2, y[I2], current_depth + 1, random_forest)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)



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
                    gain = self.variance_reduction(y, X_column, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_thresh = threshold


        else:
            for feature_idx in feature_idxs:
                X_column = X[:, feature_idx]
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    # the information gain
                    gain = self.variance_reduction(y, X_column, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_thresh = threshold


        return best_thresh, best_feature

    def variance_reduction(self, y, X_column, threshold):
        I1 = X_column <= threshold
        I2 = X_column > threshold

        if len(y[I1]) == 0 or len(y[I2]) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[I1]), len(y[I2])

        parent_var = np.var(y)
        left_var = np.var(y[I1])
        right_var = np.var(y[I2])

        return parent_var - (n_l / n) * left_var - (n_r / n) * right_var



    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])


    def traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

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

