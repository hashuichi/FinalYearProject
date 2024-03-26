from base_model import BaseModel
import numpy as np
from sklearn.metrics import mean_squared_error

class DecisionTree(BaseModel):
    def __init__(self, selected_df, X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2):
        super().__init__(selected_df, X_train, X_test, y_train, y_test)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X_train, y_train):
        self.tree = self._grow_tree(X_train, y_train)
        return self.tree

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split or num_labels == 1:
            return np.mean(y)

        # Find the best split
        best_feature, best_threshold = None, None
        best_gini = float('inf')

        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = X[:, feature_idx] <= threshold
                y_left = y[left_indices]
                y_right = y[~left_indices]

                gini = self._calculate_gini_impurity(y_left, y_right)
                if gini < best_gini:
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_gini = gini

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        # Recursively grow subtrees
        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _calculate_gini_impurity(self, y_left, y_right):
        p_left = len(y_left) / (len(y_left) + len(y_right))
        p_right = len(y_right) / (len(y_left) + len(y_right))
        gini_left = 1 - np.sum((np.unique(y_left, return_counts=True)[1] / len(y_left)) ** 2)
        gini_right = 1 - np.sum((np.unique(y_right, return_counts=True)[1] / len(y_right)) ** 2)
        return p_left * gini_left + p_right * gini_right

    def calculate_y_pred(self, X):
        return np.array([self._predict_entry(x, self.tree) for x in X])

    def _predict_entry(self, x, tree):
        if isinstance(tree, np.float64):
            return tree
        feature_idx, threshold, left_subtree, right_subtree = tree
        if x[feature_idx] <= threshold:
            return self._predict_entry(x, left_subtree)
        else:
            return self._predict_entry(x, right_subtree)

    def predict_price(self, x):
        return self._predict_entry(x, self.tree)
    
    def calculate_rmse_value(self, y_test, y_pred):
        return np.sqrt(mean_squared_error(y_test, y_pred))
