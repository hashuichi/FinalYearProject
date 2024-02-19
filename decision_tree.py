from sklearn import tree
from base_model import BaseModel
import numpy as np

class DecisionTree(BaseModel):
    def train_model(self):
        """
        Train a Decision Tree model on the given features and labels.

        Returns:
        model (DecisionTreeRegressor): The trained decision tree model.
        """
        self.model = tree.DecisionTreeRegressor(random_state=500)
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def fit(self, max_depth=5):
        self.max_depth = max_depth
        self.model = self._grow_tree(self.X_train, self.y_train, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]
        if num_samples == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return y.iloc[0]
        
        best_split = self._find_best_split(X, y)
        feature_name, threshold = best_split['feature_name'], best_split['threshold']
        left_idxs = X[feature_name] <= threshold
        right_idxs = X[feature_name] > threshold
        
        left_subtree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return {'feature_name': feature_name, 'threshold': threshold, 'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_split = {}
        best_gini = float('inf')
        for feature_name in X.columns:
            thresholds = np.unique(X[feature_name])
            for threshold in thresholds:
                left_idxs = X[feature_name] <= threshold
                right_idxs = X[feature_name] > threshold
                gini = self._gini_impurity(y[left_idxs], y[right_idxs])
                if gini < best_gini:
                    best_split = {'feature_name': feature_name, 'threshold': threshold, 'gini': gini}
                    best_gini = gini
        return best_split

    def _gini_impurity(self, left_y, right_y):
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = len(right_y) / (len(left_y) + len(right_y))
        gini_left = 1 - sum([(np.sum(left_y == c) / len(left_y))**2 for c in np.unique(left_y)])
        gini_right = 1 - sum([(np.sum(right_y == c) / len(right_y))**2 for c in np.unique(right_y)])
        gini = p_left * gini_left + p_right * gini_right
        return gini

    def predict(self, X):
        return np.array([self._predict_entry(x, self.model) for x in X.values])

    def _predict_entry(self, x, tree):
        if isinstance(tree, dict):
            feature_name, threshold = tree['feature_name'], tree['threshold']
            if x[feature_name] <= threshold:
                return self._predict_entry(x, tree['left'])
            else:
                return self._predict_entry(x, tree['right'])
        else:
            return tree