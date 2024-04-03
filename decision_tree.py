from base_model import BaseModel
import numpy as np
from sklearn.metrics import mean_squared_error

class DecisionTree(BaseModel):
    """
    A class for implementing a Decision Tree regression model.

    This class extends the BaseModel class and implements a Decision Tree regression model for predicting
    target values based on recursive binary splitting of the feature space using Gini Impurity.
    """
    def __init__(self, selected_df, X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2):
        """
        Initialises the DecisionTree object with given parameters.

        Parameters:
            selected_df (string): Dataset name.
            X_train (pd.DataFrame): Feature matrix of the training data.
            X_test (pd.DataFrame): Feature matrix of the test data.
            y_train (pd.Series): Target values of the training data.
            y_test (pd.Series): Target values of the test data.
            max_depth (int, optional): Maximum depth of the tree. Defaults to None.
            min_samples_split (int, optional): Minimum number of samples required to split an internal node. Defaults to 2.
        """
        super().__init__(selected_df, X_train, X_test, y_train, y_test)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X_train, y_train):
        """
        Fits the Decision Tree model to the training data.

        Parameters:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target values of the training data.

        Returns:
            tuple: A tuple containing the feature index, threshold, and subtrees.
        """
        self.model = self._grow_tree(X_train, y_train)
        return self.model

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows the decision tree by finding the best splits.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target values.
            depth (int): Current depth of the tree.

        Returns:
            tuple: A tuple containing the feature index, threshold, and subtrees.
        """
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
        """
        Calculates the Gini impurity of a split.

        Parameters:
            y_left (np.ndarray): Target values of the left split.
            y_right (np.ndarray): Target values of the right split.

        Returns:
            float: The Gini impurity.
        """
        p_left = len(y_left) / (len(y_left) + len(y_right))
        p_right = len(y_right) / (len(y_left) + len(y_right))
        gini_left = 1 - np.sum((np.unique(y_left, return_counts=True)[1] / len(y_left)) ** 2)
        gini_right = 1 - np.sum((np.unique(y_right, return_counts=True)[1] / len(y_right)) ** 2)
        return p_left * gini_left + p_right * gini_right

    def get_y_pred(self, X):
        """
        Calculates the predicted target values for the input features.

        Parameters:
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        return np.array([self._predict_entry(x, self.model) for x in X])

    def _predict_entry(self, new_entry, tree):
        """
        Predicts the target value for a single data entry.

        Parameters:
            new_entry (np.ndarray): The feature vector of the new data entry.
            tree (tuple): Decision tree node.

        Returns:
            float: Predicted target value.
        """
        if isinstance(tree, np.float64):
            return tree
        feature_idx, threshold, left_subtree, right_subtree = tree
        if new_entry[feature_idx] <= threshold:
            return self._predict_entry(new_entry, left_subtree)
        else:
            return self._predict_entry(new_entry, right_subtree)

    def predict(self, new_entry):
        """
        Predicts the target value for a single data entry using the trained model.

        Parameters:
            new_entry (array): The feature vector of the new data entry.

        Returns:
            float: Predicted target value.
        """
        return self._predict_entry(new_entry, self.model)
    
    def calculate_rmse(self, y_test, y_pred):
        """
        Calculates the root mean squared error (RMSE) for the decision tree model.

        Parameters:
            y_test (np.ndarray): Actual target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: THE RMSE value.
        """
        return np.sqrt(mean_squared_error(y_test, y_pred))
