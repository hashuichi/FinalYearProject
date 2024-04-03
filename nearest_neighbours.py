import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from base_model import BaseModel
from scipy.spatial import distance
import streamlit as st

class NearestNeighbours(BaseModel):
    """
    A class for implementing a K-Nearest Neighbours Regression model using Euclidean Distance.

    This class extends the BaseModel class and implements a k-nearest neighbours regression model for 
    predicting target values based on the features of the nearest neighbours.
    """

    def _find_nearest_neighbours(self, new_entry, n_neighbours):
        """
        Finds the nearest neighbours of a new data entry.

        Parameters:
            new_entry (array): The feature vector of the new data entry.
            n_neighbours (int): The number of nearest neighbours to find.

        Returns:
            pd.Series: The prices of the nearest neighbours.
        """

        distances = distance.cdist([new_entry], self.X_train, 'euclidean')[0]
        nearest_indices = np.argpartition(distances, n_neighbours)[:n_neighbours]
        nearest_y = self.y_train.iloc[nearest_indices]
        return nearest_y

    def predict_new_entry(self, new_entry, n_neighbours=5):
        """
        Predicts the target value for a new data entry using its nearest neighbors.

        Parameters:
            new_entry (array): The feature vector of the new data entry.
            n_neighbours (int, optional): The number of nearest neighbors to consider. Defaults to 5.

        Returns:
            float: The predicted target value.
        """
        self.n_neighbours = n_neighbours
        nearest_y = self._find_nearest_neighbours(new_entry, n_neighbours)
        predicted_price = np.mean(nearest_y)
        return predicted_price

    def get_y_pred(self, n_neighbours=5):
        """
        Predicts target values for all test data entries using the k-nearest neighbours model.

        Parameters:
            n_neighbours (int, optional): The number of nearest neighbours to consider. Defaults to 5.

        Returns:
            list: Predicted target values for all test data entries.
        """
        y_pred = []
        for i in range(len(self.X_test)):
            predicted_price = self.predict_new_entry(self.X_test.iloc[i], n_neighbours)
            y_pred.append(predicted_price)
        self.y_pred = y_pred
        return y_pred

    @st.cache_resource()
    def calculate_rmse(_self, _n_values):
        """
        Calculates the root mean squared error (RMSE) for different numbers of nearest neighbours.
        
        The results are cached using Streamlit's caching mechanism to improve performance on subsequent runs.
        Thus, the parameters start with an underscore to indicate that it should be excluded from Streamlit hashing.

        Parameters:
            _n_values (array): Array of numbers of nearest neighbors to evaluate.

        Returns:
            dict: A dictionary containing RMSE values for different numbers of nearest neighbors.
        """
        rmse_values = {}
        for n in _n_values:
            y_pred = _self.get_y_pred(n)
            _self.y_pred = y_pred
            rmse = np.sqrt(mean_squared_error(_self.y_test, y_pred))
            rmse_values[n] = rmse
        return rmse_values
        
    @st.cache_resource()
    def find_best_k(_self):
        """
        Finds the optimal number of nearest neighbors using grid search and cross-validation.

        The results cached using Streamlit's caching mechanism to improve performance on subsequent runs.
        Thus, the parameters start with an underscore to indicate that it should be excluded from Streamlit hashing.

        Returns:
            tuple: A tuple containing the optimal number of nearest neighbors and its corresponding RMSE.
        """
        param_grid = {'n_neighbors': list(range(1, min(17, len(_self.y_test))))}
        grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=min(5, len(_self.y_test)), scoring='neg_mean_squared_error')
        grid_search.fit(_self.X_test, _self.y_test)
        best_k = grid_search.best_params_['n_neighbors']
        best_mse = -grid_search.best_score_
        return best_k, np.sqrt(best_mse)