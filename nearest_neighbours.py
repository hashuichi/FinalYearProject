import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from base_model import BaseModel

class NearestNeighbours(BaseModel):
    def _find_nearest_neighbors(self, new_entry):
        distances = np.sqrt(np.sum((self.X_train - new_entry)**2, axis=1))
        nearest_indices = np.argsort(distances)[:self.n_neighbours]
        nearest_y = self.y_train.iloc[nearest_indices]
        return nearest_y

    def predict_new_entry(self, new_entry, n_neighbours=5):
        """
        Train a KNN model on the given features and labels.

        Parameters:
        n_neighbours (int): Number of neighbours (default: 5)

        Returns:
        knn_model (KNeighborsRegressor): The trained KNN regression model.
        """
        self.n_neighbours = n_neighbours
        nearest_y = self._find_nearest_neighbors(new_entry)
        predicted_price = np.mean(nearest_y)
        return predicted_price

    def get_y_pred(self, n_neighbours=5):
        """
        Calculates the predicted array of labels from the test set.

        Returns:
        y_pred (array): The predicted labels
        """
        y_pred = []
        for i in range(len(self.X_test)):
            predicted_price = self.predict_new_entry(self.X_test.iloc[i], n_neighbours)
            y_pred.append(predicted_price)
        return y_pred
    
    def calculate_rmse(self, n_values, st):
        """
        Calculates the RMSE values for different numbers of neighbours.

        Parameters:
        n_values (array): Array of n_neighbours values to try

        Returns:
        rmse_values (dict): Dictionary mapping n_neighbours to RMSE
        """
        rmse_values = {}
        for n in n_values:
            st.write(n)
            y_pred = self.get_y_pred(n)
            self.y_pred = y_pred
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            rmse_values[n] = rmse
        return rmse_values
        
    def find_best_k(self):
        """
        Find the best k for a KNN model using cross validation from the test features and labels.

        Returns:
        best_k (int): The best k to use for the given dataset.
        best_rmse (int): The rmse of the dataset using the best k.
        """
        param_grid = {'n_neighbors': list(range(1, min(17, len(self.y_test))))}
        grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=min(5, len(self.y_test)), scoring='neg_mean_squared_error')
        grid_search.fit(self.X_test, self.y_test)
        best_k = grid_search.best_params_['n_neighbors']
        best_mse = -grid_search.best_score_
        return best_k, np.sqrt(best_mse)