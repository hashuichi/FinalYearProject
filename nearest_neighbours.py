import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from base_model import BaseModel

class NearestNeighbours(BaseModel):
    def _find_nearest_neighbors(self, new_entry):
        distances = np.sqrt(np.sum((self.X_train - new_entry)**2, axis=1))
        nearest_indices = np.argsort(distances)[:self.n_neighbours]
        nearest_y = self.y_train[nearest_indices]
        return nearest_y
    
    def train_model2(self, n_neighbours=5):
        self.n_neighbours = n_neighbours

    def predict_new_entry(self, new_entry):
        nearest_y = self._find_nearest_neighbors(new_entry)
        predicted_price = np.mean(nearest_y)
        return predicted_price

    def y_pred(self):
        y_pred = []
        for i in range(len(self.X_test)):
            predicted_price = self.predict_new_entry(self.X_test[i])
            y_pred.append(predicted_price)
        return self.y_pred
    
    def train_model(self, n_neighbours=5):
        """
        Train a KNN model on the given features and labels.

        Parameters:
        n_neighbours (int): Number of neighbours (default: 5)

        Returns:
        knn_model (KNeighborsRegressor): The trained KNN regression model.
        """
        self.model = KNeighborsRegressor(n_neighbors=n_neighbours)
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def calculate_rmse_values(self, n_neighbours):
        """
        Calculates rmse value for every k in num_neigbours

        Parameters:
        n_neighbours (list(int)): List of integers for the number of neighbours

        Returns:
        rmse_values (array): The rmse values of the dataset for every k.
        """
        if self.model is not None:
            rmse_values = []
            for k in n_neighbours:
                self.train_model(n_neighbours=k)
                y_pred = self.model.predict(self.X_test)
                rmse_values.append(mean_squared_error(self.y_test, y_pred, squared=False))
            return rmse_values
        else:
            raise ValueError("Model has not been trained. Call train_model() first.")
        
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