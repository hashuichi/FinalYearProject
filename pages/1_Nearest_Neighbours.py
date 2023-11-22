import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from Home import Home
from DataLoader import DataLoader

class NearestNeighbours:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.knn_model = None

    def run(self):
        st.title("Nearest Neighbours")
        dl = DataLoader()
        dl.load_data(Home().get_data_selection())
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        self.train_model()
    
        star_rating, distance = self.get_new_hotel_fields()
        new_price = self.predict_price(star_rating, distance)
        st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')

        num_neighbours = list(range(1, len(self.y_test)))
        mse_values = self.calculate_rmse_values(num_neighbours)
        self.display_rmse_chart(num_neighbours, mse_values)
        best_k, best_rmse = self.find_best_k()
        self.display_best_k_and_rmse(best_k, best_rmse)

    def train_model(self, n_neighbors=5, X_train=None, y_train=None):
        """
        Train a KNN model on the given features and labels.
        Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        n_neighbours (int): Number of neighbours (default: 3)
        Returns:
        knn_model (KNeighborsRegressor): The trained KNN regression model.
        """
        if (X_train is None and y_train is None):
            X_train = self.X_train
            y_train = self.y_train
        self.knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.knn_model.fit(X_train, y_train)
        return self.knn_model

    def predict_price(self, star_rating, distance, knn_model=None):
        """
        Predict the price using a trained KNN regression model.
        Args:
        star_rating (float): Star rating of the new data point.
        distance (float): Distance to the city center of the new data point.
        Returns:
        predicted_price (float): Predicted price for the new data point.
        """
        if knn_model is None:
            knn_model = self.knn_model
        new_data = pd.DataFrame({"star_rating": [star_rating], "distance": [distance]})
        predicted_price = self.knn_model.predict(new_data)
        return predicted_price

    def find_best_k(self, X_test=None, y_test=None):
        """
        Find the best k for a KNN model using cross validation.
        Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        Returns:
        best_k (int): The best k to use for the given dataset.
        best_mse (int): The mse of the dataset using the best k.
        """
        if (X_test is None and y_test is None):
            X_test = self.X_test
            y_test = self.y_test
        param_grid = {'n_neighbors': list(range(1, min(17, len(self.y_test))))}
        grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=min(5, len(y_test)), scoring='neg_mean_squared_error')
        grid_search.fit(self.X_test, self.y_test)
        best_k = grid_search.best_params_['n_neighbors']
        best_mse = -grid_search.best_score_
        return best_k, np.sqrt(best_mse)

    def calculate_rmse_values(self, num_neighbours):
        """
        Calculates mse value for every k in num_neigbours
        """
        rmse_values = []
        for k in num_neighbours:
            self.train_model(n_neighbors=k)
            y_pred = self.knn_model.predict(self.X_test)
            rmse_values.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        return rmse_values

    def get_new_hotel_fields(self):
        """
        Calculates mse value for every k in num_neigbours
        """
        st.subheader('Predict Hotel Price')
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.text_input('Hotel Name', 'Hazel Inn')
        star_rating = col2.number_input('Star Rating', 1, 5, 2)
        distance = col3.number_input('Distance', 100, 5000, 100)
        return star_rating, distance

    def display_rmse_chart(self, num_neighbours, rmse_values):
        """
        Displays num_neighboours vs rmse_values chart
        """
        st.subheader('Root Mean Squared Error')
        rmse_data = pd.DataFrame(
            {
                "Number of Neighbours": num_neighbours,
                "RMSE Value": rmse_values
            }
        )
        st.line_chart(rmse_data, x="Number of Neighbours", y="RMSE Value")

    def display_best_k_and_rmse(self, best_k, best_rmse):
        """
        Displays the best k and the corresponding MSE value
        """
        st.write(f'Best k: {best_k}')
        st.write(f'Best RMSE: {best_rmse:.2f}')


if __name__ == '__main__':
    app = NearestNeighbours()
    app.run()