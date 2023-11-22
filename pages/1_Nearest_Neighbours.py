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
        st.set_page_config(page_title="Nearest Neighbours", layout="wide")
        file_name = Home().get_data_selection()
        dl = DataLoader(file_name)
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        self.knn_model = None

    def run(self):
        st.title("Nearest Neighbours")
        self.train_knn_model()
        hotel_name, star_rating, distance = self.get_new_hotel_fields()
        new_price = self.predict_price(star_rating, distance)
        st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')

        num_neighbours = list(range(1, len(self.y_test)))
        mse_values = self.calculate_rmse_values(num_neighbours)
        self.display_rmse_chart(num_neighbours, mse_values)
        best_k, best_rmse = self.find_best_k()
        self.display_best_k_and_rmse(best_k, best_rmse)

    def train_knn_model(self, n_neighbors=5):
        """
        Train a KNN model on the given features and labels.
        Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        n_neighbours (int): Number of neighbours (default: 3)
        Returns:
        knn_model (KNeighborsRegressor): The trained KNN regression model.
        """
        self.knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.knn_model.fit(self.X_train, self.y_train)

    def predict_price(self, star_rating, distance):
        """
        Predict the price using a trained KNN regression model.
        Args:
        knn_model (KNeighborsRegressor): The trained KNN regression model.
        star_rating (float): Star rating of the new data point.
        distance (float): Distance to the city center of the new data point.
        Returns:
        predicted_price (float): Predicted price for the new data point.
        """
        new_data = pd.DataFrame({"star_rating": [star_rating], "distance": [distance]})
        predicted_price = self.knn_model.predict(new_data)
        return predicted_price

    def find_best_k(self):
        """
        Find the best k for a KNN model using cross validation.
        Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        Returns:
        best_k (int): The best k to use for the given dataset.
        best_mse (int): The mse of the dataset using the best k.
        """
        param_grid = {'n_neighbors': list(range(1, min(17, len(self.y_test))))}
        grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
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
            self.train_knn_model(n_neighbors=k)
            y_pred = self.knn_model.predict(self.X_test)
            rmse_values.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
        return rmse_values

    def get_new_hotel_fields(self):
        """
        Calculates mse value for every k in num_neigbours
        """
        st.subheader('Predict Hotel Price')
        col1, col2, col3 = st.columns([3, 1, 1])
        new_hotel_name = col1.text_input('Hotel Name', 'Hazel Inn')
        star_rating = col2.number_input('Star Rating', 1, 5, 2)
        distance = col3.number_input('Distance', 100, 5000, 100)
        return new_hotel_name, star_rating, distance

    def display_rmse_chart(self, num_neighbours, mse_values):
        """
        Displays num_neighboours vs rmse_values chart
        """
        st.subheader('Root Mean Squared Error')
        st.line_chart(dict(zip(num_neighbours, mse_values)))

    def display_best_k_and_rmse(self, best_k, best_rmse):
        """
        Displays the best k and the corresponding MSE value
        """
        st.write(f'Best k: {best_k}')
        st.write(f'Best RMSE: {best_rmse:.2f}')


if __name__ == '__main__':
    app = NearestNeighbours()
    app.run()