import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.y_pred = None

    def train_model(self):
        """
        Train a Linear Regression model on the given features and labels.

        Returns:
        model (LinearRegression): The trained linear regression model.
        """
        self.model = linear_model.LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict_price(self, star_rating, distance):
        """
        Predict the price using a trained linear regression model.

        Parameters:
        star_rating (float): Star rating of the new data point.
        distance (float): Distance to the city center of the new data point.

        Returns:
        predicted_price (float): Predicted price for the new data point.
        """
        if self.model is not None:
            new_data = pd.DataFrame({"star_rating": [star_rating], "distance": [distance]})
            predicted_price = self.model.predict(new_data)
            return predicted_price
        else:
            raise ValueError("Model has not been trained. Call train_model() first.")
        
    def calculate_y_pred(self):
        """
        Calculates rmse using the predicted labels for the test set

        Returns:
        rmse_value (int): The RMSE value of the dataset.
        """
        if self.model is not None:
            self.y_pred = self.model.predict(self.X_test)
            return self.y_pred
        else:
            raise ValueError("Model has not been trained. Call train_model() first.")

    def calculate_rmse_value(self):
        """
        Calculates rmse using the predicted labels for the test set

        Returns:
        rmse_value (int): The RMSE value of the dataset.
        """
        if self.model is not None:
            rmse = mean_squared_error(self.y_test, self.y_pred, squared=False)
            return rmse
        else:
            raise ValueError("Model has not been trained. Call train_model() first.")