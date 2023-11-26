import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from base_model import BaseModel

class LinearRegression(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)

    def train_model(self):
        """
        Train a Linear Regression model on the given features and labels.

        Returns:
        model (LinearRegression): The trained linear regression model.
        """
        self.model = linear_model.LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def calculate_rmse_value(self):
        """
        Calculates rmse using the predicted labels for the test set

        Returns:
        rmse_value (int): The RMSE value of the dataset.
        """
        if self.model is not None:
            rmse = mean_squared_error(self.y_test, self.calculate_y_pred(), squared=False)
            return rmse
        else:
            raise ValueError("Model has not been trained. Call train_model() first.")