import pandas as pd
from sklearn import tree
from sklearn.metrics import mean_squared_error
from base_model import BaseModel

class DecisionTree(BaseModel):
    def train_model(self):
        """
        Train a Linear Regression model on the given features and labels.

        Returns:
        model (LinearRegression): The trained linear regression model.
        """
        self.model = tree.DecisionTreeRegressor(random_state=500)
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