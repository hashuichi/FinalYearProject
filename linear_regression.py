from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
from base_model import BaseModel

class LinearRegression(BaseModel):
    def train_model(self):
        """
        Train a Linear Regression model on the given features and labels.

        Returns:
        model (LinearRegression): The trained linear regression model.
        """
        self.model = linear_model.LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def fit_normal_eq(self):
        X = np.concatenate((np.ones((self.X_train.shape[0], 1)), self.X_train), axis=1)
        y = self.y_train.to_numpy().reshape(-1, 1)
        self.model_normal_eq = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict_normal_eq(self, new_entry):
        if self.model_normal_eq is None:
            print("Normal equation model not trained. Call fit_normal_eq() first.")
            return
        X_new = np.concatenate(([1], new_entry))
        y_pred = X_new.dot(self.model_normal_eq)
        return y_pred[0]

    def fit_gradient_descent(self, learning_rate=0.01, iterations=1000):
        X = np.concatenate((np.ones((self.X_train.shape[0], 1)), self.X_train), axis=1)
        y = self.y_train.to_numpy().reshape(-1, 1)
        theta = np.zeros((X.shape[1], 1))

        for _ in range(iterations):
            gradients = 2 / len(X) * X.T.dot(X.dot(theta) - y)
            theta -= learning_rate * gradients

        self.model_gradient_descent = theta

    def predict_gradient_descent(self, new_entry):
        if self.model_gradient_descent is None:
            print("Gradient descent model not trained. Call fit_gradient_descent() first.")
            return
        X_new = np.concatenate(([1], new_entry))
        y_pred = X_new.dot(self.model_gradient_descent)
        return y_pred[0]
    
    def get_y_pred_normal_eq(self):
        """
        Calculates the predicted array of labels from the test set.

        Returns:
        y_pred (array): The predicted labels
        """
        y_pred = []
        for i in range(len(self.X_test)):
            predicted_price = self.predict_normal_eq(self.X_test.iloc[i])
            y_pred.append(predicted_price)
        self.y_pred = y_pred
        return y_pred
    
    def get_y_pred_gradient_descent(self):
        """
        Calculates the predicted array of labels from the test set.

        Returns:
        y_pred (array): The predicted labels
        """
        y_pred = []
        for i in range(len(self.X_test)):
            predicted_price = self.predict_gradient_descent(self.X_test.iloc[i])
            y_pred.append(predicted_price)
        return y_pred
    
    def calculate_rmse(self, gradient_descent):
        """
        Calculates the RMSE values for different methods of linear regression

        Parameters:
        gradient_descent (boolean): If true, will use gradient descent method. Otherwise, normal eq method

        Returns:
        rmse_value (int): RMSE value
        """
        if (gradient_descent):
            y_pred = self.get_y_pred_gradient_descent()
        else:
            y_pred = self.get_y_pred_normal_eq()
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return rmse