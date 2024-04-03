import numpy as np
from sklearn.metrics import mean_squared_error
from base_model import BaseModel

class LinearRegression(BaseModel):
    """A class for implementing a Linear Regression model using different methods for training and prediction.

    This class extends the BaseModel class and implements linear regression models using normal equation and
    gradient descent methods for predicting target values based on features.
    """

    def fit_normal_eq(self):
        """Fits the linear regression model using the normal equation method."""
        X = np.concatenate((np.ones((self.X_train.shape[0], 1)), self.X_train), axis=1)
        y = self.y_train.to_numpy().reshape(-1, 1)
        self.model_normal_eq = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict_normal_eq(self, new_entry):
        """
        Predicts the target value for a new data entry using the linear regression model trained with normal equation.

        Parameters:
            new_entry (array): The feature vector of the new data entry.

        Returns:
            float: The predicted target value.
        """
        if self.model_normal_eq is None:
            print("Normal equation model not trained. Call fit_normal_eq() first.")
            return
        X_new = np.concatenate(([1], new_entry))
        y_pred = X_new.dot(self.model_normal_eq)
        return y_pred[0]

    def fit_gradient_descent(self, learning_rate=0.01, iterations=1000):
        """
        Fits the linear regression model using the gradient descent method.

        Parameters:
            learning_rate (float, optional): The learning rate for gradient descent. Defaults to 0.01.
            iterations (int, optional): The number of iterations for gradient descent. Defaults to 1000.
        """
        X = np.concatenate((np.ones((self.X_train.shape[0], 1)), self.X_train), axis=1)
        y = self.y_train.to_numpy().reshape(-1, 1)
        theta = np.zeros((X.shape[1], 1))

        for _ in range(iterations):
            gradients = 2 / len(X) * X.T.dot(X.dot(theta) - y)
            theta -= learning_rate * gradients

        self.model_gradient_descent = theta

    def predict_gradient_descent(self, new_entry):
        """
        Predicts the target value for a new data entry using the linear regression model trained with gradient descent.

        Parameters:
            new_entry (array): The feature vector of the new data entry.

        Returns:
            float: The predicted target value.
        """
        if self.model_gradient_descent is None:
            print("Gradient descent model not trained. Call fit_gradient_descent() first.")
            return
        X_new = np.concatenate(([1], new_entry))
        y_pred = X_new.dot(self.model_gradient_descent)
        return y_pred[0]
    
    def get_y_pred_normal_eq(self):
        """
        Predicts target values for all test data entries using the linear regression model trained with normal equation.

        Returns:
            list: Predicted target values for all test data entries.
        """
        y_pred = []
        for i in range(len(self.X_test)):
            predicted_price = self.predict_normal_eq(self.X_test.iloc[i])
            y_pred.append(predicted_price)
        self.y_pred = y_pred
        return y_pred
    
    def get_y_pred_gradient_descent(self):
        """
        Predicts target values for all test data entries using the linear regression model trained with gradient descent.

        Returns:
            list: Predicted target values for all test data entries.
        """
        y_pred = []
        for i in range(len(self.X_test)):
            predicted_price = self.predict_gradient_descent(self.X_test.iloc[i])
            y_pred.append(predicted_price)
        return y_pred
    
    def calculate_rmse(self, gradient_descent):
        """
        Calculates the root mean squared error (RMSE) for the linear regression model.

        Args:
            gradient_descent (bool): Flag indicating whether to use the gradient descent model.

        Returns:
            float: The RMSE value.
        """
        if (gradient_descent):
            y_pred = self.get_y_pred_gradient_descent()
        else:
            y_pred = self.get_y_pred_normal_eq()
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return rmse