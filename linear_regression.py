from sklearn import linear_model
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