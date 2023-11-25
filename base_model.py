import pandas as pd

class BaseModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None

    def train_model(self):
        raise NotImplementedError("train_model method must be implemented in the subclass.")

    def predict_price(self, star_rating, distance):
        """
        Predict the price using a trained model.

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
