import unittest
import pandas as pd
from data_loader import DataLoader
from neural_networks import NeuralNetworks

class TestNN(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            "star_rating": [3, 4, 5, 2, 4, 3, 5, 1, 1, 2],
            "distance": [1500, 2000, 1000, 3000, 500, 1501, 2001, 1001, 3001, 501],
            "price": [200, 250, 180, 300, 220, 50, 70, 90, 130, 150]
        })
        dl = DataLoader()
        dl.load_data(dataframe=self.sample_data)
        self.X, self.y = dl.get_features_labels()
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        self.neural_net = NeuralNetworks(self.sample_data, self.X_train, self.X_test, self.y_train, self.y_test, 2)
        self.feedforward_model = self.neural_net.train_feedforward()
        self.recurrent_model = self.neural_net.train_recurrent()

    def test_feedforward_model(self):
        self.assertIsNotNone(self.feedforward_model)

    def test_recurrent_model(self):
        self.assertIsNotNone(self.recurrent_model)

    def test_predict_feedforward(self):
        predicted_price = self.neural_net.predict_feedforward(self.feedforward_model, [3, 2500])
        self.assertTrue(isinstance(predicted_price, (float)))

    def test_predict_recurrent(self):
        predicted_price = self.neural_net.predict_recurrent(self.recurrent_model, [3, 2500])
        self.assertTrue(isinstance(float(predicted_price), (float)))

    def test_evaluate_model(self):
        rmse_value = self.neural_net.evaluate_model(self.feedforward_model)
        self.assertTrue(isinstance(rmse_value, (float)))

if __name__ == '__main__':
    unittest.main()