import unittest
import pandas as pd
from data_loader import DataLoader
from decision_tree import DecisionTree

class TestDT(unittest.TestCase):
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
        self.model = DecisionTree(self.sample_data, self.X_train, self.X_test, self.y_train, self.y_test)
        self.model.train_model()

    def test_train_model(self):
        self.assertIsNotNone(self.model)

    def test_predict_price(self):
        predicted_price = self.model.predict_price([3, 2500])
        self.assertEqual(predicted_price, 70.0)

    def test_calculate_y_pred(self):
        y_pred = self.model.calculate_y_pred()
        self.assertEqual(round(y_pred[0]), 300)

    def test_calculate_rmse_value(self):
        rmse_value = self.model.calculate_rmse_value()
        self.assertEqual(round(rmse_value), 175)

if __name__ == '__main__':
    unittest.main()