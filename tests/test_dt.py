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
        self.model.fit(self.X_train.values, self.y_train.values)
        self.y_pred = self.model.get_y_pred(self.X_test.values)

    def test_train_model(self):
        self.assertIsNotNone(self.model)

    def test_predict_price(self):
        predicted_price = self.model.predict([3, 2500])
        self.assertEqual(predicted_price, 50.0)

    def test_calculate_y_pred(self):
        self.assertEqual(round(self.y_pred[0]), 90)

    def test_calculate_rmse_value(self):
        rmse_value = self.model.calculate_rmse(self.y_test, self.y_pred)
        self.assertEqual(round(rmse_value), 35)

if __name__ == '__main__':
    unittest.main()