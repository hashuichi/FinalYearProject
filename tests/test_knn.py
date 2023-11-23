import unittest
import pandas as pd
from data_loader import DataLoader
from nearest_neighbours import NearestNeighbours

class TestKNN(unittest.TestCase):
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
        self.model = NearestNeighbours(self.X_train, self.X_test, self.y_train, self.y_test)
        self.model.train_model(1)

    def test_train_model(self):
        self.assertIsNotNone(self.model)

    def test_predict_price(self):
        predicted_price = self.model.predict_price(3, 2500)
        self.assertEqual(predicted_price[0], 70)

    def test_calculate_rmse_values(self):
        list_neighbours = list(range(1, len(self.y_test)))
        rmse_values = self.model.calculate_rmse_values(list_neighbours)
        self.assertEqual(rmse_values[0], 175.07141400011596)

    def test_find_best_k(self):
        max_k=2
        best_k, _ = self.model.find_best_k()
        self.assertTrue(1 <= best_k <= max_k)

if __name__ == '__main__':
    unittest.main()