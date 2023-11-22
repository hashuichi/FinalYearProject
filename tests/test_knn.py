import unittest
import pandas as pd
import importlib
from DataLoader import DataLoader

class TestKNN(unittest.TestCase):
    def setUp(self):
        self.knn = importlib.import_module('pages.1_Nearest_Neighbours').NearestNeighbours
        self.sample_data = pd.DataFrame({
            "star_rating": [3, 4, 5, 2, 4, 3, 5, 1, 1, 2],
            "distance": [1500, 2000, 1000, 3000, 500, 1501, 2001, 1001, 3001, 501],
            "price": [200, 250, 180, 300, 220, 50, 70, 90, 130, 150]
        })
        dl = DataLoader()
        dl.load_data(dataframe=self.sample_data)
        self.X, self.y = dl.get_features_labels()
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        self.model = self.knn.train_model(self, n_neighbors=1, X_train=self.X_train, y_train=self.y_train)

    def test_knn_model_fit(self):
        self.assertIsNotNone(self.model)

    def test_knn_model_predict(self):
        predicted_price = self.knn.predict_price(self, 3, 2500, knn_model=self.model)
        self.assertEqual(predicted_price[0], 70)

    def test_find_best_k(self):
        max_k=2
        best_k, _ = self.knn.find_best_k(self, X_test=self.X_test, y_test=self.y_test)
        self.assertTrue(1 <= best_k <= max_k)

if __name__ == '__main__':
    unittest.main()