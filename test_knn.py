import unittest
import knn
import pandas as pd

class TestKNN(unittest.TestCase):
    def setUp(self):
        # Set up any data or resources needed for the tests
        self.sample_data = pd.DataFrame({
            "star_rating": [3, 4, 5, 2, 4, 3, 5, 1, 1, 2],
            "distance": [1500, 2000, 1000, 3000, 500, 1501, 2001, 1001, 3001, 501],
            "price": [200, 250, 180, 300, 220, 50, 70, 90, 130, 150]
        })
        self.X = self.sample_data[["star_rating", "distance"]]
        self.y = self.y = self.sample_data["price"]
        self.X_train, self.X_test, self.y_train, self.y_test = knn.split_data(self.X, self.y, 42)
        self.model = knn.train_knn_model(self.X_train, self.y_train, 1)

    def test_split_data(self):
        self.assertEqual(len(self.X_train), 7)
        self.assertEqual(len(self.X_test), 3)
        self.assertEqual(len(self.y_train), 7)
        self.assertEqual(len(self.y_test), 3)

    def test_knn_model_fit(self):
        self.assertIsNotNone(self.model)

    def test_knn_model_predict(self):
        # Predict the price for a known data point
        predicted_price = knn.predict_price(self.model, 3, 2500)
        self.assertEqual(predicted_price[0], 70)

    def test_find_best_k(self):
        max_k=2
        best_k = knn.find_best_k(self.X_test, self.y_test, max_k, cv=3)
        self.assertTrue(1 <= best_k <= max_k)
    
    def test_accuracy(self):
        accuracy = knn.calculate_accuracy(self.model, self.X_test, self.y_test)
        self.assertIsNotNone(accuracy)

if __name__ == '__main__':
    unittest.main()