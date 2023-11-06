import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def process_fake_data():
    """
    Split the data into training and testing sets.

    Args:
    X (pd.DataFrame): Features.
    y (pd.series): Labels.
    test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    random_state (int or None): Random seed for reproducibility (default is None).

    Returns:
    X_ (pd.DataFrame): Features.
    y_ (pd.Series): Labels.
    """
    data = pd.read_csv("fake_structured_data.csv")
    X = data[["star_rating", "distance"]]
    y = data["price"]
    return X, y

def split_data(X, y, random_state):
    """
    Split the data into training and testing sets.

    Args:
    X (pd.DataFrame): Features.
    y (pd.series): Labels.
    test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    random_state (int or None): Random seed for reproducibility (default is None).

    Returns:
    X_train (pd.DataFrame): Features for training.
    X_test (pd.DataFrame): Features for testing.
    y_train (pd.Series): Target variable for training.
    y_test (pd.Series): Target variable for testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_knn_model(X, y, n_neighbors):
    """
    Train a KNN regression model on the given features (X) and target variable (y).

    Args:
    X (pd.DataFrame): Features.
    y (pd.Series): Labels.
    n_neighbors (int): Number of neighbors to use (default is 5).

    Returns:
    knn_model (KNeighborsRegressor): The trained KNN regression model.
    """
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_model.fit(X, y)
    return knn_model

def predict_price(knn_model, star_rating, distance):
    """
    Predict the price using a trained KNN regression model.

    Args:
    knn_model (KNeighborsRegressor): The trained KNN regression model.
    star_rating (float): Star rating of the new data point.
    distance (float): Distance to the city center of the new data point.

    Returns:
    predicted_price (float): Predicted price for the new data point.
    """
    new_data = pd.DataFrame({"star_rating": [star_rating], "distance": [distance]})
    predicted_price = knn_model.predict(new_data)
    return predicted_price

def find_best_k(X, y, max_k, cv=5):
    """
    Find the best k for a KNN model using a set of data features (X) and labels (y).

    Args:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    max_k (int): Number of max neighbours to test.
    cv (int): Number of fold for the cross-validator.

    Returns:
    best_k (int): The best k to use for the given dataset.
    """
    k_values = list(range(1, max_k + 1))
    k_scores = []

    for k in k_values:
        knn_model = train_knn_model(X, y, n_neighbors=k)
        scores = cross_val_score(knn_model, X, y, cv=cv, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        k_scores.append(rmse)

    best_k = k_values[np.argmin(k_scores)]
    return best_k

def calculate_accuracy(model, X_test, y_test):
    """
    Calculates the accuracy of the KNN model
    """
    y_pred = model.predict(X_test)
    print(y_test)
    print(y_pred)
    return(np.mean(y_test == y_pred))
    # mse = mean_squared_error(y_test, y_pred)
    # print(mse)
