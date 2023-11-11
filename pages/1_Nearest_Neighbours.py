import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import gui

label_encoder = LabelEncoder()

def main():
    data = load_data('fake_dataset.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    knn_model = train_knn_model(X_train, y_train)

    hotel_name, star_rating, distance = gui.display_new_hotel_fields()
    
    new_price = predict_price(knn_model, hotel_name, star_rating, distance)
    gui.display_prediction(new_price)

    num_neighbours = list(range(1, 31))
    mse_values = calculate_mse_values(num_neighbours, X_train, X_test, y_train, y_test)
    gui.display_mse_chart(num_neighbours, mse_values)
    best_k, best_mse = find_best_k(X, y)
    gui.display_best_k_and_mse(best_k, best_mse)

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data['Hotel Name'] = label_encoder.fit_transform(data['Hotel Name'])

    X = data.drop('Price', axis=1)
    y = data['Price']

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_knn_model(X_train, y_train, n_neighbors=3):
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    return knn_model

def calculate_mse(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return mse

def predict_price(model, hotel_name, star_rating, distance):
    new_hotel = np.array([[label_encoder.fit_transform([hotel_name])[0], star_rating, distance]])
    new_price = model.predict(new_hotel)
    return new_price

def find_best_k(X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 31))}
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']
    best_mse = -grid_search.best_score_  # Convert negative MSE back to positive
    return best_k, best_mse

def calculate_mse_values(num_neighbours, X_train, X_test, y_train, y_test):
    mse_values = []
    for k in num_neighbours:
        knn_model = train_knn_model(X_train, y_train, n_neighbors=k)
        y_pred = knn_model.predict(X_test)
        mse_values.append(calculate_mse(y_test, y_pred))
    return mse_values

if __name__ == '__main__':
    main()