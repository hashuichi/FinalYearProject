from sklearn.calibration import LabelEncoder
import streamlit as st
import numpy as np
import nearest_neighbours
import gui

def main():
    knn()

def knn():
    data = nearest_neighbours.load_data('fake_dataset.csv')
    X, y = nearest_neighbours.preprocess_data(data)
    X_train, X_test, y_train, y_test = nearest_neighbours.split_data(X, y)
    knn_model = nearest_neighbours.train_knn_model(X_train, y_train)

    hotel_name, star_rating, distance = gui.display_new_hotel_fields()
    
    new_price = nearest_neighbours.predict_price(knn_model, hotel_name, star_rating, distance)
    gui.display_prediction(new_price)

    num_neighbours = list(range(1, 31))
    mse_values = nearest_neighbours.calculate_mse_values(num_neighbours, X_train, X_test, y_train, y_test)
    gui.display_mse_chart(num_neighbours, mse_values)
    best_k, best_mse = nearest_neighbours.find_best_k(X, y, num_neighbours)
    gui.display_best_k_and_mse(best_k, best_mse)

if __name__ == '__main__':
    main()