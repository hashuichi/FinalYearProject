import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import process_data

def main():
    st.title("Nearest Neighbours")
    X, y = process_data.get_fake_structured_data()
    X_train, X_test, y_train, y_test = process_data.split_data(X, y)
    knn_model = train_knn_model(X_train, y_train)

    hotel_name, star_rating, distance = get_new_hotel_fields()
    
    new_price = predict_price(knn_model, star_rating, distance)
    st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')

    num_neighbours = list(range(1, 31))
    mse_values = calculate_mse_values(num_neighbours, X_train, X_test, y_train, y_test)
    display_mse_chart(num_neighbours, mse_values)
    best_k, best_mse = find_best_k(X, y)
    display_best_k_and_mse(best_k, best_mse)

def train_knn_model(X_train, y_train, n_neighbors=3):
    """
    Train a KNN model on the given features and labels.

    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training labels
    n_neighbours (int): Number of neighbours (default: 3)

    Returns:
    knn_model (KNeighborsRegressor): The trained KNN regression model.
    """
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
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

def find_best_k(X, y):
    """
    Find the best k for a KNN model using cross validation.

    Args:
    X (pd.DataFrame): Features.
    y (pd.Series): Labels.

    Returns:
    best_k (int): The best k to use for the given dataset.
    best_mse (int): The mse of the dataset using the best k.
    """
    param_grid = {'n_neighbors': list(range(1, 31))}
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_k = grid_search.best_params_['n_neighbors']
    best_mse = -grid_search.best_score_  # Convert negative MSE back to positive
    return best_k, best_mse

def calculate_mse_values(num_neighbours, X_train, X_test, y_train, y_test):
    """
    Calculates mse value for every k in num_neigbours
    """
    mse_values = []
    for k in num_neighbours:
        knn_model = train_knn_model(X_train, y_train, n_neighbors=k)
        y_pred = knn_model.predict(X_test)
        mse_values.append(mean_squared_error(y_test, y_pred))
    return mse_values

def get_new_hotel_fields():
    """
    Calculates mse value for every k in num_neigbours
    """
    st.subheader('Predict Hotel Price')
    col1, col2, col3 = st.columns([3,1,1])
    new_hotel_name = col1.text_input('Hotel Name', 'Hazel Inn')
    star_rating = col2.number_input('Star Rating', 1, 5, 2)
    distance = col3.number_input('Distance', 100, 5000, 100)

    return new_hotel_name, star_rating, distance

def display_mse_chart(num_neighbours, mse_values):
    """
    Displays num_neighboours vs mse_values chart
    """
    st.subheader('Mean Squared Error for n_neighbors 1 to 25')
    st.line_chart(dict(zip(num_neighbours, mse_values)))

def display_best_k_and_mse(best_k, best_mse):
    """
    Displays the best k and the corresponding MSE value
    """
    st.write(f'Best k: {best_k}')
    st.write(f'Best Mean Squared Error: {best_mse:.2f}')

def calculate_accuracy(model, X_test, y_test):
    """
    Calculates the accuracy of the KNN model
    """
    y_pred = model.predict(X_test)
    print(y_test)
    print(y_pred)
    return(np.mean(y_test == y_pred))

if __name__ == '__main__':
    main()
    