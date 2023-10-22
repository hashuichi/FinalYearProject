import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

# Step 2: Load the dataset
data = pd.read_csv('fake_dataset.csv')

# Step 3: Preprocess the data
label_encoder = LabelEncoder()
data['Hotel Name'] = label_encoder.fit_transform(data['Hotel Name'])

X = data.drop('Price', axis=1)
y = data['Price']

# Step 4: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train a KNN model
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Step 6: Evaluate the model on the test data
y_pred = knn_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')

new_hotel_name = 'Hazel Inn'
new_hotel = np.array([[label_encoder.fit_transform([new_hotel_name])[0], 2, 590]])
new_price = knn_model.predict(new_hotel)
print(f'Predicted Price Per Night for {new_hotel_name}: £{new_price[0]}')

param_grid = {'n_neighbors': list(range(1, 11))}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']
best_mse = -grid_search.best_score_  # Convert negative MSE back to positive
print(f'Best k: {best_k}')
print(f'Best Mean Squared Error: {best_mse}')