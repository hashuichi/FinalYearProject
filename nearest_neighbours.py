import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 2: Load the dataset
data = pd.read_csv('fake_dataset.csv')

# Step 3: Preprocess the data
label_encoder = LabelEncoder()
data['Hotel Name'] = label_encoder.fit_transform(data['Hotel Name'])
data['Room Type'] = label_encoder.fit_transform(data['Room Type'])

X = data.drop('Price', axis=1)
y = data['Price']

# Step 4: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train a KNN model
k = 5  # You can choose the number of neighbors
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Step 6: Evaluate the model on the test data
y_pred = knn_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')
