import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from base_model import BaseModel
from tqdm import tqdm

class NeuralNetworks(BaseModel):
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def train_model(self, max_depth):
        self.max_depth = max_depth
        self.model = self.build_model()
        for epoch in tqdm(range(100), desc='Training', unit='epoch'):
            self.model.fit(self.X_train, self.y_train, self.max_depth, batch_size=10, verbose=0)

    def predict(self, new_entry):
        input_data = np.array(new_entry).reshape(1, -1)
        layer_input = input_data
        for layer in self.model.layers:
            layer_output = layer.activation(np.dot(layer_input, layer.get_weights()[0]) + layer.get_weights()[1])
            layer_input = layer_output
        return layer_output[0][0]
    
    def calculate_y_pred(self):
        self.y_pred = []
        for index, row in self.X_test.iterrows():
            y_pred_single = self.predict(row.values)
            self.y_pred.append(y_pred_single)
        self.y_pred = np.array(self.y_pred)

    def get_rmse(self):
        return np.sqrt(np.mean((self.y_test - self.y_pred) ** 2))
    
    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    # def sigmoid_derivative(self, x):
    #     return x * (1 - x)

    # def train_model(self, max_depth, learning_rate=0.1):
    #     self.weights = np.random.rand(self.X_train.shape[1], 1)
    #     self.bias = np.random.rand(1)
        
    #     for epoch in range(max_depth):
    #         # Forward propagation
    #         output = self.sigmoid(np.dot(self.X_train, self.weights) + self.bias)

    #         # Backpropagation
    #         error = self.y_train.values - output
    #         d_output = error * self.sigmoid_derivative(output)
    #         d_weights = np.dot(self.X_train.T, d_output)
    #         d_bias = np.sum(d_output)

    #         # Update weights and bias
    #         self.weights += learning_rate * d_weights
    #         self.bias += learning_rate * d_bias

    # def predict(self, X):
    #     return self.sigmoid(np.dot(X, self.weights) + self.bias)

    # def __init__(self, selected_df, X_train, X_test, y_train, y_test, max_depth=3):
    #     super().__init__(selected_df, X_train, X_test, y_train, y_test)
    #     self.max_depth = max_depth
    #     self.weights = [None] * max_depth
    #     self.biases = [None] * max_depth
    #     self.activations = [None] * max_depth
    #     self.deltas = [None] * max_depth
    
    # def initialize_parameters(self, input_size, hidden_size):
    #     np.random.seed(42)
    #     weights = np.random.randn(input_size, hidden_size) * 0.01
    #     bias = np.zeros((1, hidden_size))
    #     return weights, bias

    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    # def sigmoid_derivative(self, x):
    #     return x * (1 - x)

    # def relu(self, x):
    #     return np.maximum(0, x)

    # def relu_derivative(self, x):
    #     return np.where(x <= 0, 0, 1)

    # def forward_propagation(self, X):
    #     self.activations[0] = X
    #     for i in range(1, self.max_depth):
    #         z = np.dot(self.activations[i - 1], self.weights[i - 1]) + self.biases[i - 1]
    #         if i != self.max_depth - 1:
    #             self.activations[i] = self.relu(z)
    #         else:
    #             self.activations[i] = z

    # def backward_propagation(self, y):
    #     for i in range(self.max_depth - 1, 0, -1):
    #         if i == self.max_depth - 1:
    #             self.deltas[i] = self.activations[i] - y.values.reshape(-1, 1)  # Convert y_train to numpy array
    #         else:
    #             self.deltas[i] = np.dot(self.deltas[i + 1], self.weights[i].T) * self.relu_derivative(self.activations[i])
    #         self.weights[i - 1] -= np.dot(self.activations[i - 1].T, self.deltas[i]) * 0.01
    #         self.biases[i - 1] -= np.sum(self.deltas[i], axis=0, keepdims=True) * 0.01

    # def fit(self):
    #     input_size = self.X_train.shape[1]
    #     hidden_size = 64
    #     output_size = 1

    #     self.weights[0], self.biases[0] = self.initialize_parameters(input_size, hidden_size)
    #     for i in range(1, self.max_depth - 1):
    #         self.weights[i], self.biases[i] = self.initialize_parameters(hidden_size, hidden_size)
    #     self.weights[self.max_depth - 1], self.biases[self.max_depth - 1] = self.initialize_parameters(hidden_size, output_size)

    #     for epoch in range(50):
    #         self.forward_propagation(self.X_train)
    #         self.backward_propagation(self.y_train)

    # def predict(self, X):
    #     self.forward_propagation(X)
    #     return self.activations[self.max_depth - 1]
