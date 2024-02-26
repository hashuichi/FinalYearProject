import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from base_model import BaseModel
import streamlit as st

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
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=10, verbose=0)

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