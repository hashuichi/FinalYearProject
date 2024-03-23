import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from base_model import BaseModel
import streamlit as st

class NeuralNetworks(BaseModel):
    def __init__(_self, selected_df, X_train, X_test, y_train, y_test, max_depth):
        super().__init__(selected_df, X_train, X_test, y_train, y_test)
        _self.max_depth = max_depth

    def build_model(_self):
        model = Sequential()
        model.add(Dense(64, input_dim=_self.X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @st.cache_resource()
    def train_model(_self):
        _self.model = _self.build_model()
        _self.model.fit(_self.X_train, _self.y_train, epochs=1, batch_size=10, verbose=0)
        return _self.model

    def predict(_self, model, new_entry):
        input_data = np.array(new_entry).reshape(1, -1)
        layer_input = input_data
        for layer in model.layers:
            layer_output = layer.activation(np.dot(layer_input, layer.get_weights()[0]) + layer.get_weights()[1])
            layer_input = layer_output
        return layer_output[0][0]
    
    @st.cache_resource()
    def calculate_y_pred(_self, _model):
        y_pred = []
        for index, row in _self.X_test.iterrows():
            y_pred_single = _self.predict(_model, row.values)
            y_pred.append(y_pred_single)
        _self.y_pred = y_pred
        return y_pred

    def get_rmse(_self, y_pred):
        return np.sqrt(np.mean((_self.y_test - y_pred) ** 2))