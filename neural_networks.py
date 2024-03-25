import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from base_model import BaseModel
import streamlit as st

class NeuralNetworks(BaseModel):
    def __init__(_self, selected_df, X_train, X_test, y_train, y_test, max_depth):
        super().__init__(selected_df, X_train, X_test, y_train, y_test)
        _self.max_depth = max_depth

    def build_feedforward(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @st.cache_resource()
    def train_feedforward(_self):
        _self.model = _self.build_feedforward()
        _self.model.fit(_self.X_train, _self.y_train, epochs=1, batch_size=10, verbose=0)
        return _self.model

    def predict_feedforward(self, model, new_entry):
        input_data = np.array(new_entry).reshape(1, -1)
        layer_input = input_data
        for layer in model.layers:
            layer_output = layer.activation(np.dot(layer_input, layer.get_weights()[0]) + layer.get_weights()[1])
            layer_input = layer_output
        return layer_output[0][0]
    
    @st.cache_resource()
    def calculate_y_pred_feedforward(_self, _model):
        y_pred = []
        for index, row in _self.X_test.iterrows():
            y_pred_single = _self.predict_feedforward(_model, row.values)
            y_pred.append(y_pred_single)
        _self.y_pred = y_pred
        return y_pred
    
    def build_recurrent(_self):    
        model = Sequential()
        model.add(LSTM(64, input_shape=(_self.X_train.shape[1], 1), activation='relu'))
        model.add(Dense(32, activation='relu'))
        # model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    @st.cache_resource()
    def train_recurrent(_self):
        model = _self.build_recurrent()
        model.fit(_self.X_train, _self.y_train, epochs=1, batch_size=10, verbose=0)
        return model
    
    def predict_recurrent(_self, model, new_entry):
        input_data = np.array(new_entry).reshape(1, _self.X_train.shape[1], 1)
        return model.predict(input_data)[0][0]
    
    def evaluate_model(self, model):
        return np.sqrt(model.evaluate(self.X_test, self.y_test))

    def get_rmse(_self, y_pred):
        return np.sqrt(np.mean((_self.y_test - y_pred) ** 2))