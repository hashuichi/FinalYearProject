import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from base_model import BaseModel
import streamlit as st

class NeuralNetworks(BaseModel):
    """
    A class for implementing Feedforward Neural Networks and Recurrent Neural Networks regression models.

    This class extends the BaseModel class and implements Neural Networks regression models for predicting
    target values using feedforward and recurrent architectures.
    """
    def __init__(_self, selected_df, X_train, X_test, y_train, y_test, num_layers):
        """
         Initialises the NeuralNetworks object with given parameters.

        Parameters:
            selected_df (string): Dataset name.
            X_train (pd.DataFrame): Feature matrix of the training data.
            X_test (pd.DataFrame): Feature matrix of the test data.
            y_train (pd.Series): Target values of the training data.
            y_test (pd.Series): Target values of the test data.
            num_layers (int): Number of layers in the neural network.
        """
        super().__init__(selected_df, X_train, X_test, y_train, y_test)
        _self.num_layers = num_layers

    def build_model(self):
        """
        Builds a neural network model with the specified architecture.

        Constructs a neural network model with the specified number of layers and neurons in each layer.
        The model architecture consists of an input layer, followed by multiple hidden layers with ReLU 
        activation, and an output layer with linear activation.

        Returns:
            Sequential: The constructed neural network model.
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.X_train.shape[1], activation='relu'))
        for _ in range(self.num_layers - 1):
            model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @st.cache_resource()
    def train_feedforward(_self):
        """
        Trains a feedforward neural network model.

        The trained model is cached using Streamlit's caching mechanism to improve performance on subsequent runs.
        Thus, the parameters start with an underscore to indicate that it should be excluded from Streamlit hashing.

        Returns:
            Sequential: The trained feedforward neural network model.
        """
        _self.model = _self.build_model()
        _self.model.fit(_self.X_train, _self.y_train, epochs=1, batch_size=10, verbose=0)
        return _self.model

    def predict_feedforward(self, model, new_entry):
        """
        Predicts the target value for a new data entry using a trained feedforward neural network model.

        Parameters:
            model (Sequential): The trained feedforward neural network model.
            new_entry (array): The feature vector of the new data entry.

        Returns:
            float: The predicted target value.
        """
        input_data = np.array(new_entry).reshape(1, -1)
        layer_input = input_data
        for layer in model.layers:
            layer_output = layer.activation(np.dot(layer_input, layer.get_weights()[0]) + layer.get_weights()[1])
            layer_input = layer_output
        return layer_output[0][0]
    
    @st.cache_resource()
    def get_y_pred_feedforward(_self, _model):
        """
        Calculates predicted target values for the test data using a trained feedforward neural network model.

        The results are cached using Streamlit's caching mechanism to improve performance on subsequent runs.
        Thus, the parameters start with an underscore to indicate that it should be excluded from Streamlit hashing.

        Parameters:
            _model (Sequential): The trained feedforward neural network model.

        Returns:
            list: Predicted target values for the test data.
        """
        y_pred = []
        for index, row in _self.X_test.iterrows():
            y_pred_single = _self.predict_feedforward(_model, row.values)
            y_pred.append(y_pred_single)
        _self.y_pred = y_pred
        return y_pred
    
    @st.cache_resource()
    def train_recurrent(_self):
        """
        Trains a recurrent neural network model.

        The trained model is cached using Streamlit's caching mechanism to improve performance on subsequent runs.
        Thus, the parameters start with an underscore to indicate that it should be excluded from Streamlit hashing.

        Returns:
            Sequential: The trained recurrent neural network model.
        """
        model = _self.build_model()
        model.fit(_self.X_train, _self.y_train, epochs=1, batch_size=10, verbose=0)
        return model
    
    def predict_recurrent(_self, model, new_entry):
        """
        Predicts the target value for a new data entry using a trained recurrent neural network model.

        Parameters:
            model (Sequential): The trained recurrent neural network model.
            new_entry (array): The feature vector of the new data entry.

        Returns:
            float: The predicted target value.
        """
        input_data = np.array(new_entry).reshape(1, _self.X_train.shape[1], 1)
        return model.predict(input_data)[0][0]
    
    def evaluate_model(self, model):
        """
        Evaluates the performance of a neural network model on the test data.

        Parameters:
            model (Sequential): The trained neural network model.

        Returns:
            float: The root mean squared error (RMSE) of the model on the test data.
        """
        return np.sqrt(model.evaluate(self.X_test, self.y_test))

    def calculate_rmse(_self, y_pred):
        """
        Calculates the root mean squared error (RMSE) between the predicted and actual target values.

        Parameters:
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: Root mean squared error (RMSE) value.
        """
        return np.sqrt(np.mean((_self.y_test - y_pred) ** 2))
    