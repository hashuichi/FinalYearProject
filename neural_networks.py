import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from base_model import BaseModel

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
        self.model.fit(self.X_train, self.y_train, epochs=self.max_depth, batch_size=32, verbose=0)

    def predict_price(self, new_entry):
        new_entry = StandardScaler().fit(self.X_train).transform([new_entry])
        self.y_pred = self.model.predict(new_entry)
        return self.y_pred
