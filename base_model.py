import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
import streamlit as st

class BaseModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.y_pred = None

    def train_model(self):
        raise NotImplementedError("train_model method must be implemented in the subclass.")
    
    def get_new_hotel_fields(self):
        """
        Creates inputs for hotel data and returns the values
        """
        st.subheader('Predict Hotel Price')
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.text_input('Hotel Name', 'Hazel Inn')
        star_rating = col2.number_input('Star Rating', 1, 5, 2)
        distance = col3.number_input('Distance', 100, 5000, 100)
        return star_rating, distance

    def predict_price(self, star_rating, distance):
        """
        Predict the price using a trained model.

        Parameters:
        star_rating (float): Star rating of the new data point.
        distance (float): Distance to the city center of the new data point.

        Returns:
        predicted_price (float): Predicted price for the new data point.
        """
        if self.model is not None:
            new_data = pd.DataFrame({"star_rating": [star_rating], "distance": [distance]})
            predicted_price = self.model.predict(new_data)
            return predicted_price
        else:
            raise ValueError("Model has not been trained. Call train_model() first.")
        
    def calculate_y_pred(self):
        """
        Calculates the predicted array of labels from the test set.

        Returns:
        y_pred (array): The predicted labels
        """
        if self.model is not None:
            self.y_pred = self.model.predict(self.X_test)
            return self.y_pred
        else:
            raise ValueError("Model has not been trained. Call train_model() first.")
        
    def generate_plots(self):
        '''
        Displays the different plots to visualise the performance of the model.
        '''
        st.subheader("Performance Graphs")
        col1, col2 = st.columns(2)
        if self.y_pred is None:
            self.calculate_y_pred()
        col1.pyplot(self.plot_residual_plot())
        col2.pyplot(self.plot_predicted_actual())
        
    def plot_predicted_actual(self):
        """
        Plots the scatter plot of predicted vs actual prices

        Returns:
        fig (Figure): The figure containing the plot
        """
        fig, ax = plt.subplots()
        ax.scatter(self.y_test, self.y_pred, c='blue', label='True Prices')
        ax.set_title('Predicted Prices vs Actual Prices')
        ax.set_xlabel('Actual Prices')
        ax.set_ylabel('Predicted Prices')
        return fig
    
    def plot_residual_plot(self):
        """
        Plots the residual of the actual prices - predicted prices

        Returns:
        fig (Figure): The figure containing the plot
        """
        fig, ax = plt.subplots()
        scatter = ax.scatter(self.X_test['distance'], (self.y_test - self.y_pred), c=self.X_test['star_rating'], cmap='viridis', label='Residual Plot')
        ax.set_title('Residual Plot')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.legend(*scatter.legend_elements(), title='Star Rating')
        return fig
    
    def plot_learning_curve(self):
        """
        Plots the learning curve for the model

        Returns:
        fig (Figure): The figure containing the plot
        """
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error')
        
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        
        fig, ax = plt.subplots()
        ax.plot(train_sizes, train_scores_mean, label='Training error')
        ax.plot(train_sizes, test_scores_mean, label='Validation error')
        
        ax.set_title('Learning Curve')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('Mean Squared Error')
        ax.legend()
        return fig
    
    def plot_residuals_distribution(self):
        """
        Plots a histogram of the distribution of residuals

        Returns:
        fig (Figure): The figure containing the plot
        """
        fig, ax = plt.subplots()
        ax.hist((self.y_test - self.y_pred), bins=30, color='blue', edgecolor='black')
        ax.set_title('Distribution of Residuals')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        return fig