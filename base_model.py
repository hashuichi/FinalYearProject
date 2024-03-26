import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error

class BaseModel:
    def __init__(self, selected_df, X_train, X_test, y_train, y_test):
        self.selected_df = selected_df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.y_pred = None
        self.room_type_map = {
            'Private room': 1,
            'Entire home/apt': 2,
            'Shared room': 3,
            'Hotel room': 4
        }

    def train_model(self):
        raise NotImplementedError("train_model method must be implemented in the subclass.")
    
    def get_new_hotel_fields(self, st):
        """
        Creates inputs for hotel data and returns the values
        """
        if self.selected_df == "Benchmark Dataset":
            col1, col2, col3, col4 = st.columns(4)
            room_type = col1.selectbox('Room Type', self.room_type_map.keys())
            occupancy = col2.number_input('Number of Occupants', 1, 16, 2)
            bathrooms = col3.selectbox('Number of Bathrooms', sorted(self.X_train['bathrooms'].unique()), 1)
            beds = col4.number_input('Number of Beds', 1, 50, 1)
            return [self.room_type_map[room_type], occupancy, bathrooms, beds]
        else:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.text_input('Hotel Name', 'Hazel Inn')
            star_rating = col2.number_input('Star Rating', 1, 5, 2)
            distance = col3.number_input('Distance', 100, 5000, 100)
            return [star_rating, distance]

    def predict_price(self, new_entry):
        """
        Predict the price using a trained model.

        Parameters:
        new_entry (array): Array containing values of new features.

        Returns:
        predicted_price (float): Predicted price for the new data point.
        """
        if self.model is not None:
            if self.selected_df == "Benchmark Dataset":
                new_data = pd.DataFrame({
                    "room_type": new_entry[0], 
                    "accommodates": new_entry[1], 
                    "bathrooms": new_entry[2], 
                    "beds": new_entry[3]}, index=[0])
            else:    
                new_data = pd.DataFrame({"star_rating": new_entry[0], "distance": new_entry[1]}, index=[0])
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
        
    def get_y_pred(self):
        raise NotImplementedError("get_y_pred method must be implemented in the subclass.")
        
    def calculate_rmse_value(self):
        """
        Calculates rmse using the predicted labels for the test set

        Returns:
        rmse_value (int): The RMSE value of the dataset.
        """
        if self.y_pred is None:
            self.calculate_y_pred()
        if self.model is not None:
            rmse = mean_squared_error(self.y_test, self.calculate_y_pred(), squared=False)
            return rmse
        else:
            raise ValueError("Model has not been trained. Call train_model() first.")
        
    def generate_plots(self, st, optional_y_pred=None):
        '''
        Displays the different plots to visualise the performance of the model.
        '''
        col1, col2 = st.columns(2)
        if optional_y_pred is not None:
            self.y_pred = optional_y_pred

        col2.pyplot(self.plot_predicted_actual())
        # col2.pyplot(self.plot_learning_curve())
        col1.pyplot(self.plot_residuals_distribution())
        # col1.pyplot(self.plot_residual_plot())
        
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
