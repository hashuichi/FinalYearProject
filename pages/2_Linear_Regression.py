import streamlit as st
from Home import DataPage
from data_loader import DataLoader
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

class LinearRegressionPage:
    def __init__(self):
        self.y_pred = None
        self.lr = None

    def run(self):
        st.title("Linear Regression")
        dl = DataLoader()
        dl.load_data(DataPage().get_data_selection())
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        self.lr = LinearRegression(self.X_train, self.X_test, self.y_train, self.y_test)
        self.lr.train_model()

        star_rating, distance = self.get_new_hotel_fields()
        new_price = self.lr.predict_price(star_rating, distance)
        st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')

        st.subheader('Results')
        self.y_pred = self.lr.calculate_y_pred()
        rmse = self.lr.calculate_rmse_value()
        st.write(f'**RMSE:** {rmse:.2f}')

        col1, col2 = st.columns(2)
        col1.pyplot(self.plot_regression_line())
        col2.pyplot(self.plot_scatter_graph())
        
    
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
    
    def plot_scatter_graph(self):
        """
        Displays and plots the scatter graph of price vs distance
        """
        star_rating = self.X_test.iloc[:, 0]
        distance = self.X_test.iloc[:, 1]

        fig, ax = plt.subplots()
        scatter = ax.scatter(distance, self.y_test, c=star_rating, cmap='viridis', marker='o')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Price')
        ax.legend(*scatter.legend_elements(), title='Star Rating')

        return fig
    
    def plot_regression_line(self):
        """
        Displays and plots the regression line
        """
        fig, ax = plt.subplots()
        ax.scatter(self.y_test, self.y_pred, color='red', label='Regression Line')
        
        ax.set_xlabel('True Prices')
        ax.set_ylabel('Predicted Prices')
        return fig

if __name__ == '__main__':
    app = LinearRegressionPage()
    app.run()