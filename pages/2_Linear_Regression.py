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
        col1.pyplot(self.plot_predicted_prices())
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
        fig, ax = plt.subplots()
        scatter = ax.scatter(self.X_test['distance'], self.y_test, c=self.X_test['star_rating'], cmap='viridis', marker='o')
        # ax.scatter(self.X_test['distance'], self.y_pred, color='red')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Price')
        ax.legend(*scatter.legend_elements(), title='Star Rating')

        return fig
    
    def plot_predicted_prices(self):
        """
        Displays and plots the predicted prices compared to the true prices
        """
        fig, ax = plt.subplots()
        ax.scatter(self.X_test['distance'], self.y_test, c='blue', label='True Prices')
        ax.scatter(self.X_test['distance'], self.y_pred, color='red', label='Predicted Prices')
        ax.legend()
        ax.set_xlabel('Distance')
        ax.set_ylabel('Price')
        return fig

if __name__ == '__main__':
    app = LinearRegressionPage()
    app.run()