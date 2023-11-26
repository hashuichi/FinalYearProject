import streamlit as st
from Home import DataPage
from data_loader import DataLoader
from linear_regression import LinearRegression

class LinearRegressionPage:
    def __init__(self):
        self.lr = None

    def run(self):
        st.title("Linear Regression")
        dl = DataLoader()
        dl.load_data(DataPage().get_data_selection())
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        lr = LinearRegression(self.X_train, self.X_test, self.y_train, self.y_test)
        lr.train_model()

        star_rating, distance = lr.get_new_hotel_fields()
        new_price = lr.predict_price(star_rating, distance)
        st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')

        st.subheader('Results')
        lr.calculate_y_pred()
        rmse = lr.calculate_rmse_value()
        st.write(f'**RMSE:** {rmse:.2f}')

        lr.generate_plots()
    
if __name__ == '__main__':
    app = LinearRegressionPage()
    app.run()