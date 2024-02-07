import streamlit as st
from Data import DataPage
from data_loader import DataLoader
from linear_regression import LinearRegression

class LinearRegressionPage:
    def run(self):
        st.title("Linear Regression")
        dl = DataLoader()
        dl.load_data(DataPage().load_data())
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        lr = LinearRegression(self.X_train, self.X_test, self.y_train, self.y_test)
        lr.train_model()

        st.subheader('Predict Hotel Price')
        star_rating, distance = lr.get_new_hotel_fields(st)
        new_price = lr.predict_price(star_rating, distance)
        st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')
        
        st.subheader('Results')
        rmse = lr.calculate_rmse_value()
        st.write(f'**RMSE:** {rmse:.2f}')

        st.subheader('Performance Graphs')
        lr.generate_plots(st)
    
if __name__ == '__main__':
    app = LinearRegressionPage()
    app.run()