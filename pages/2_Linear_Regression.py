import streamlit as st
import matplotlib.pyplot as plt
from Data import DataPage
from data_loader import DataLoader
from linear_regression import LinearRegression

class LinearRegressionPage:
    def run(self):
        st.title("Linear Regression")
        dl = DataLoader()
        selected_df = DataPage().get_data_selection()
        dl.load_data(selected_df)
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()

        lr = LinearRegression(selected_df, self.X_train, self.X_test, self.y_train, self.y_test)
        lr.train_model()
        lr.fit_normal_eq()
        lr.fit_gradient_descent()

        st.subheader('Optimise Hotel Price')
        new_entry = lr.get_new_hotel_fields(st)
        predicted_price_normal_eq = lr.predict_normal_eq(new_entry)
        st.write(f"Predicted price using normal equation method: £{predicted_price_normal_eq:.2f}")
        predicted_price_gradient_descent = lr.predict_gradient_descent(new_entry)
        st.write(f"Predicted price using gradient descent method: £{predicted_price_gradient_descent:.2f}")
        
        st.subheader('Results')
        st.pyplot(self.plot_actual_vs_predicted_prices(lr))
        rmse_normal_eq = lr.calculate_rmse(False)
        st.write(f'**RMSE using normal equation method:** {rmse_normal_eq:.2f}')
        rmse_gradient_descent = lr.calculate_rmse(True)
        st.write(f'**RMSE using gradient descent method:** {rmse_gradient_descent:.2f}')

        st.subheader('Performance Graphs')
        lr.generate_plots(st)
  
    def plot_actual_vs_predicted_prices(self, lr):
        # Predict prices for X_test using normal equation method
        y_pred_normal_eq = lr.get_y_pred_normal_eq()

        # Predict prices for X_test using gradient descent method
        y_pred_gradient_descent = lr.get_y_pred_gradient_descent()

        # Create scatter plots for actual vs predicted prices
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, y_pred_normal_eq, color='blue')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Prices (Normal Equation)')

        plt.subplot(1, 2, 2)
        plt.scatter(self.y_test, y_pred_gradient_descent, color='green')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Prices (Gradient Descent)')

        plt.tight_layout()

        return plt
    
if __name__ == '__main__':
    app = LinearRegressionPage()
    app.run()