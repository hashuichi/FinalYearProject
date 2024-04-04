import streamlit as st
import pandas as pd
from Data import DataPage
from data_loader import DataLoader
from nearest_neighbours import NearestNeighbours

class NearestNeighboursPage:
    """
    A class that encapsulates the functionality for displaying a page dedicated to the 
    Nearest Neighbours algorithm using Streamlit. It allows users to upload data, 
    perform predictions on new data points, and visualise the algorithm's performance.
    """
    
    def run(self):
        """
        Main execution function for the Nearest Neighbours page. It handles data loading, 
        training of the Nearest Neighbours model, prediction of new entries, and 
        visualisation of the model's performance.
        """
        st.title("Nearest Neighbours")
        dl = DataLoader()
        selected_df = DataPage().get_data_selection()
        dl.load_data(selected_df)
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        knn = NearestNeighbours(selected_df, self.X_train, self.X_test, self.y_train, self.y_test)
        y_pred = knn.get_y_pred()

        st.subheader('Optimise Hotel Price')
        new_entry = knn.get_new_hotel_fields(st)
        new_price = knn.predict_new_entry(new_entry)
        st.write(f'Best Selling Price Per Night: £{new_price:.2f}')

        st.subheader('Results')
        if st.button("Genereate RMSE Plot (~5 Minutes)"):
            n_values = list(range(2, 20))
            rmse_values = knn.calculate_rmse(n_values)
            self.display_rmse_chart(rmse_values)
            best_k, best_rmse = knn.find_best_k()
            st.write(f'Best k: {best_k}')
            st.write(f'Best RMSE: {best_rmse:.2f}')

        st.subheader('Performance Graphs')
        if st.button("Generate Performance Graphs (~1 Minute)"):
            knn.generate_plots(st, y_pred)


    def display_rmse_chart(self, rmse_values):
        """
        Displays a line chart of the Root Mean Square Error (RMSE) values for different 
        numbers of neighbours in the Nearest Neighbours algorithm.

        Parameters:
            rmse_values (dict): A dictionary where keys are the number of neighbours and values are the corresponding RMSE values.
        """
        n_values = list(rmse_values.keys())
        rmse_scores = list(rmse_values.values())
        rmse_data = pd.DataFrame(
            {
                "Number of Neighbours": n_values,
                "RMSE Value": rmse_scores
            }
        )
        st.line_chart(rmse_data, x="Number of Neighbours", y="RMSE Value")

if __name__ == '__main__':
    app = NearestNeighboursPage()
    app.run()