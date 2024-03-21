import streamlit as st
import pandas as pd
from Data import DataPage
from data_loader import DataLoader
from nearest_neighbours import NearestNeighbours

class NearestNeighboursPage:
    def run(self):
        st.title("Nearest Neighbours")
        dl = DataLoader()
        selected_df = DataPage().get_data_selection()
        dl.load_data(selected_df)
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        knn = NearestNeighbours(selected_df, self.X_train, self.X_test, self.y_train, self.y_test)
        knn.get_y_pred()

        st.subheader('Optimise Hotel Price')
        new_entry = knn.get_new_hotel_fields(st)
        new_price = knn.predict_new_entry(new_entry)
        st.write(f'Best Selling Price Per Night Scratch: £{new_price:.2f}')

        if st.button("Genereate RMSE Plot (~5 Minutes)"):
            n_values = list(range(2, 20))
            rmse_values = knn.calculate_rmse(n_values)
            self.display_rmse_chart(rmse_values)
            best_k, best_rmse = knn.find_best_k()
            self.display_best_k_and_rmse(best_k, best_rmse)

        st.subheader('Performance Graphs (~1 Minute)')
        if st.button("Generate Performance Graphs"):
            knn.get_y_pred()
            knn.generate_plots(st)


    def display_rmse_chart(self, rmse_values):
        """
        Displays num_neighboours vs rmse_values chart
        """
        st.subheader('Results')
        n_values = list(rmse_values.keys())
        rmse_scores = list(rmse_values.values())
        rmse_data = pd.DataFrame(
            {
                "Number of Neighbours": n_values,
                "RMSE Value": rmse_scores
            }
        )
        st.line_chart(rmse_data, x="Number of Neighbours", y="RMSE Value")

    def display_best_k_and_rmse(self, best_k, best_rmse):
        """
        Displays the best k and the corresponding RMSE value
        """
        st.write(f'Best k: {best_k}')
        st.write(f'Best RMSE: {best_rmse:.2f}')


if __name__ == '__main__':
    app = NearestNeighboursPage()
    app.run()