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
        knn.train_model()

        st.subheader('Optimise Hotel Price')
        new_entry = knn.get_new_hotel_fields(st)
        # st.write(new_entry)
        # st.write("Shape of self.X_train:", self.X_train)
        # st.write("Shape of self.y_train:", self.y_train)
        new_price = knn.predict_price(new_entry)
        st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')
        knn.train_model2()
        new_price2 = knn.predict_new_entry(new_entry)
        st.write(f'Predicted Price Per Night Scratch: £{new_price2:.2f}')

        num_neighbours = list(range(1, len(self.y_test)))
        rmse_values = knn.calculate_rmse_values(num_neighbours)
        self.display_rmse_chart(num_neighbours, rmse_values)
        best_k, best_rmse = knn.find_best_k()
        self.display_best_k_and_rmse(best_k, best_rmse)

        st.subheader('Performance Graphs')
        knn.generate_plots(st)

    def display_rmse_chart(self, num_neighbours, rmse_values):
        """
        Displays num_neighboours vs rmse_values chart
        """
        st.subheader('Results')
        rmse_data = pd.DataFrame(
            {
                "Number of Neighbours": num_neighbours,
                "RMSE Value": rmse_values
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