import streamlit as st
import pandas as pd
from Home import DataPage
from data_loader import DataLoader
from nearest_neighbours import NearestNeighbours

class NearestNeighboursPage:
    def run(self):
        st.title("Nearest Neighbours")
        dl = DataLoader()
        dl.load_data(DataPage().get_data_selection())
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        knn = NearestNeighbours(self.X_train, self.X_test, self.y_train, self.y_test)
        knn.train_model()
    
        star_rating, distance = self.get_new_hotel_fields()
        new_price = knn.predict_price(star_rating, distance)
        st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')

        num_neighbours = list(range(1, len(self.y_test)))
        rmse_values = knn.calculate_rmse_values(num_neighbours)
        self.display_rmse_chart(num_neighbours, rmse_values)
        best_k, best_rmse = knn.find_best_k()
        self.display_best_k_and_rmse(best_k, best_rmse)

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

    def display_rmse_chart(self, num_neighbours, rmse_values):
        """
        Displays num_neighboours vs rmse_values chart
        """
        st.subheader('Root Mean Squared Error')
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